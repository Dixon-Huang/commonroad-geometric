from typing import Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch_geometric.data import Batch, Data

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.common.torch_utils.sampling import sample_indices
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.geometric_models.drivable_area.utils.confusion_matrix import BaseConfusionMatrix
from projects.geometric_models.drivable_area.models.decoder.common import UpsampleConv
from projects.geometric_models.drivable_area.models.decoder.transformer_decoder import TransformerDecoder

class SelfAttention(nn.Module):
    """ A simple self-attention layer for images """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

class DrivableAreaDecoder(BaseModel):

    def __init__(
        self,
        cfg: Config,
        target_attribute: str = "occupancy"
    ):
        super().__init__()
        self.cfg = cfg
        self.target_attribute = target_attribute
    
        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type == "ConvTranspose":
            # 2d transposed convolution decoder
            assert self.cfg.prediction_size == 64
            self.decoder = nn.Sequential(
                # 1 x 1 x node_feature_size
                # SelfAttention(in_channels=self.cfg.node_feature_size),
                nn.ConvTranspose2d(
                    in_channels=self.cfg.node_feature_size, 
                    out_channels=128,
                    kernel_size=4, stride=1, padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 4 x 4 x 128
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 8 x 8 x 64
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # 16 x 16 x 32
                nn.ConvTranspose2d(
                    in_channels=32, out_channels=16,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                # 32 x 32 x 16
                nn.ConvTranspose2d(
                    in_channels=16, out_channels=1,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.Sigmoid(),
                # 64 x 64 x 1
            )

        elif self.cfg.decoder_type == "Upsample Conv":
            # https://distill.pub/2016/deconv-checkerboard/
            self.decoder = nn.Sequential(
                # 1 x 1 x node_feature_size
                nn.Conv2d(
                    in_channels=self.cfg.node_feature_size, out_channels=256,
                    kernel_size=1, stride=1, padding=0,
                    bias=False,
                ),
                # 1 x 1 x 256
                UpsampleConv(out_size=4, in_channels=256, out_channels=128),
                # 4 x 4 x 128
                UpsampleConv(out_size=8, in_channels=128, out_channels=64),
                # 8 x 8 x 64
                UpsampleConv(out_size=16, in_channels=64, out_channels=32),
                # 16 x 16 x 32
                UpsampleConv(out_size=32, in_channels=32, out_channels=16),
                # 32 x 32 x 16
                UpsampleConv(out_size=64, in_channels=16, out_channels=1),
                nn.Sigmoid(),
                # 64 x 64 x 1
            )

        elif self.cfg.decoder_type == "Transformer":
            self.decoder = TransformerDecoder(
                node_feature_size=self.cfg.node_feature_size,
                sigmoid_out=True
            )

        elif self.cfg.decoder_type == "MLP":
            # MLP decoder
            self.decoder = nn.Sequential(
                nn.Linear(self.cfg.node_feature_size, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Linear(512, self.cfg.prediction_size ** 2, bias=False),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(f"Unknown decoder_type config value: {self.cfg.decoder_type}")

        # self.loss_weights: Optional[Tensor] = None
        # if self.cfg.reweight_loss is not False:
        #     # re-weight loss to prevent overfitting to straight road
        #     # loss outside the typical straight road is weighted more
        #     loss_weights = torch.ones(
        #         (self.cfg.prediction_size, self.cfg.prediction_size),
        #         dtype=torch.float,
        #         requires_grad=False,
        #     )
        #     loss_weights[:, 21:43] = self.cfg.reweight_loss
        #     self.loss_weights = loss_weights.view(1, -1)

    @overload
    def forward(self, data: CommonRoadData, x: Tensor, sampling_weights: Optional[Tensor] = None) -> Tensor:
        ...

    @overload
    def forward(self, data: CommonRoadData, x: Tensor,
                sampling_weights: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        ...

    def forward(self, data: CommonRoadData, x: Union[Tensor, Tuple[Union[Data, Batch], Tensor]],
                sampling_weights: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        N = x.size(0)
        assert_size(x, (N, self.cfg.node_feature_size))

        # only compute a prediction for a subset of samples during training
        sample_ind = None
        if sampling_weights is not None:
            try:
                sample_ind = sample_indices(sampling_weights, num_samples=self.cfg.training_resample_ratio)
            except RuntimeError:
                pass
            else:
                x = x[sample_ind]
                N = sample_ind.size(0)
                assert_size(x, (N, self.cfg.node_feature_size))

        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type in {"ConvTranspose", "Upsample Conv"}:
            x = x.view(N, self.cfg.node_feature_size, 1, 1)

        prediction = self.decoder(x)
        # prediction in [0, 1]

        if self.cfg.decoder_type == "MLP":
            assert_size(prediction, (N, self.cfg.prediction_size ** 2))
        else:
            assert_size(prediction, (N, 1, self.cfg.prediction_size, self.cfg.prediction_size))
            prediction = prediction.view(N, -1)

        # import matplotlib.pyplot as plt
        # if not '_plot_counter' in self.__dict__:
        #     self._plot_counter = 0
        # if self._plot_counter % 50 == 0:
        #     # indices = torch.where(data.v.id == -1)[0].tolist()
        #     indices = list(range(min(6, data.v.num_nodes)))
        #     try:
        #         drivable_area = data.v[self.target_attribute]
        #     except AttributeError:
        #         drivable_area = None
        #     prediction_size = self.cfg.prediction_size
        #     fig, axes = plt.subplots(nrows=len(indices), ncols=2, figsize=(12, 16))
        #     for i, idx in enumerate(indices):
        #         # 'Raw' image subplot
        #         if drivable_area is not None:
        #             axes[i, 0].imshow(drivable_area[idx, :, :].detach().cpu().numpy())
        #         try:
        #             batch = data.v.batch[idx].item()
        #         except AttributeError:
        #             batch = 0
        #         axes[i, 0].set_title('Raw ' + f"{data.scenario_id if isinstance(data.scenario_id, str) else data.scenario_id[batch]} ({data.time_step if isinstance(data.time_step, int) else data.time_step[batch]})")
        #         axes[i, 0].axis('off')  # Turn off axis labels
        #         # 'Predicted' image subplot
        #         axes[i, 1].imshow(prediction[idx].view((prediction_size, prediction_size)).detach().cpu().numpy())
        #         axes[i, 1].set_title('Predicted ' + f"{self.training=}")
        #         axes[i, 1].axis('off')  # Turn off axis labels
        #     plt.tight_layout()
        #     plt.show()
        # self._plot_counter += 1

        # shape N x (SIZE ** 2)
        # return (prediction, sample_ind) if self.cfg.use_sampling_weights else 
        return prediction

    def compute_loss(self, data: Union[CommonRoadData, CommonRoadDataTemporal], prediction: Tensor, only_ego: bool = False) -> Tensor:
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
            drivable_area = data.v[self.target_attribute].index_select(
                0, sample_ind) if sample_ind is not None else data.v[self.target_attribute]
        else:
            drivable_area = data.v[self.target_attribute]

        if only_ego:
            is_ego_mask = data.v.is_ego_mask.bool().squeeze(-1)
            drivable_area = drivable_area[is_ego_mask, :]

        N = prediction.shape[0]
        prediction_size = self.cfg.prediction_size
        drivable_area = drivable_area.view(N, prediction_size, prediction_size)
        assert_size(prediction, (N, prediction_size ** 2))

        vehicle_type_filter = (data.v.type == 0).squeeze(-1)
        if only_ego:
            vehicle_type_filter = torch.ones(N, dtype=torch.bool, device=drivable_area.device)
        
        # enable this for debugging
        # import matplotlib.pyplot as plt
        # indices = torch.where(vehicle_type_filter)[0][:5]  # Get first 5 indices that meet the condition
        # # Create a figure with 5 rows and 2 columns
        # fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))
        # for i, idx in enumerate(indices):
        #     # 'Raw' image subplot
        #     axes[i, 0].imshow(drivable_area[idx, :, :].detach().cpu().numpy())
        #     axes[i, 0].set_title('Raw')
        #     axes[i, 0].axis('off')  # Turn off axis labels
        #     # 'Predicted' image subplot
        #     axes[i, 1].imshow(prediction[idx].view((prediction_size, prediction_size)).detach().cpu().numpy())
        #     axes[i, 1].set_title('Predicted')
        #     axes[i, 1].axis('off')  # Turn off axis labels
        # plt.tight_layout()
        # plt.show()

        # Bird's eye view binary drivable area
        true_drivable_area = drivable_area.view(N, prediction_size ** 2)
        loss = F.binary_cross_entropy(input=prediction[vehicle_type_filter], target=true_drivable_area[vehicle_type_filter], reduction="mean")
        return loss

        # if self.loss_weights is not None:
        #     if self.loss_weights.device != prediction.device:
        #         self.loss_weights = self.loss_weights.to(prediction.device)
        #     loss = loss * self.loss_weights
        # return loss.mean()

    @staticmethod
    def thresholded_drivable_area(
        drivable_area: Tensor,
        prediction: Tensor,
    ) -> Tuple[BoolTensor, BoolTensor]:
        N, prediction_size, _ = drivable_area.size()
        assert_size(drivable_area, (N, prediction_size, prediction_size))
        assert_size(prediction, (N, prediction_size ** 2))
        true_drivable_area = drivable_area.view(-1, prediction_size ** 2)
        true_drivable_area_tresh = true_drivable_area > 0.5
        prediction_thresh = prediction > 0.5
        return true_drivable_area_tresh, prediction_thresh

    def compute_binary_loss(
        self,
        data: Union[CommonRoadData, CommonRoadDataTemporal],
        prediction: Tensor
    ) -> Tensor:
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
            drivable_area = data.v[self.target_attribute].index_select(
                0, sample_ind) if sample_ind is not None else data.v[self.target_attribute]
        else:
            drivable_area = data.v[self.target_attribute]
        N = prediction.shape[0]
        prediction_size = self.cfg.prediction_size

        drivable_area = drivable_area.view(N, prediction_size, prediction_size)

        true_drivable_area_thresh, prediction_thresh = DrivableAreaDecoder.thresholded_drivable_area(
            drivable_area=drivable_area, prediction=prediction,
        )
        return F.binary_cross_entropy(
            input=prediction_thresh.type(torch.float32),
            target=true_drivable_area_thresh.type(torch.float32),
            reduction="mean",
        )

    @staticmethod
    def thresholded_prediction_confusion_matrix(
        drivable_area: Tensor,
        prediction: Tensor,
    ) -> BaseConfusionMatrix:
        true_drivable_area_thresh, prediction_thresh = DrivableAreaDecoder.thresholded_drivable_area(
            drivable_area=drivable_area, prediction=prediction,
        )
        actual_pos, actual_neg = true_drivable_area_thresh, ~true_drivable_area_thresh
        pred_pos, pred_neg = prediction_thresh, ~prediction_thresh
        return BaseConfusionMatrix(
            positive=actual_pos.sum(dim=-1).cpu(),
            predicted_positive=pred_pos.sum(dim=-1).cpu(),
            negative=actual_neg.sum(dim=-1).cpu(),
            predicted_negative=pred_neg.sum(dim=-1).cpu(),
            true_positive=pred_pos[actual_pos].sum(dim=-1).cpu(),
            true_negative=pred_neg[actual_neg].sum(dim=-1).cpu(),
            false_positive=pred_pos[actual_neg].sum(dim=-1).cpu(),
            false_negative=pred_neg[actual_pos].sum(dim=-1).cpu(),
        )

