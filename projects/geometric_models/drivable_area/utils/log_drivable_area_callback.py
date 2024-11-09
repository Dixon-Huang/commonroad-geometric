from typing import List

import torch
import wandb
from torch import Tensor

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, EarlyStoppingCallbacksParams, StepCallbackParams
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
from projects.geometric_models.drivable_area.utils.visualization.plotting import create_drivable_area_prediction_image


class LogDrivableAreaWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self, 
        wandb_service: WandbService, 
        target_attribute: str = "occupancy"
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.target_attribute = target_attribute

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[Tensor] = []
        try:
            time_step, prediction = params.output[1], params.output[2]
        except IndexError:
            return  # TODO, trajectory prediction model
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        try:
            prediction = prediction[self.target_attribute]
        except KeyError:
            return
        target = params.batch.vehicle[self.target_attribute]
        assert prediction.size(0) == params.batch.v[self.target_attribute].size(0)

        prediction = prediction.view(prediction.size(0), prediction_size, prediction_size)
        prediction_rgb = create_drivable_area_prediction_image(prediction).numpy()
        # torch.randint(0, prediction.size(0), size=(12,)):
        for idx in range(min(30, target.size(0))):
            prediction_img = prediction_rgb[idx].astype(int)
            drivable_area_img = (target[idx] * 255).type(torch.uint8).cpu().numpy().astype(int)
            images += [
                # wandb.data_types.Image(batch.road_coverage[idx], mode="L", caption="actual"),
                wandb.data_types.Image(drivable_area_img, mode="L", caption="actual"),
                wandb.data_types.Image(prediction_img, mode="RGB", caption="predicted"),
            ]
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(figsize=(6, 3.2), ncols=2)
            # axes[0].set_title('true')
            # axes[0].imshow(drivable_area_img)
            # axes[1].set_title('predicted')
            # axes[1].imshow(prediction_img)
            # #ax.set_aspect('equal')
            # plt.show()

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )

        return


class LogVelocityFlowWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self, 
        wandb_service: WandbService,
        occupancy_attribute: str = "occupancy",
        target_attribute: str = "occupancy_flow"
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.occupancy_attribute = occupancy_attribute
        self.target_attribute = target_attribute

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[Tensor] = []
        try:
            time_step, prediction = params.output[1], params.output[2]
        except IndexError:
            return  # TODO, trajectory prediction model
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        try:
            prediction = prediction[self.target_attribute]
        except KeyError:
            return
        
        target = params.batch.vehicle[self.target_attribute]
        assert prediction.size(0) == target.size(0)

        prediction = prediction.view(prediction.size(0), prediction_size, prediction_size, 2)
        # torch.randint(0, prediction.size(0), size=(12,)):
        for idx in range(min(30, target.size(0))):

            occupancy_mask = (params.batch.vehicle[self.occupancy_attribute][idx, ...] == 0).int()

            true_relative_velocity_0 = target[idx, :, :, 0]
            true_relative_velocity_1 = target[idx, :, :, 1]
            true_relative_velocity_rb_image = torch.clip(0.5 + torch.stack([true_relative_velocity_0, true_relative_velocity_1], dim=-1)/10, 0.0, 1.0)
            target_img = torch.cat([true_relative_velocity_rb_image, torch.ones_like(occupancy_mask[:, :, None])], dim=-1)
            target_img_np = (target_img.detach().cpu().numpy()*255).astype(int)

            pred_relative_velocity_0 = prediction[idx, :, :, 0]
            pred_relative_velocity_1 = prediction[idx, :, :, 1]
            pred_relative_velocity_rb_image = torch.clip(0.5 + torch.stack([pred_relative_velocity_0, pred_relative_velocity_1], dim=-1)/10, 0.0, 1.0)
            pred_img = torch.cat([pred_relative_velocity_rb_image, torch.ones_like(occupancy_mask[:, :, None])], dim=-1)
            pred_img_np = (pred_img.detach().cpu().numpy()*255).astype(int)

            images += [
                # wandb.data_types.Image(batch.road_coverage[idx], mode="L", caption="actual"),
                wandb.data_types.Image(target_img_np, mode="RGB", caption="actual"),
                wandb.data_types.Image(pred_img_np, mode="RGB", caption="predicted"),
            ]
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(figsize=(6, 3.2), ncols=2)
            # axes[0].set_title('true')
            # axes[0].imshow(target_img_np)
            # axes[1].set_title('predicted')
            # axes[1].imshow(pred_img_np)
            # #ax.set_aspect('equal')
            # plt.show()

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )

        return
