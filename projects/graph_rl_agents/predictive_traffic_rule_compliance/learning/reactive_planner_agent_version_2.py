import time
import torch as th
import numpy as np
from collections import deque
from typing import Any, ClassVar, Dict, Type, TypeVar, Union, Tuple
from stable_baselines3.common import utils
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import (
    VecEnv,
    unwrap_vec_normalize,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
)
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.actor.reactive_planner_actor import ReactivePlannerActor

import torch.nn.functional as F

from gymnasium import spaces

from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.cost_function.RobustnessCostFunction import RobustnessCostFunction

SelfReactivePlannerAgent = TypeVar("SelfReactivePlannerAgent", bound="ReactivePlannerAgent")


class ReactivePlannerAgent(OnPolicyAlgorithm):
    """
    Custom Agent using Reactive Planner for action selection.
    This agent replaces the Actor network with a Reactive Planner.
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        tensorboard_log: str = None,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: int = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        actor_options: Dict[str, Any] = None,
    ):
        super(ReactivePlannerAgent, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self._ego_vehicle_simulations = None
        self.reactive_actor = ReactivePlannerActor(actor_options)

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._ego_vehicle_simulations is not None, "No ego vehicle simulation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # actions, values, log_probs = self.policy(obs_tensor)
                _, values, log_probs = self.policy(obs_tensor)
            # actions = actions.cpu().numpy()

            # Change: 从reactive_planner_actor中获取actions
            actions = self.reactive_actor.step(self._ego_vehicle_simulations, self.policy)

            new_obs, rewards, dones, infos = env.step(actions)

            # Change 从info中获取ego_vehicle_simulation
            ego_vehicle_simulations = [info.get('ego_vehicle_simulation') for info in infos]
            self._ego_vehicle_simulations = ego_vehicle_simulations
            infos = remove_ego_vehicle_simulation(infos)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        value_losses = []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values = rollout_data.returns
                values_pred = self.policy.predict_values(rollout_data.observations).flatten()

                value_loss = F.mse_loss(values_pred, values)
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfReactivePlannerAgent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ReactivePlannerAgent",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfReactivePlannerAgent:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]

            # Change: 从info中获取ego_vehicle_simulation
            self._ego_vehicle_simulations = [info.get('ego_vehicle_simulation') for info in self.env.reset_infos]
            self.env.rest_infos = {}

            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None
            self._ego_vehicle_simulations = None

        self.n_envs = env.num_envs
        self.env = env

# def remove_ego_vehicle_simulation(infos):
#     return [{k: v for k, v in info.items() if k != 'ego_vehicle_simulation'} for info in infos]

def remove_ego_vehicle_simulation(infos):
    for info in infos:
        if 'ego_vehicle_simulation' in info:
            del info['ego_vehicle_simulation']
    return infos