from typing import List, Optional, Tuple, Type

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import Actor, BaseFeaturesExtractor, TD3Policy, BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.utils import get_action_dim


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """

    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # Example architecture with dropout before the last layer:
        # (Modify according to your needs)
        net_arch = kwargs.get("net_arch", [400, 300])
        dropout_rate = kwargs.get("dropout_rate", 0.2)

        layers = []
        for i in range(len(net_arch)):
            layers.append(
                nn.Linear(net_arch[i - 1] if i > 0 else self.features_dim, net_arch[i])
            )
            layers.append(nn.ReLU())
            if (
                i < len(net_arch) - 1
            ):  # typically, you don't apply dropout right before the output layer
                layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(net_arch[-1], self.action_space.shape[0]))
        layers.append(nn.Tanh())  # Output is squashed via tanh

        self.mu = nn.Sequential(*layers)


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            q_net = nn.Sequential(...)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


# To register a policy, so you can use a string to create the network
# TD3.policy_aliases["CustomTD3Policy"] = CustomTD3Policy
