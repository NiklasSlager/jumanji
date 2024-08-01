# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jumanji.environments.distillation import Observation, Distillation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_distillation(
        distillation: Distillation,
        num_channels: int,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Snake` environment."""
    num_values = distillation.action_spec.num_values
    num_agents = num_values.shape[0]
    num_actions = num_values[0]
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network(
        num_agents=num_agents,
        num_actions=num_actions,
        mlp_units=policy_layers,
        conv_n_channels=num_channels,
        time_limit=distillation._max_steps,
    )
    value_network = make_critic_network(
        mlp_units=value_layers,
        num_conv_channels=num_channels,
        time_limit=distillation._max_steps,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_distillation_cnn(
        num_outputs: int,
        mlp_units: Sequence[int],
        conv_n_channels: int,
        time_limit: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        '''
        torso = hk.Sequential(
            [
                hk.Dense(32),
                jax.nn.relu,
                hk.Dense(32),
                jax.nn.relu,
            ]
        )
        embedding = torso(observation.grid)
        norm_step_count = jnp.expand_dims(observation.step_count / time_limit, axis=-1)
        embedding = jnp.concatenate([embedding, norm_step_count], axis=-1)
        '''

        embedding = observation.grid

        head = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        if num_outputs == 1:
            value = jnp.squeeze(head(embedding), axis=-1)
            return value
        else:
            logits = head(embedding)
            logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )
            return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_network(
        num_agents: int,
        num_actions: int,
        conv_n_channels: Sequence[int],
        mlp_units: Sequence[int],
        time_limit: int
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Shapes: B: batch size, N: number of agents, W: grid width, H: grid height
        obs = observation.grid
        embedding = jnp.tile(obs, num_agents).reshape(obs.shape[0], num_agents, obs.shape[-1])  # (B, N, W * H)

        num_agents_parallel = embedding
        normalised_step_count = jnp.repeat(
            jnp.expand_dims(observation.step_count, axis=(1, 2)) / time_limit,
            num_agents,
            axis=1,
        )  # (B, N, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )  # (B, N, W*H+1)
        head = hk.nets.MLP((*mlp_units, 100), activate_final=False)
        logits = head(output)  # (B, N, 4)
        return jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network(
        num_conv_channels: Sequence[int], mlp_units: Sequence[int], time_limit: int
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        embedding = observation.grid  # (B, W * H)
        normalised_step_count = (
                jnp.expand_dims(observation.step_count, axis=-1) / time_limit
        )  # (B, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )  # (B, W*H+1)
        values = hk.nets.MLP((*mlp_units, 1), activate_final=False)(output)  # (B, 1)
        return jnp.squeeze(values, axis=-1)  # (B,)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)