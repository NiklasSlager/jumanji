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

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.environments.distillation.env import Distillation, State
#from jumanji.environments.distillation.types import Position
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture(scope="module")
def distillation() -> Distillation:
    """Instantiates a default Snake environment."""
    return Distillation()


def test_distillation__reset(distillation: Distillation) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(chex.assert_max_traces(distillation.reset, n=1))
    state1, timestep1 = reset_fn(jax.random.PRNGKey(1))
    state2, timestep2 = reset_fn(jax.random.PRNGKey(2))
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0
    assert state1.length == 1
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert state1.n_value != state2.n_value
    #assert state1.fruit_position != state2.fruit_position
    assert not jnp.array_equal(state1.key, state2.key)
    assert not jnp.array_equal(state1.key, state2.key)


def test_distillation__step(distillation: Distillation) -> None:
    """Validates the jitted step function of the environment."""
    step_fn = jax.jit(chex.assert_max_traces(distillation.step, n=1))
    state_key, action_key = jax.random.split(jax.random.PRNGKey(0))
    state, timestep = distillation.reset(state_key)
    # Sample two different actions
    action1, action2 = jax.random.choice(
        action_key,
        jnp.arange(distillation.action_spec._num_values),
        shape=(2,),
        replace=False,
    )
    state1, timestep1 = step_fn(state, action1)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check that the state has changed
    assert state1.step_count != state.step_count
    assert state1.n_value != state.n_value
    # Check that two different actions lead to two different states
    state2, timestep2 = step_fn(state, action2)
    assert state1.n_value != state2.n_value
    # Check that the state update and timestep creation work as expected



def test_distillation__does_not_smoke(distillation: Distillation) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(distillation)


def test_distillation__specs_does_not_smoke(distillation: Distillation) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(distillation)
'''

def test_snake__no_nan(snake: Snake) -> None:
    """Validates that no nan is encountered in either the state or the observation throughout an
    episode. Checks both exiting from the top and right of the board as jax out-of-bounds indices
    have different behaviors if positive or negative.
    """
    reset_fn = jax.jit(snake.reset)
    step_fn = jax.jit(snake.step)
    key = jax.random.PRNGKey(0)
    # Check exiting the board to the top
    state, timestep = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep = step_fn(state, action=0)
        chex.assert_tree_all_finite((state, timestep))
    # Check exiting the board to the right
    state, timestep = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep = step_fn(state, action=1)
        chex.assert_tree_all_finite((state, timestep))
'''
