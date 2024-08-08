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
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

@dataclass
class StreamSpec:
    flows: chex.Array
    nr: chex.Array
    isproduct: chex.Array
    value: chex.Array
    reflux: chex.Array
    pressure: chex.Array
    stages: chex.Array
    converged: chex.Array
    action: chex.Array
    iterations: chex.Array
    
@dataclass
class State:
    stream: StreamSpec
    step_count: chex.Numeric  # ()
    column_count: chex.Numeric
    action_mask_stream: chex.Array  # (4,)
    overall_stream_actions: chex.Array
    action_mask_column: chex.Array
    key: chex.PRNGKey  # (2,)
    

class ColumnInputSpecification(NamedTuple):
    n_stages: chex.Scalar
    reflux_ratio: chex.Scalar
    distillate: chex.Scalar
    pressure: chex.Scalar


class Observation(NamedTuple):
    """
    grid: feature maps that include information about the fruit, the snake head, its body and tail.
    step_count: current number of steps in the episode.
    action_mask: array specifying which directions the snake can move in from its current position.
    """

    grid: chex.Array  # (num_rows, num_cols, 5)
    step_count: chex.Numeric  # Shape ()
    action_mask: chex.Array  # (4,)
