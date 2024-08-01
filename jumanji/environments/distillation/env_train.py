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

from functools import cached_property
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
# from jumanji.environments.distillation.types import Observation, State
# from jumanji.environments.distillation.small_model import simulation
from jumanji.environments.distillation.NR_model_test.distillation_types import State as ColumnState
from jumanji.environments.distillation.types import Observation, State, StreamSpec, ColumnInputSpecification
from jumanji.environments.distillation.NR_model_test.NR_model import inside_simulation as simulation
from jumanji.environments.distillation.NR_model_test.NR_model import initialize
from jumanji.types import TimeStep, restart, termination, transition


class Distillation(Environment[State, specs.DiscreteArray, Observation]):

    def __init__(
            self,
            stage_bound: Tuple[int, int] = (5, 80),
            pressure_bound: Tuple[float, float] = (1, 10),
            reflux_bound: Tuple[float, float] = (0.01, 20),
            distillate_bound: Tuple[float, float] = (0.01, 1e3),
            step_limit: int = 30,
    ):
        """Instantiates a `Snake` environment.

        Args:
            num_rows: number of rows of the 2D grid. Defaults to 12.
            num_cols: number of columns of the 2D grid. Defaults to 12.
            time_limit: time_limit of an episode, i.e. number of environment steps before
                the episode ends. Defaults to 4000.
            viewer: `Viewer` used for rendering. Defaults to `SnakeViewer`.
        """
        self._stage_bounds = stage_bound
        self._pressure_bounds = pressure_bound
        self._reflux_bounds = reflux_bound
        self._distillate_bounds = distillate_bound
        self._max_steps = step_limit

        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random key used to sample the snake and fruit positions.

        Returns:
             state: `State` object corresponding to the new state of the environment.
             timestep: `TimeStep` object corresponding to the first timestep returned by the
                environment.
        """

        feed = jnp.array([0.2, 0.3, 0.5], dtype=float) * jnp.array(1000., dtype=float)
        
        stream = self._stream_table_reset(len(feed))
        stream = stream.replace(flows=stream.flows.at[0, 0].set(feed))
        
        state = State(
            stream=stream,
            step_count=jnp.ones((), dtype=int),
            action_mask_stream=jnp.zeros((len(feed), len(feed)), dtype=bool).at[0, 0].set(True),
            action_mask_column=jnp.ones(4.5e5, dtype=bool),
            key=jax.random.PRNGKey(0)  # (2,)
        )
        # output = simulation(N_reset)
        reward = jnp.zeros((), dtype=float)
        timestep = restart(observation=self._state_to_observation(state), extras={"reward": reward, "stages": column_input.n_stages, "pressure": column_input.pressure, "reflux": column_input.reflux_ratio})
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        
        key, N_key = jax.random.split(state.key, 2)
        #new_N = action+self.min_N

        column_input = self._action_to_column_spec(action)

        
        feed_flow = state.stream.flows[state.action_mask_stream.transpose()]
        feed = jnp.sum(feed_flow)
        z = feed_flow/jnp.sum(feed_flow)
        
        init_column = initialize()
        column_state, iterator, res = jax.jit(simulation)(
            state=init_column,
            nstages=jnp.int32(column_input.n_stages),
            pressure=column_input.pressure,
            feed=feed,
            z=z[0],
            distillate=column_input.distillate/jnp.sum(state.stream.flows[0, 0])*feed,
            rr=column_input.reflux_ratio
        )
        
        
        next_state = self._stream_table_update(state, column_state)
        next_state = self._get_action_mask_stream(next_state)
        reward = jnp.sum(next_state.stream.value[:, next_state.step_count])
        next_state = next_state.replace(
            step_count=state.step_count + 1,
            key=key,
        )

        done = (next_state.step_count >= self._max_steps)

        

       
        observation = self._state_to_observation(next_state)

        extras = {"reward": reward, "stages": column_input.n_stages, "pressure": column_input.pressure, "reflux": column_input.reflux_ratio}
        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
        )
        return next_state, timestep


    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (float) of shape (num_rows, num_cols, 5).
            - step_count: DiscreteArray (num_values = time_limit) of shape ().
            - action_mask: BoundedArray (bool) of shape (4,).
        """


        grid = specs.BoundedArray(
            shape=(3,),
            minimum=10.,
            maximum=1000.,
            dtype=float,
            name="grid",
        )
        step_count = specs.DiscreteArray(
            self._max_steps, dtype=jnp.int32, name="step_count"
        )
        action_mask = specs.BoundedArray(
            shape=(4.5e5, ),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            step_count=step_count,
            action_mask=action_mask,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        continuous_spec = specs.BoundedArray(shape=(5,), dtype=float, minimum=-1, maximum=1,
                                             name="action_continuous")
        discrete_spec = specs.DiscreteArray(num_values=4.5e5, name="action_discrete")
        return discrete_spec


    def _continuous_action_to_column_spec(self, action) -> ColumnInputSpecification:
        """All actions are assumed to be bounded between -1 and 1, and we then translate these into
        the relevant values for the ColumnInputSpecification."""

        n_stages = jnp.round(jnp.interp(action[0], jnp.array([-1, 1]), jnp.array(self._stage_bounds)) + 0.5)
        reflux_ratio = jnp.interp(action[1], jnp.array([-1, 1]), jnp.array(self._reflux_bounds))
        distillate = jnp.interp(action[2], jnp.array([-1, 1]), jnp.array(self._distillate_bounds))
        pressure = jnp.interp(action[3], jnp.array([-1, 1]), jnp.array(self._pressure_bounds))
        column_spec = ColumnInputSpecification(
            n_stages=n_stages,
            reflux_ratio=reflux_ratio,
            distillate=distillate,
            pressure=pressure
        )

        return column_spec


    def _action_to_column_spec(self, action: chex.Array):
        action_N = action % 75
        action_P = jnp.floor(action % 10)
        action_RR = jnp.floor(action % 30)
        action_D = jnp.floor(action % 20)

        new_N = jnp.int32(jnp.interp(action_N, jnp.array([0, 74]), jnp.array(self._stage_bounds)))
        new_P = jnp.interp(action_P, jnp.array([0, 10]), jnp.array(self._pressure_bounds))
        new_RR = jnp.interp(action_RR, jnp.array([0, 30]), jnp.array(self._reflux_bounds))
        new_D = jnp.interp(action_D, jnp.array([0, 20]), jnp.array(self._distillate_bounds))
        column_spec = ColumnInputSpecification(
            n_stages=new_N,
            reflux_ratio=new_RR,
            distillate=new_D,
            pressure=new_P
        )


    def _state_to_observation(self, state: State) -> Observation:
        '''
        grid = jnp.concatenate(jax.tree_util.tree_map(
                lambda x: x[..., None], [state.Nstages, state.CD, state.RD, state.TAC]),
            axis=-1,
            dtype=float,
        )
        '''

        flows = state.stream.flows[state.action_mask_stream][0]
        return Observation(
            grid=jnp.array([flows]),
            step_count=state.step_count,
            action_mask=state.action_mask_column,
        )


    def _stream_table_reset(self, matsize):
        nr_of_streams = jnp.int32((matsize / 2 + 0.5) * matsize)
        feed = jnp.zeros((matsize, matsize, matsize), dtype=float)

        row = jnp.arange(1, matsize + 1)
        stream_nr = (jnp.triu(
            jnp.tile(jnp.arange(matsize).transpose(),
                     (matsize, 1)).transpose()) + 0.5 * row ** 2 - 0.5 * row + 1) * jnp.triu(
            jnp.ones((matsize, matsize)))
        value = jnp.zeros_like(stream_nr)
        return StreamSpec(
            flows=feed,
            nr=stream_nr,
            value=value,
            isproduct=jnp.zeros((matsize, matsize)),
            reflux=jnp.zeros((matsize, matsize)),
            pressure=jnp.zeros((matsize, matsize)),
            stages=jnp.zeros((matsize, matsize))
        )

    
    def _is_product_stream(self, mole_flow: chex.Array):
        return (jnp.max(mole_flow/jnp.sum(mole_flow)) > 0.95) | (jnp.sum(mole_flow) < 10)


    def _stream_table_update(self, state: State, column_state: ColumnState):

        product_prices = (jnp.arange(len(column_state.components)) * 1.2 + 5)/150
        
        state = state.replace(stream=state.stream.replace(
            flows=state.stream.flows.at[state.step_count, :].set(state.stream.flows[state.step_count - 1, :])))
        indices = jnp.where(
            (state.stream.isproduct[:, state.step_count] == 0) & (state.stream.nr[:, state.step_count] > 0),
            state.stream.nr[:, state.step_count],
            jnp.max(state.stream.nr[:, state.step_count]) - 0.5
        ) - state.stream.nr[:, state.step_count][0]

        bot_flow = column_state.X[:, column_state.Nstages - 1] * (jnp.sum(column_state.F) - column_state.distillate)
        top_flow = column_state.Y[:, 0] * jnp.sum(column_state.distillate)
        bot_flow_isproduct = self._is_product_stream(bot_flow)
        top_flow_isproduct = self._is_product_stream(top_flow)
        stream_table = state.stream.replace(
            flows=state.stream.flows.at[
                state.step_count, [jnp.int32(jnp.min(indices)), jnp.int32(jnp.max(indices))]].set(
                jnp.array([top_flow, bot_flow])),
            value=state.stream.value.at[
                [jnp.int32(jnp.min(indices)), jnp.int32(jnp.max(indices))], state.step_count].set(
                -column_state.TAC / jnp.sum(column_state.F) * jnp.sum(jnp.array([top_flow, bot_flow]), axis=1) +
               jnp.array([top_flow_isproduct, bot_flow_isproduct]) * jnp.sum(jnp.array([top_flow, bot_flow]) * product_prices, axis=1),
            ),
            isproduct=state.stream.isproduct.at[
                jnp.int32(jnp.min(indices)), state.step_count:].set([top_flow_isproduct]).at[
                      jnp.int32(jnp.max(indices)), state.step_count:].set([bot_flow_isproduct])
            
        )

        return state.replace(
            stream=stream_table,
        )

    def _get_action_mask_stream(self, state: State):
        step_mask = jnp.where((state.stream.isproduct[:, state.step_count] == 0)
                              & (jnp.triu(jnp.ones(state.action_mask_stream.shape, dtype=bool))[:, state.step_count]),
                              jnp.arange(1, len(state.stream.flows) + 1),
                              10)
        return state.replace(
            action_mask_stream=jnp.zeros_like(state.action_mask_stream).at[:, state.step_count].set(
                jnp.array(jnp.where(step_mask == jnp.min(step_mask), 1, 0), dtype=bool))
        )

        
