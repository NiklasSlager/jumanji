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
from jumanji.environments.distillation.NR_model_test.NR_homotopy import inside_simulation as simulation
from jumanji.environments.distillation.NR_model_test.NR_homotopy import initialize as initialize_column
from jumanji.types import TimeStep, restart, termination, transition


class Distillation(Environment[State, specs.DiscreteArray, Observation]):

    def __init__(
            self,
            stage_bound: Tuple[int, int] = (5, 85),
            pressure_bound: Tuple[float, float] = (1, 10),
            reflux_bound: Tuple[float, float] = (0.1, 10.),
            distillate_bound: Tuple[float, float] = (0.010, 0.990),
            feed_bound: Tuple[float, float] = (1.2, 3.),
            step_limit: int = 9,

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
        self._feed_bounds = feed_bound
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

        feed = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        feed = feed/jnp.sum(feed) * jnp.array(2000., dtype=float)
        stream = self._stream_table_reset(self._max_steps+1, len(feed))
        stream = stream.replace(flows=stream.flows.at[0, 0].set(feed))
        action_mask = jnp.array(
            (jnp.concatenate((jnp.ones(81, dtype=bool), jnp.zeros(69, dtype=bool))),
             #jnp.concatenate((jnp.ones(65, dtype=bool), jnp.zeros(35, dtype=bool))),
             jnp.concatenate((jnp.ones(100, dtype=bool), jnp.zeros(50, dtype=bool))),
             jnp.ones(150,dtype=bool),
             jnp.concatenate((jnp.ones(30, dtype=bool), jnp.zeros(120, dtype=bool)))
             ))
        state = State(
            stream=stream,
            step_count=jnp.zeros((), dtype=int),
            column_count=jnp.zeros((), dtype=int),
            action_mask_stream=jnp.zeros((self._max_steps+1, self._max_steps+1), dtype=bool).at[0, 0].set(True),
            overall_stream_actions=jnp.zeros((self._max_steps + 1, self._max_steps + 1), dtype=bool).at[0, 0].set(True),
            action_mask_column=action_mask,
            key=jax.random.PRNGKey(0)  # (2,)
        )
        # output = simulation(N_reset)
        reward = jnp.array(-50, dtype=float)

        timestep = restart(observation=self._state_to_observation(state),
                           extras={"reward": reward,
                                   "stages_C1": jnp.zeros_like(state.stream.stages[0,0]), "stages_C2": jnp.zeros_like(state.stream.stages[0,0]),
                                   "pressure_C1": jnp.zeros_like(state.stream.stages[0,0]), "pressure_C2": jnp.zeros_like(state.stream.stages[0,0]),
                                   "reflux_C1": jnp.zeros_like(state.stream.stages[0,0]), "reflux_C2": jnp.zeros_like(state.stream.stages[0,0]),
                                   "distillate_C1": jnp.zeros_like(state.stream.stages[0,0]),
                                   "distillate_C1.1": state.stream.flows[0, 1, 0],
                                   "distillate_C1.2": state.stream.flows[0, 1, 1],
                                   "distillate_C1.3": state.stream.flows[0, 1, 2],
                                   "bottom_C2": jnp.zeros_like(state.stream.stages[0, 0]),
                                   "bottom_C2.1": state.stream.flows[0, 1, 0],
                                   "bottom_C2.2": state.stream.flows[0, 1, 1],
                                   "bottom_C2.3": state.stream.flows[0, 1, 2],
                                   "action_C1.1": state.stream.flows[0, 1, 0],
                                   "action_C1.2": state.stream.flows[0, 1, 1],
                                   "action_C1.3": state.stream.flows[0, 1, 2],
                                   "action_C2.1": state.stream.flows[-1, 2, 0],
                                   "action_C2.2": state.stream.flows[-1, 2, 1],
                                   "action_C2.3": state.stream.flows[-1, 2, 2],
                                   "nr_product_streams": jnp.sum(jnp.max(state.stream.isproduct, axis=1)),
                                   "nr_columns": jnp.sum(state.overall_stream_actions),
                                   "converged": jnp.zeros((), dtype=bool),
                                   "outflow": jnp.zeros((), dtype=float),
                                   #"action_mask": self._matrix_to_binary_integer(jnp.concatenate(jnp.int32(state.overall_stream_actions))),
                                   #"product_mask": self._matrix_to_binary_integer(jnp.concatenate(state.stream.isproduct)),
                                   })

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        key, N_key = jax.random.split(state.key, 2)
        # new_N = action+self.min_N

        column_input = self._action_to_column_spec(action)
        feed_flow = state.stream.flows[:, state.column_count]*state.action_mask_stream[:, state.column_count][:, None]
        feed_flow = jnp.sum(feed_flow, axis=0)
        feed = jnp.sum(feed_flow)
        z = feed_flow / jnp.sum(feed_flow)

        init_column = initialize_column()


        column_state = jax.jit(simulation)(
            state=init_column,
            nstages=jnp.int32(column_input.n_stages),
            feedstage=column_input.feed_stage,
            pressure=column_input.pressure,
            feed=feed,
            z=z,
            distillate=column_input.distillate * feed,
            rr=column_input.reflux_ratio
        )
        column_state = column_state.replace(converged = jnp.nan_to_num(column_state.converged))

        next_state = self._stream_table_update(state, column_state, action)
        next_state = self._get_action_mask_stream(next_state)

        converged = jnp.asarray(((jnp.sum(column_state.V[0]) > 0) & (column_state.converged == 1)), dtype=int)


        next_state = next_state.replace(
            step_count=state.step_count + 1,
            column_count=state.column_count + converged,
            overall_stream_actions=state.overall_stream_actions+next_state.action_mask_stream,
            key=N_key,
        )
        reward = jnp.sum(next_state.stream.value[:, next_state.column_count])
        done = (next_state.step_count >= self._max_steps) | (jnp.max(state.action_mask_stream) == 0)

        observation = self._state_to_observation(next_state)

        #x_column, y_column, products, level_step = self._get_flowchart_configuration(state)
        
        extras = {"reward": reward,
                  "stages_C1": next_state.stream.stages[0, 1], "stages_C2": next_state.stream.stages[2, 2],
                  "pressure_C1": next_state.stream.pressure[0, 1], "pressure_C2": next_state.stream.pressure[2, 2],
                  "reflux_C1": next_state.stream.reflux[0, 1], "reflux_C2": next_state.stream.reflux[2, 2],
                  "distillate_C1": jnp.sum(next_state.stream.flows[0, 1]),
                  "distillate_C1.1": next_state.stream.flows[0, 1, 0],
                  "distillate_C1.2": next_state.stream.flows[0, 1, 1],
                  "distillate_C1.3": next_state.stream.flows[0, 1, 2],
                  "bottom_C2": jnp.sum(next_state.stream.flows[2, 2]),
                  "bottom_C2.1": next_state.stream.flows[-1, -1, 0],
                  "bottom_C2.2": next_state.stream.flows[-1, -1, 1],
                  "bottom_C2.3": next_state.stream.flows[-1, -1, 2],
                  "action_C1.1": next_state.stream.action[0, 1, 0],
                  "action_C1.2": next_state.stream.action[0, 1, 1],
                  "action_C1.3": next_state.stream.action[0, 1, 2],
                  "action_C2.1": next_state.stream.action[-1, -1, 0],
                  "action_C2.2": next_state.stream.action[-1, -1, 1],
                  "action_C2.3": next_state.stream.action[-1, -1, 2],
                  "nr_product_streams": jnp.sum(jnp.max(next_state.stream.isproduct, axis=1)),
                  "nr_columns": jnp.sum(state.overall_stream_actions),
                  "converged": column_state.converged,
                  "outflow": jnp.sum(jnp.max(next_state.stream.isproduct*jnp.sum(next_state.stream.flows, axis=2), axis=1)),
                  #"action_mask": self._matrix_to_binary_integer(jnp.concatenate(jnp.int32(state.overall_stream_actions))),
                  #"product_mask": self._matrix_to_binary_integer(jnp.concatenate(state.stream.isproduct)),
                  }

                  
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
            shape=(11,),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="grid",
        )
        step_count = specs.DiscreteArray(
            self._max_steps, dtype=jnp.int32, name="step_count"
        )
        action_mask = specs.BoundedArray(
            shape=(4, 150),
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
        discrete_spec = specs.MultiDiscreteArray(
            num_values=jnp.array([150] * 4, jnp.int32),
            dtype=jnp.int32,
            name="action_discrete")
        return discrete_spec

    def _continuous_action_to_column_spec(self, action) -> ColumnInputSpecification:
        """All actions are assumed to be bounded between -1 and 1, and we then translate these into
        the relevant values for the ColumnInputSpecification."""

        n_stages = jnp.round(jnp.interp(action[0], jnp.array([-1, 1]), jnp.array(self._stage_bounds)) + 0.5)
        reflux_ratio = jnp.interp(action[1], jnp.array([-1, 1]), jnp.array(self._reflux_bounds))
        distillate = jnp.interp(action[2], jnp.array([-1, 1]), jnp.array(self._distillate_bounds))
        pressure = jnp.interp(action[3], jnp.array([-1, 1]), jnp.array(self._pressure_bounds))
        column_spec = ColumnInputSpecification(
            n_stages=n_stages[0],
            reflux_ratio=reflux_ratio[0],
            distillate=distillate[0],
            pressure=pressure[0]
        )

        return column_spec

    def _action_to_column_spec(self, action: chex.Array):
        action_N, action_RR, action_D, action_F = action
        #action_N = action // 2500
        #action_RR = action % 2500 // 50
        #action_D = action % 50

        new_N = jnp.int32(jnp.interp(action_N, jnp.array([0, 80]), jnp.array(self._stage_bounds)))
        #new_P = jnp.interp(action_P, jnp.array([0, 10]), jnp.array(self._pressure_bounds))
        new_RR = jnp.interp(action_RR, jnp.array([0, 99]), jnp.array(self._reflux_bounds))
        new_D = jnp.interp(action_D, jnp.array([0, 149]), jnp.array(self._distillate_bounds))
        new_F = jnp.interp(action_F, jnp.array([0, 29]), jnp.array(self._feed_bounds))

        return ColumnInputSpecification(
            n_stages=new_N,
            reflux_ratio=new_RR,
            distillate=new_D,
            pressure=jnp.array(1., dtype=float),
            feed_stage=jnp.array(new_N/new_F, dtype=int)
        )

    def _state_to_observation(self, state: State) -> Observation:
        '''
        grid = jnp.concatenate(jax.tree_util.tree_map(
                lambda x: x[..., None], [state.Nstages, state.CD, state.RD, state.TAC]),
            axis=-1,
            dtype=float,
        )
        '''
        
        flows = jnp.sum(state.stream.flows[:, state.column_count]*state.action_mask_stream[:, state.column_count][:, None], axis=0)

        return Observation(
            grid=jnp.nan_to_num(jnp.concatenate((flows/jnp.sum(flows), jnp.array([jnp.sum(flows)/jnp.array(2000., dtype=float)])))),
            step_count=state.step_count,
            action_mask=state.action_mask_column,
        )

    def _stream_table_reset(self, matsize, feedsize):
        nr_of_streams = jnp.int32((matsize / 2 + 0.5) * matsize)
        feed = jnp.zeros((matsize, matsize, feedsize), dtype=float)

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
            stages=jnp.zeros((matsize, matsize)),
            converged=jnp.zeros((matsize, matsize)),
            action=jnp.zeros((matsize, matsize, 4)),
            iterations=jnp.zeros((matsize, matsize)),
        )


    def _is_product_stream(self, mole_flow: chex.Array, converged: chex.Scalar):
        return ((jnp.max(mole_flow / jnp.sum(mole_flow)) > 0.95) | (jnp.sum(mole_flow) < 10)) & (converged == True)


    def _stream_table_update(self, state: State, column_state: ColumnState, action: chex.Array):
        product_prices = 0.10
        converged = jnp.asarray((jnp.nan_to_num(jnp.sum(column_state.V[0]))>0) & (column_state.converged==1))
        step = state.column_count - (1-converged) + 1
        state = state.replace(stream=state.stream.replace(
            flows=state.stream.flows.at[:, step].set(state.stream.flows[:, state.column_count]),
            isproduct=state.stream.isproduct.at[:, step].set(state.stream.isproduct[:, state.column_count])))
        indices = jnp.where(
            (state.stream.isproduct[:, state.column_count] == 0) & (state.stream.nr[:, state.column_count+1] > 0),
            state.stream.nr[:, state.column_count+1],
            jnp.max(state.stream.nr[:, state.column_count+1]) - 0.5
        ) - state.stream.nr[:, state.column_count+1][0]

        bot_flow = jnp.nan_to_num(column_state.X[:, column_state.Nstages - 1] * (jnp.sum(column_state.F) - column_state.V[0]))
        #bot_flow = jnp.where(bot_flow <= jnp.array([200, 300, 500]), bot_flow, -10)
        top_flow = jnp.nan_to_num(column_state.Y[:, 0] * jnp.sum(column_state.V[0]))
        #top_flow = jnp.where(top_flow <= jnp.array([200, 300, 500]), top_flow, -10)

        bot_flow_isproduct = self._is_product_stream(bot_flow, converged)
        top_flow_isproduct = self._is_product_stream(top_flow, converged)
        feedflows = state.stream.flows[(jnp.int32(jnp.min(indices)), jnp.int32(jnp.max(indices))), state.column_count-1]
        real_flows = jnp.where(converged == True, jnp.array((top_flow, bot_flow)), feedflows)
        column_cost = jnp.nan_to_num(-column_state.TAC / jnp.sum(column_state.F))
        column_cost = jnp.where(jnp.abs(column_cost) > 75, -75/jnp.array(1000.), column_cost)
        stream_table = state.stream.replace(
            flows=state.stream.flows.at[
                (jnp.int32(jnp.min(indices)), jnp.int32(jnp.max(indices))), step].set(
                real_flows
            ),
            value=state.stream.value.at[
                (jnp.int32(jnp.min(indices)), jnp.int32(jnp.max(indices))), step].set(
                column_cost * jnp.sum(real_flows, axis=1) +
                jnp.sum(real_flows * product_prices, axis=1) * jnp.array([top_flow_isproduct, bot_flow_isproduct])
            ),
            isproduct=state.stream.isproduct.at[
                      jnp.int32(jnp.min(indices)), step].set(top_flow_isproduct).at[
                      jnp.int32(jnp.max(indices)), step].set(bot_flow_isproduct),
            reflux=state.stream.reflux.at[
                      jnp.int32(jnp.min(indices)), step].set(column_state.RR).at[
                      jnp.int32(jnp.max(indices)), step].set(column_state.RR),
            pressure=state.stream.pressure.at[
                      jnp.int32(jnp.min(indices)), step].set(column_state.pressure).at[
                      jnp.int32(jnp.max(indices)), step].set(column_state.pressure),
            stages=state.stream.stages.at[
                      jnp.int32(jnp.min(indices)), step].set(column_state.Nstages).at[
                      jnp.int32(jnp.max(indices)), step].set(column_state.Nstages),
            converged=state.stream.converged.at[
                      jnp.int32(jnp.min(indices)), step].set(converged).at[
                      jnp.int32(jnp.max(indices)), step].set(converged),
            action=state.stream.action.at[
                jnp.int32(jnp.min(indices)), step].set(action).at[
                jnp.int32(jnp.max(indices)), step].set(action),
            iterations=state.stream.iterations.at[
                jnp.int32(jnp.min(indices)), step].set(column_state.NR_iterations).at[
                jnp.int32(jnp.max(indices)), step].set(column_state.NR_iterations),

        )
        
        return state.replace(
            stream=stream_table,
        )

    def _get_action_mask_stream(self, state: State):
        step = state.column_count - jnp.array((1-jnp.max(state.stream.converged[:, state.column_count+1])), dtype=int) + 1
        step_mask = jnp.where((state.stream.isproduct[:, step] == 0)
                              & (jnp.triu(jnp.ones(state.action_mask_stream.shape, dtype=bool))[:, step]),
                              jnp.arange(1, len(state.stream.flows) + 1),
                              12)
        return state.replace(
            action_mask_stream=jnp.zeros_like(state.action_mask_stream).at[:, step].set(
                jnp.array(jnp.where((step_mask == jnp.min(step_mask)) & (jnp.min(step_mask) !=12), 1, 0), dtype=bool))
        )


    def _get_flowchart_configuration(self, state: State):
        action_mask = state.overall_stream_actions
        product_mask = state.stream.isproduct

        row_count = jnp.tile(jnp.arange(len(action_mask)), (len(action_mask), 1)).transpose() * action_mask
        row_check = jnp.concatenate((jnp.array(0)[None], 2 ** jnp.arange(len(action_mask) - 1)))
        row_repeat = jnp.repeat(row_check[:-1], row_check[1:])[:len(action_mask)]
        initial_streams = jnp.triu(jnp.ones_like(action_mask)).at[0, 0].set(0)
        x_column = jnp.arange(len(action_mask))
        carry = action_mask, product_mask, row_repeat, jnp.max(row_count, axis=0), x_column, initial_streams
        carry, _ = jax.lax.scan(self._stream_level_scan, carry, jnp.arange(len(action_mask)))
        action_mask, product_mask, row_repeat, row_count, x_column, stream_levels = carry
        
        y_column = jnp.max(jnp.where(action_mask, stream_levels, 0), axis=0)[1:]
        products = product_mask[x_column, jnp.max(jnp.where(action_mask, stream_levels, 0), axis=0)]
        level_step = jnp.asarray(jnp.where(jnp.diff(x_column) - jnp.diff(row_count) == 1, 1, 0), dtype=float)
        return x_column, y_column, products, level_step


    def _stream_level_scan(self, carry, i):
        action_mask, product_mask, row_repeat, row_count, x_column, streams = carry
        product = jnp.concatenate((jnp.array(0)[None], jnp.diff(jnp.sum(product_mask, axis=0))))

        column_repeat = jnp.arange(len(row_repeat))
        correction = jnp.concatenate(
            (jnp.array(0)[None], jnp.cumsum(jnp.diff(column_repeat) - jnp.diff(x_column))))
        # x_column = jnp.where((row_count[i] == row_count[i-1]) | (product[i-1] == 2), column_repeat - jnp.sum(column_repeat-x_column), row_repeat)

        row_repeat = row_repeat.at[i + 1].set(
            jnp.where(((row_count[i] == row_count[i - 1]) | (product_mask[i - 1, i] == 1)),
                      row_repeat[i], row_repeat[i + 1]))
        x_column = x_column.at[i].set(jnp.where((row_count[i] == row_count[i - 1]) | (product_mask[i - 1, i] == 1),
                                                column_repeat[i] - correction[i - 1], row_repeat[i]))
        top_action = jnp.where(i == 0, False, action_mask[row_count[i], i] - action_mask[row_count[i], i - 1] == 0)
        level_change = jnp.where(top_action, 1, 0)
        row_level = jnp.where(jnp.arange(len(row_count)) > i, level_change, 0)
        streams = streams.at[row_count[i]].set(streams[row_count[i]] + row_level)
        streams = streams.at[i + 1].set(jnp.where(streams[i + 1] > 0, streams[row_count[i]], streams[i + 1]))
        carry = action_mask, product_mask, row_repeat, row_count, x_column, streams
        return carry, None
