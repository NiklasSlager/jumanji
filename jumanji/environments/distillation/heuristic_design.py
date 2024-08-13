import jax.numpy as jnp
import numpy as np
from time import time
from jax import jit, vmap
import jax

import matplotlib.pyplot as plt
from jumanji.environments.distillation.env import Distillation
from jumanji.environments.distillation.types import State
from jumanji.environments.distillation.NR_model_test.plot_generation import plot_function
from jumanji.environments.distillation.NR_model_test.NR_model import inside_simulation as simulation
from jumanji.environments.distillation.NR_model_test.NR_model import initialize

import os
import subprocess
import warnings

env = Distillation()
x = jnp.array([0, 99])
x_d = jnp.array([0, 149])
x_n = jnp.array([0, 80])
x_f = jnp.array([0, 29])

n_bound = jnp.array([5, 85])
rr_bound = jnp.array([0.1, 10])
distillate_bound = jnp.array([0.01, 0.99])
feed_bound = jnp.array([1.2, 3])



n = jnp.array([38, 53, 16, 54, 18, 65, 46, 77, 51])
reflux = jnp.array([1.44, 3.77, 1.47, 5.50, 3.33, 5.22, 3.79, 5.66, 3.18])
distillate = jnp.array([0.558, 0.722, 0.646, 0.524, 0.363, 0.296, 0.340, 0.454, 0.610])
feed = jnp.array([21, 32, 11, 32, 12, 37, 24, 43, 28])
feed = n/feed

'''
n = jnp.array([49, 76, 44, 65, 36, 53, 16, 53, 18])
reflux = jnp.array([0.97, 2.14, 1.5, 2.54, 1.89, 3.77, 1.48, 5.51, 3.32])
distillate = jnp.array([0.955, 0.928, 0.895, 0.867, 0.810, 0.722, 0.646, 0.523, 0.363])
feed = jnp.array([28, 43, 24, 37, 23, 32, 9, 32, 12])
feed = n/feed

n = jnp.array([20, 12, 9, 13, 30])
reflux = jnp.array([0.71, 1.97, 2.87, 0.48, 1.69])
distillate = jnp.array([0.523, 0.544, 0.333, 0.498, 0.598])

n = jnp.array([27, 12, 16, 12, 9])
reflux = jnp.array([0.90, 0.77, 1.70, 1.96, 2.86])
distillate = jnp.array([0.903, 0.843, 0.685, 0.544, 0.333])

n = jnp.array([12, 12, 16, 10, 30])
reflux = jnp.array([0.82, 0.82, 1.02, 0.48, 1.69])
distillate = jnp.array([0.095, 0.209, 0.332, 0.498, 0.598])
'''
n_value = jnp.int32(jnp.round(jnp.interp(n, n_bound, x_n)))
rr_value = jnp.int32(jnp.round(jnp.interp(reflux, rr_bound, x)))
distillate_value = jnp.int32(jnp.round(jnp.interp(distillate, distillate_bound, x_d)))
feed_value = jnp.int32(jnp.round(jnp.interp(feed, feed_bound, x_f)))


state, timestep = env.reset(jax.random.PRNGKey(0))

'''
state, timestep = env.step(state, jnp.array([n_value[0], rr_value[0], distillate_value[0], feed_value[0]]))
state, timestep = env.step(state, jnp.array([n_value[1], rr_value[1], distillate_value[1], feed_value[1]]))
state, timestep = env.step(state, jnp.array([n_value[2], rr_value[2], distillate_value[2], feed_value[2]]))
state, timestep = env.step(state, jnp.array([n_value[3], rr_value[3], distillate_value[3], feed_value[3]]))
'''

def flowchart(carry, i):
    state, action_n, action_r, action_d, action_f = carry
    state, timestep = jax.jit(env.step)(state, jnp.array([action_n[i], action_r[i], action_d[i], action_f[i]]))
    carry = state, action_n, action_r, action_d, action_f
    return carry, None

def for_body(carry, i):
    action_mask, product_mask, row_repeat, row_count, x_column, streams = carry
    product = jnp.concatenate((jnp.array(0)[None], jnp.diff(jnp.sum(product_mask, axis=0))))


    column_repeat = jnp.arange(len(row_repeat))
    correction = jnp.concatenate(
        (jnp.array(0)[None], jnp.cumsum(jnp.diff(column_repeat)-jnp.diff(x_column))))
    #x_column = jnp.where((row_count[i] == row_count[i-1]) | (product[i-1] == 2), column_repeat - jnp.sum(column_repeat-x_column), row_repeat)

    row_check = jnp.concatenate((jnp.array(0)[None], 2 ** jnp.arange(len(action_mask) - 1)))
    row_repeat = row_repeat.at[i+1].set(jnp.where(i>0, jnp.where(((row_count[i] == row_count[i - 1])),
                                            row_repeat[i], row_repeat[i+1]), row_repeat[i+1]))

    x_column = x_column.at[i].set(jnp.where((row_count[i] == row_count[i - 1]) | (jnp.cumsum(product)[i] >= row_check[i]) | (jnp.sum(action_mask, axis=1)[i-1] == 1),
                         column_repeat[i] - correction[i-1], row_repeat[i]))
    #top_action = jnp.max(jnp.concatenate((jnp.zeros(len(action_mask))[:, None], -jnp.diff(action_mask)), axis=1), axis=0)
    top_action = jnp.where(i == 0, False, action_mask[row_count[i], i]-action_mask[row_count[i], i-1] == 0)
    level_change = jnp.where(top_action, 1, 0)
    row_level = jnp.where(jnp.arange(len(row_count)) > i, level_change, 0)
    streams = streams.at[row_count[i]].set(streams[row_count[i]]+row_level)
    #row_level = jnp.where((row_level == 0) & (streams[i] > 1), streams[i+1], row_level)
    streams = streams.at[i+1].set(jnp.where(streams[i+1] > 0, streams[row_count[i]], streams[i+1]))
    carry = action_mask, product_mask, row_repeat, row_count, x_column, streams
    return carry, None


def render(state: State):
    actions = jnp.sum(jnp.int32(state.overall_stream_actions))
    min_products = jnp.sum(jnp.where(state.stream.flows[0,0]>0, 1, 0))
    action_mask = jnp.int32(state.overall_stream_actions)[:min_products, :min_products]
    product_mask = state.stream.isproduct[:min_products, :min_products]

    row_count = jnp.max(jnp.tile(jnp.arange(len(action_mask)), (len(action_mask), 1)).transpose() * action_mask, axis=0)
    row_check = jnp.concatenate((jnp.array(0)[None], 2 ** jnp.arange(len(action_mask) - 1)))
    row_repeat = jnp.repeat(row_check[:-1], row_check[1:])[:len(action_mask)]

    initial_streams = jnp.triu(jnp.ones_like(action_mask)).at[0, 0].set(0)
    x_column = jnp.arange(len(action_mask))
    carry = action_mask, product_mask, row_repeat, row_count, x_column, initial_streams

    for i in jnp.arange(len(action_mask)):
        carry, _ = for_body(carry, i)
    #carry, _ = jax.lax.scan(for_body, carry, jnp.arange(len(action_mask)))
    _, _, _, row_count, x_column, stream_levels = carry

    products = jnp.sum(jnp.diff(product_mask), axis=1)

    v = jnp.asarray(jnp.where(jnp.diff(x_column) - jnp.diff(row_count) == 1, 1, 0), dtype=float)
    v = jnp.where(v[-1] == 1, v.at[-1].set(0.5), v)

    u = jnp.ones_like(v, dtype=float).at[-1].set(0.5)  # jnp.abs(jnp.diff(x_column))


    y_start = jnp.max(jnp.where(action_mask, stream_levels, 0), axis=0)[1:]


    y_side = jnp.concatenate((jnp.array(1)[None], y_start[:-1]+v[:-1]))
    y_side = jnp.repeat(y_side, 2)
    x_side = jnp.concatenate((jnp.array(1)[None], x_column[1:-1]+u[:-1]))
    x_side = jnp.repeat(x_side, 2)

    rows, count = jnp.unique(jnp.stack((x_column[1:], y_start)), axis=1, return_counts=True)
    repeats = jnp.repeat(count, count)
    v_end = jnp.ravel(jnp.column_stack((jnp.roll(v, 0), jnp.roll(v, 0))))

    repeats = jnp.concatenate((jnp.repeat(repeats, 3 - repeats), jnp.ones(jnp.int32(2*len(x_column[1:])))))[:2*len(x_column[1:])]
    v_up = jnp.where(jnp.int32(v_end) & (repeats == 1), 1, 0)

    #repeats = jnp.where(x_side == jnp.max(x_side), 1, 0) | (repeats-v_end == 1)
    side_v = jnp.tile(jnp.array([0,1]), jnp.int32(len(y_side)/2))/2
    side_u = jnp.ones_like(side_v) - 0.5
    side_u = jnp.where((x_side == jnp.max(x_side)), side_u, side_v)
    side_u = jnp.where(repeats==1, side_u, 0)
    side_v = jnp.where((repeats==1) & (v_up == 0), side_v, 0)
    #side_v = jnp.where(repeats, side_v, 0)
    color_switch = jnp.zeros(len(side_v)).at[jnp.nonzero(jnp.where((side_v > 0) | (side_u > 0), 1, 0))].set(jnp.roll(products,1))
    colors = np.where(color_switch, 'green', 'red')

    plt.figure()
    plt.tick_params(labelleft=False)

    plt.quiver(x_side, y_side, side_u*1.5, side_v*1.5, color=colors, angles='xy', scale_units='xy', scale=1)
    plt.quiver(x_column[1:-1], y_start[:-1], u[:-1], v[:-1], angles='xy', scale_units='xy', scale=1)
    plt.scatter(x_column[1:-1], y_start[:-1],  linewidth=4)
    plt.quiver(0, 1, 1, 0, angles='xy', scale_units='xy',scale=1)
    plt.xlim([0, min_products])
    plt.ylim([0, min_products])

    #plt.grid(True, axis='x')
    plt.show()
    return 1


init_carry = state, n_value, rr_value, distillate_value, feed_value
carry, _ = jax.lax.scan(flowchart, init_carry, jnp.arange(9))
state, _, _, _, _ = carry
#state, timestep = env.step(state, jnp.array([n_value[-1], rr_value[-1], distillate_value[-1], feed_value[-1]]))
print(jnp.sum(state.stream.value))
render(state)


