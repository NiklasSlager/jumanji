import jax.numpy as jnp
import jax
from jax import vmap
from jumanji.environments.distillation.NR_model_test.distillation_types import FUG_state, State
from jumanji.environments.distillation.NR_model_test import functions, thermodynamics, NR_model
from jaxopt import Bisection


def initialize(z, heavy_spec, light_spec, pressure, feed):
    return FUG_state(
        components=jnp.arange(len(z)),
        z=jnp.array(z, dtype=float),
        pressure=jnp.array(pressure, dtype=float),
        heavy_key=jnp.zeros((), dtype=int),
        light_key=jnp.zeros((), dtype=int),
        heavy_x=jnp.zeros(len(z), dtype=float),
        light_x=jnp.zeros(len(z), dtype=float),
        heavy_spec=jnp.array(heavy_spec, dtype=float),
        light_spec=jnp.array(light_spec, dtype=float),
        alpha_avg=jnp.zeros(len(z), dtype=float),
        feed=jnp.array(feed, dtype=float),
        stages=jnp.zeros((), dtype=int),
        feed_stage=jnp.zeros((), dtype=int),
        reflux=jnp.zeros((), dtype=float),
        t_cond=jnp.zeros((), dtype=float),
        t_reb=jnp.zeros((), dtype=float),
        distillate=jnp.zeros((), dtype=float)
    )

def end_tray_t_solver(state: FUG_state, temperature, xc):
    def function(tempcalc):
        return 1 - jnp.sum(vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(tempcalc, state.components, state.pressure) * xc)
    result = Bisection(optimality_fun=function, lower=500., upper=100., check_bracket=False).run(temperature).params
    return result


def minimum_reflux(state: FUG_state):
    nr_components = jnp.sum(jnp.where(state.z > 0, 1, 0))
    components = jnp.where(state.z > 0, jnp.arange(len(state.z)), 0)
    temperature = jnp.array(500., dtype=float)
    k_eq = vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(temperature, state.components, state.pressure).transpose()
    k_eq = jnp.where(state.z>0, k_eq, 1e10)
    non_heavy_key = jnp.where(k_eq != jnp.min(k_eq), components, 0)
    light_key = jnp.max(jnp.where((state.z == jnp.max(state.z)) & (components == non_heavy_key),
                          jnp.arange(len(components)), jnp.max(jnp.where(state.z == jnp.max(state.z[non_heavy_key]), non_heavy_key, 0))))
    heavy_key = jnp.int32(jnp.min(jnp.where(k_eq[light_key]>k_eq, components, 1e2)))

    light_x = jnp.where((k_eq[light_key] <= k_eq) & (components > 0), 1, 0) * state.z
    light_x = light_x / jnp.sum(light_x)

    heavy_x = jnp.where((k_eq[light_key] > k_eq) & (components > 0), 1, 0) * state.z
    heavy_x = heavy_x / jnp.sum(heavy_x)

    distillate = jnp.sum(jnp.where((k_eq > k_eq[light_key]) & (components > 0), state.z, 0)*state.feed) + state.z[light_key]*state.feed*state.light_spec + state.z[heavy_key]*state.feed*(1-state.light_spec)

    #carry = state, heavy_x, light_x, jnp.zeros(len(state.components), dtype=float), jnp.array(0, dtype=int), heavy_key, light_key
    state = state._replace(
        heavy_x=heavy_x,
        light_x=light_x,
        heavy_key=heavy_key,
        light_key=light_key,
        distillate=distillate,
    )
    state, iterations = body((state, jnp.array(0, dtype=int)))
    #state, iterations = jax.lax.while_loop(condition, body, (state, jnp.array(0, dtype=int)))


    alpha_light_key = jnp.sum(jnp.where(state.components == state.light_key, state.alpha_avg, 0))

    def underwood(theta):
        return jnp.sum(state.alpha_avg * state.z / (state.alpha_avg - theta))

    theta = Bisection(optimality_fun=underwood, lower=(1.0001), upper=alpha_light_key - 0.001,
                             check_bracket=False).run((alpha_light_key + 1)/2).params

    rr_min = jnp.sum(state.alpha_avg * state.light_x / (state.alpha_avg - theta)) - 1
    return state._replace(reflux=jnp.where(rr_min > 0.01, rr_min*1.3, 0.01))


def condition(carry):
    state, iterations = carry
    return iterations < 5


def body(carry):
    state, iterations = carry
    iterations += 1
    temperature = jnp.array(500., dtype=float)
    t_cond = end_tray_t_solver(state,  temperature, state.light_x)
    t_reb = end_tray_t_solver(state, temperature, state.heavy_x)

    K_cond = vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(t_cond, state.components, state.pressure).transpose()
    alpha_cond = K_cond / jnp.sum(jnp.where(state.components == state.heavy_key, K_cond, 0))
    K_reb = vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(t_reb, state.components, state.pressure).transpose()
    alpha_reb = K_reb / jnp.sum(jnp.where(state.components == state.heavy_key, K_reb, 0))
    alpha_avg = jnp.sqrt(alpha_cond * alpha_reb)
    alpha_lightkey = jnp.sum(jnp.where(state.components == state.light_key, alpha_avg, 0))
    f_hk = jnp.sum(jnp.where(state.components == state.heavy_key, state.z, 0) * state.feed)
    f_lk = jnp.sum(jnp.where(state.components == state.light_key, state.z, 0) * state.feed)
    d_lk = jnp.array(state.light_spec * state.feed * state.z[state.light_key], dtype=float)
    b_hk = jnp.array(state.heavy_spec * state.feed * state.z[state.heavy_key], dtype=float)

    A = jnp.log10((1 - b_hk / f_hk) / (b_hk / f_hk))
    B = jnp.log10(((d_lk / f_lk) / (1 - d_lk / f_lk) * (b_hk / f_hk) / (1 - b_hk / f_hk))) / jnp.log10(alpha_lightkey)

    d_i = jnp.nan_to_num(state.z * state.feed * (10 ** A * alpha_avg ** B / (1 + 10 ** A * alpha_avg ** B)))
    b_i = jnp.nan_to_num(state.z * state.feed - d_i)
    state = state._replace(
        heavy_x=b_i / jnp.sum(b_i),
        light_x=d_i / jnp.sum(d_i),
        alpha_avg=alpha_avg,
        t_cond=t_cond,
        t_reb=t_reb,
    )
    carry = state, iterations
    return carry


def minimum_stages(state: FUG_state):
    n_min = (jnp.log10(
        (state.light_x[state.light_key]/state.light_x[state.heavy_key]) *
        (state.heavy_x[state.heavy_key]/state.heavy_x[state.light_key])) /
             jnp.log10(state.alpha_avg[state.light_key]))
    return state._replace(stages=n_min)

def stagenumber(state: FUG_state):
    X = (state.reflux - state.reflux/1.3)/(state.reflux + 1)
    Y = 1 - jnp.exp(((1+54.4*X)/(11+117.2*X))*((X-1)/X**0.5))
    return state._replace(stages=jnp.int32(jnp.ceil((-state.stages - 1)/(Y-1))))


def feedstage(state: FUG_state):
    ratio = ((state.feed-state.distillate)/state.distillate * state.z[state.heavy_key]/state.z[state.light_key] * (state.heavy_x[state.light_key]/state.light_x[state.heavy_key])**2)**0.2
    return state._replace(feed_stage=jnp.int32(jnp.ceil(state.stages*ratio/(1+ratio)+0.45)))


def FUG(z, heavy_spec, light_spec, pressure, feed):
    state = initialize(z, heavy_spec, light_spec, pressure, feed)
    state = minimum_reflux(state)
    state = minimum_stages(state)
    state = stagenumber(state)
    state = feedstage(state)
    return state

def FUG_specs(z, heavy_spec, light_spec, pressure, feed):
    state = initialize(z, heavy_spec, light_spec, pressure, feed)
    state = minimum_reflux(state)
    return state