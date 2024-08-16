from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
import chex





class Tray(NamedTuple):
    l: chex.Array
    v: chex.Array
    T: chex.Array


class Mesh(NamedTuple):
    H: jnp.ndarray
    M: jnp.ndarray
    E: jnp.ndarray


class Derivatives(NamedTuple):
    H: Tray
    M: Tray
    E: Tray


class Thermo(NamedTuple):
    psat_params: chex.Array
    hvap_params: chex.Array
    hform_params: chex.Array
    cpvap_params: chex.Array


class FUG_state(NamedTuple):
    components: chex.Array
    z: chex.Array
    pressure: chex.Array
    heavy_key: chex.Array
    light_key: chex.Array
    heavy_x: chex.Array
    light_x: chex.Array
    heavy_spec: chex.Array
    light_spec: chex.Array
    alpha_avg: chex.Array
    feed: chex.Array
    stages: chex.Array
    feed_stage: chex.Array
    reflux: chex.Array
    t_cond: chex.Array
    t_reb: chex.Array
    distillate: chex.Array
    
    
@dataclass
class State:
    L: chex.Array
    V: chex.Array
    U: chex.Array
    W: chex.Array
    X: chex.Array
    Y: chex.Array
    temperature: chex.Array
    temperature_bounds: chex.Array
    F: chex.Array
    components: chex.Array
    z: chex.Array
    pressure: chex.Array
    RR: chex.Array
    distillate: chex.Array
    Nstages: chex.Array
    Hliq: chex.Array
    Hvap: chex.Array
    Hfeed: chex.Array
    RD: chex.Array
    CD: chex.Array
    TAC: chex.Array
    heavy_key: chex.Array
    light_key: chex.Array
    heavy_spec: chex.Array
    light_spec: chex.Array
    NR_residuals: chex.Array
    EQU_residuals: chex.Array
    analytics: bool
    converged: bool
    NR_iterations: chex.Array
    EQU_iterations: chex.Array
    BP_iterations: chex.Array
