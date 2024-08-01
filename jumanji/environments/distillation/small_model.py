from jumanji.environments.distillation.types import State
import jax.numpy as jnp


def simulation(input: float):
    return -((input-5.)**2)