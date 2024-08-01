import jax.numpy as jnp
from jaxopt import Bisection
from jax import vmap
from jumanji.environments.distillation.NR_model_test.distillation_types import State, Thermo
import os
from jumanji.environments.distillation.NR_model_test.physical_data_sets import psat_params, cpvap_params, hvap_params, hform_params

'''
def retrieve_params(dir):
    params_psat_list = psat_params(dir)
    params_cpvap_list = cpvap_params(dir)
    params_hvap_list = hvap_params(dir)
    hform_list = hform_data(dir)
    return params_psat_list, params_cpvap_list, params_hvap_list, hform_list

def psat_params(dir):
    params_psat = []
    with jnp.open(os.path.join(dir, 'Antoine.csv'), 'r') as fs:
        next(fs)  # skip header row
        for line in fs:
            fields = line.strip().split(',')
            params_psat.append(list(map(float, fields[1:])))
    return jnp.array(params_psat, dtype=float)


def cpvap_params(dir):
    params_cpvap = []
    with open(os.path.join(dir, 'Vapor_Cp.csv'), 'r') as fs:
        next(fs)  # skip header row
        for line in fs:
            fields = line.strip().split(',')
            params_cpvap.append(list(map(float, fields[1:])))

    return jnp.array(params_cpvap, dtype=float)


def hvap_params(dir):
    params_hvap = []
    with open(os.path.join(dir, 'Heat_of_evaporation.csv'), 'r') as fs:
        next(fs)  # skip header row
        for line in fs:
            fields = line.strip().split(',')
            params_hvap.append(list(map(float, fields[1:])))
    return jnp.array(params_hvap, dtype=float)


def hform_data(dir):
    hform = []
    with open(os.path.join(dir, 'Heat_of_formation.csv'), 'r') as fs:
        next(fs)  # skip header row
        for line in fs:
            fields = line.strip().split(',')
            hform.append(fields[1])

    hform = jnp.array(hform, float) * 4.184
    return hform
'''


def p_sat(temp, compound):
    #dir = os.path.join(os.getcwd(), 'Pure component parameters')
    params = psat_params()[compound]
    a, b, c, d, e, f, g, h, i = params
    return jnp.exp(a + b / (temp + c) + d * temp + e * jnp.log(temp) + f * temp ** g)


def t_sat(compound, tguess, pressure):
    def equality(tempcalc, compound, pressure):
        return 1 - p_sat(tempcalc, compound) / pressure
    return Bisection(optimality_fun=equality, lower=500., upper=100., check_bracket=False).run(tguess, compound, pressure).params


def k_eq(temp, component, pressure):

    result = p_sat(temp, component)/pressure
    return result

def cp_vap(temperature, compound):
    params = cpvap_params()[compound]
    a, b, c, d, e, f, g = params

    def cp(temp):
        return a*temp + b*c/(jnp.tanh(c/temp))-d*e*jnp.tanh(e/temp)

    return (cp(temperature)-cp(298.15)) / 1000 * 4.184


def h_evap(temperature, compound):
    params = hvap_params()[compound]
    a, b, c, d, e, f, g = params

    tc = g + 273.15
    tr = temperature / tc
    return jnp.where(tr < 1, (a * (1 - tr) ** (b + c * tr + d * tr ** 2 + e * tr ** 3)) * 4.184, 0)


def liquid_enthalpy(temperature, components):
    def calculate_h_liq(temperature, i):
        return hform_params()[i] + cp_vap(temperature, i) - h_evap(temperature, i)
    return jnp.where(components>0, vmap(calculate_h_liq, in_axes=(None, 0))(temperature, components), 0)


def vapor_enthalpy(temperature, components):
    def calculate_h_vap(temperature, i):
        return hform_params()[i] + cp_vap(temperature, i)
    return jnp.where(components > 0, vmap(calculate_h_vap, in_axes=(None, 0))(temperature, components), 0)


def t_solver(state, tray, temperature, xc):
    def function(tempcalc):
        return 1 - jnp.sum(vmap(k_eq, in_axes=(None, 0, None))(tempcalc, state.components, state.pressure) * xc)
    result = Bisection(optimality_fun=function, lower=500., upper=100., check_bracket=False).run(temperature[tray]).params
    return result


def feed_enthalpy(state: State):
    feed_temp = t_solver(state, None, state.pressure, jnp.array(state.z))
    return liquid_enthalpy(feed_temp[0], state.components)
