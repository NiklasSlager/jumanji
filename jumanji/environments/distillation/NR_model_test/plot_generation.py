from matplotlib import pyplot as plt
import jax.numpy as jnp

def plot_function(stages, L, V, temperature, H_liq, H_vap, xnorm, y, components):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    # Plot data on each subplot
    axs[0, 0].plot(stages, L)
    axs[0, 0].set_title('Liquid flow ')
    axs[0, 0].set_xlabel('Stages')
    axs[0, 0].set_ylabel('Mole flow (kmol/hr)')
    axs[0, 0].grid(True)
    axs[0, 0].set_xticks(jnp.arange(1, len(stages)))

    axs[0, 1].plot(stages, V)
    axs[0, 1].set_xlabel('Stages')
    axs[0, 1].set_ylabel('Mole flow (kmol/hr)')
    axs[0, 1].set_title('Vapor flow')
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks(jnp.arange(1, len(stages)))

    axs[0, 2].plot(stages, temperature - 273.15)
    axs[0, 2].set_xlabel('Stages')
    axs[0, 2].set_ylabel('Temperature (C)')
    axs[0, 2].set_title('Stage temperature')
    axs[0, 2].grid(True)
    axs[0, 2].set_xticks(jnp.arange(1, len(stages)))

    axs[1, 0].plot(stages, jnp.array(H_liq) / 4.184)
    axs[1, 0].set_title('Liquid enthalpy')
    axs[1, 0].set_xlabel('Stages')
    axs[1, 0].set_ylabel('Molar enthalpy (kj/mol)')
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks(jnp.arange(1, len(stages)))

    axs[1, 1].plot(stages, jnp.array(H_vap) / 4.184)
    axs[1, 1].set_title('Vapor enthalpy')
    axs[1, 1].set_xlabel('Stages')
    axs[1, 1].set_ylabel('Molar enthalpy (kj/mol)')
    axs[1, 1].grid(True)
    axs[1, 1].set_xticks(jnp.arange(1, len(stages)))

    cat = ["C3", "I-C4", "C4", "C5", "C-C5", "C6", "T-C4", "C7", "D-C6", "C8"]
    components = jnp.where(components>0, components+1, 0)
    for xc in components:
        if xc > 0:
            axs[1, 2].plot(stages, xnorm[xc-1, :], label=component_names[xc-1]) #, label=[key for key, value in components.items() if value == xc]


    axs[1, 2].set_title('Composition')
    axs[1, 2].set_xlabel('Stages')
    axs[1, 2].set_ylabel('Composition(-)')
    axs[1, 2].set_ylim(0, 1)
    axs[1, 2].grid(True)
    axs[1, 2].legend()
    axs[1, 2].set_xticks(jnp.arange(1, len(stages)))
