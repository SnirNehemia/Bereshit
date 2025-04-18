import dataclasses

import numpy as np


@dataclasses.dataclass
class PhysicalModel:
    # gravity force
    g: float = 10  # [m/sec^2]

    # drag force (air resistence)
    gamma: float = 1e-4  # [F]/[v] = [kg/sec] linear drag air resistence (dominant in low speeds)
    c_drag: float = 5e-3  # [F]/[v^2] quadratic drag air resistence (dominant in high speeds)

    # friction force
    mu_static: float = 1.5  # [no units]  # higher mu_static means faster movement
    mu_kinetic: float = 0.5  # [no units]
    alpha_mu: float = 1  # [no units]
    assert mu_static > mu_kinetic, f'Physical model error: {mu_static=} must be larger than {mu_kinetic=}'

    # convert physical parameters to energy
    energy_conversion_factors = {
        'activity_efficiency': 0.5,  #0.25,  # propulsion force to energy factor (higher -> efficient)
        'heat_loss': 0.01,  # propulsion force to wasted energy (higher -> more loss)
        'rest': 0.00075,  # constant for Basal Metabolic Rate (BMR) energy
        'digest': 0.05,  # [E]/[no units] convert digest factor to digest energy
        'brain_consumption': 0.005,  # [E]/[no units] convert brain size to brain consumption energy
        'height': 1,  # [E]/[h]  # convert height to height energy
        'mass': 25  # [E]/[m] = [v^2] = [m^2/sec^4]  convert mass to mass energy
    }

    # trait change formulas based on given energy
    trait_energy_func = lambda factor, rate, age: factor * np.exp(-rate * age)
    trait_energy_params_dict = {
        'height': {'factor': 0.001, 'rate': 5e-4},  # [factor] = [1], [rate] = [1/sec]
        'mass': {'factor': 0.005, 'rate': 2e-4}  # [factor] = [1], [rate] = [1/sec]
    }

    # Check that sum of energy factors wasted on traits is between 0 and 1
    assert 0 <= np.sum(
        [trait_energy_params['factor'] for trait_energy_params in trait_energy_params_dict.values()]) <= 1, \
        'Physical model error: creature cannot digest more energy than food energy'
