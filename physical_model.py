import dataclasses

import numpy as np


@dataclasses.dataclass
class PhysicalModel:
    # inertia mock-up (to limit angle turn)
    intertia_limiting_factor: float = 1  # [radians]*[mass]

    # gravity force
    g: float = 10  # [m/sec^2]

    # drag force (air resistence)
    gamma: float = 0.1  # [F]/[v] = [kg/sec] linear drag air resistence (dominant in low speeds)
    c_drag: float = 0.1  # [F]/[v^2] quadratic drag air resistence (dominant in high speeds)

    # friction force
    mu_static: float = 0.2  # [no units]
    mu_kinetic: float = 0.1  # [no units]
    alpha_mu: float = 1  # [no units]
    assert mu_static > mu_kinetic, f'Physical model error: {mu_static=} must be larger than {mu_kinetic=}'

    # convert physical parameters to energy
    energy_conversion_factors = {
        'activity_efficiency': 0.25,  # propulsion force to energy factor (higher -> efficient)
        'heat_loss': 0.2,  # propulsion force to wasted energy (higher -> more loss)
        'rest': 0.2,  # constant for Basal Metabolic Rate (BMR) energy
        'digest': 0.3,  # [E]/[no units] convert digest factor to digest energy
        'height': 5,  # [E]/[h]  # convert height to height energy
        'mass': 3  # [E]/[m] = [v^2] = [m^2/sec^4]  convert mass to mass energy
    }

    # trait change formulas based on given energy
    trait_energy_func = lambda factor, rate, age: factor * np.exp(-rate * age)
    trait_energy_params_dict = {
        'height': {'factor': 0.3, 'rate': 0.1},  # [factor] = [1], [rate] = [1/sec]
        'mass': {'factor': 0.2, 'rate': 0.2}  # [factor] = [1], [rate] = [1/sec]
    }

    # Check that sum of energy factors wasted on traits is between 0 and 1
    assert 0 <= np.sum(
        [trait_energy_params['factor'] for trait_energy_params in trait_energy_params_dict.values()]) <= 1, \
        'Physical model error: creature cannot digest more energy than food energy'
