import numpy as np

from input.codes.config import Config
from input.codes.physical_model import PhysicalModel


def validate_config(config: Config):
    assert list(config.NUM_STEPS_FROM_FRAME_DICT.keys())[0] == 0, \
        'Config Error: first key in NUM_STEPS_FROM_FRAME_DICT must be 0'

    for c_digest in config.INIT_DIGEST_DICT.values():
        assert 0 <= c_digest <= 1, \
            'Config Error: INIT_DIGEST_DICT is not set correctly (values between 0-1).'


def validate_physical_model(physical_model: PhysicalModel):
    assert physical_model.mu_static > physical_model.mu_kinetic, \
        f'Physical model error: {physical_model.mu_static=} must be larger' \
        f' than {physical_model.mu_kinetic=}'

    # Check that sum of energy factors wasted on traits is between 0 and 1
    assert 0 <= np.sum(
        [trait_energy_params['factor'] for trait_energy_params in
         physical_model.trait_energy_params_dict.values()]) <= 1, \
        'Physical model error: creature cannot digest more energy than food energy'
