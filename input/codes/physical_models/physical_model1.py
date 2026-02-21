from input.codes.physical_models.physical_model_abc import PhysicalModel


class PhysicalModel1(PhysicalModel):
    def __init__(self, **params):
        # init config based on data from yaml
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)

    def move_creature(self, creature, decision, dt):
        pass

    def digest_food(self, creature, food_type, food_energy):
        pass
