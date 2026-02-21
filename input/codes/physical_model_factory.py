import importlib
import pkgutil

import input.codes.physical_models as physical_models  # Your folder containing model scripts
from input.codes.physical_models.physical_model_abc import PhysicalModel


class PhysicalModelFactory:
    _registry = {}

    @classmethod
    def discover_models(cls):
        """Automatically finds and registers models in the 'models' package."""
        for loader, name, is_pkg in pkgutil.iter_modules(physical_models.__path__):
            # Dynamically import the module
            module = importlib.import_module(f"input.codes.physical_models.{name}")

            # Find classes in that module that inherit from PhysicalModel
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, PhysicalModel) and attr is not PhysicalModel:
                    # Use a lowercase version of the class name as the key
                    cls._registry[attr_name.lower()] = attr
        # print(f"Discovered physical models: {list(cls._registry.keys())}")

    @classmethod
    def create(cls, model_type, params):
        model_class = cls._registry.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Model '{model_type}' not found.")
        return model_class(**params)  # Unpacking params into the constructor
