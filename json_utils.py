import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and complex objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Serializable:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        for key, value in data.items():
            obj.__setattr__(key, value)
        return obj

    def to_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, cls=NumpyEncoder)

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

