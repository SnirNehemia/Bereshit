import json


class Serializable:
    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)  # create instance without calling __init__
        obj.__dict__.update(data)
        return obj


def save_to_json(filepath, obj):
    # Save to JSON file
    with open(filepath, "w") as f:
        json.dump(obj.to_dict(), f, indent=2)


def load_json(filepath):
    # Read from JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    pass
    # Example of saving the object and then loading and reconstructing it:
    # import json_utils
    # json_utils.save_to_json(filepath='a.json', obj=self.statistics_logs)
    # data = json_utils.load_json(filepath='a.json')
    # loaded_obj = StatisticsLogs.from_dict(data)
    # print(loaded_obj.__dict__)
