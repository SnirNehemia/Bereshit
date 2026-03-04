class CreaturesLogs:
    def __init__(self, id):
        self.creature_id = id
        # IMPORTANT: logs are in step units for now
        self.record = {'eat': [], 'reproduce': [], 'speed': [], 'energy': [], 'energy_consumption': []}

    def add_record(self, name, value):
        if name not in self.record:
            self.record[name] = []
        self.record[name].append(value)

