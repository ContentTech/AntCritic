import copy


from utils.accessor import load_json, load_yaml


class ConfigContainer(object):
    def __init__(self):
        super(ConfigContainer, self).__init__()
        self.config = {}

    def load_config(self, path, name):
        config = load_yaml("./config/" + path + ".yaml")
        self.config.update({name: config})
        return self.config[name]

    def flatten_config(self, path, prefix, common_name):
        config = load_yaml("./config/" + path + ".yaml")
        assert common_name in config, "Invalid common config name!"
        common_config = config[common_name]
        for configName in config:
            if configName != common_name:
                config[configName].update(common_config)
                self.config.update({prefix + "_" + configName: config[configName]})

    def fetch_config(self, name):
        assert name in self.config, "Invalid config name!"
        return self.config[name]

    def concat_config(self, target_name, source_name):
        assert target_name in self.config, "Invalid target config name!"
        assert source_name in self.config, "Invalid source config name!"
        self.config[target_name].update(self.config[source_name])


class ModelContainer(object):
    def __init__(self):
        super(ModelContainer, self).__init__()
        self.model = {}
        self.data = {}

    def save_model(self, model, name, data, criterion, greater_better=True):
        assert criterion in data, "Incompatible criterion name!"
        if (name not in self.model) or ((self.data[name][criterion] <= data[criterion]) == greater_better):
            self.model.update({name: model})
            self.data.update({name: data})
        return self.data[name]

    def fetch_model(self, name):
        assert name in self.model, "Invalid model name!"
        return self.model[name], self.data[name]


def merge_dicts(a, b):
    for key, value in b.items():
        a[key] = value + a[key] if key in a else value
    return a


class ValueContainer(object):
    def __init__(self):
        super().__init__()
        self.data = {}

    def reset(self, name):
        self.data[name] = {"count": 0}

    def update(self, name, metrics, step=1):
        if name not in self.data:
            self.reset(name)
        if not isinstance(metrics, dict):
            metrics = {"__default__": metrics}
        self.data[name] = merge_dicts(self.data[name], metrics)
        self.data[name]["count"] += step

    def calculate_average(self, name, reset=True):
        result = {}
        cum_result = self.data[name]
        for key in cum_result:
            if key != "count":
                result[key] = cum_result[key] / cum_result["count"]
        if reset:
            self.reset(name)
        if "__default__" in result:
            return result["__default__"]
        return result


configContainer = ConfigContainer()
modelContainer = ModelContainer()
metricsContainer = ValueContainer()
