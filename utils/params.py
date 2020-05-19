import yaml
import dotmap

def load_params(path):
    params = yaml.load(open(path), Loader=yaml.SafeLoader)
    params = dotmap.DotMap(params)
    return params