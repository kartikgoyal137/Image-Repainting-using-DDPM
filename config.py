import os
import yaml
from collections import defaultdict

class Conf(defaultdict):
    def __init__(self):
        super().__init__(lambda: None)

    def __getattr__(self, attr):
        return self.get(attr)

    def pget(self, name, default=None):
        d = self
        for n in (name.split('.') if '.' in name else [name]):
            d = d.get(n, default) if isinstance(d, dict) else default
            if d is None:
                return default
        return d

def load_conf(path):
    with open(os.path.expanduser(path), 'r') as f:
        data = yaml.safe_load(f.read())
    conf = Conf()
    conf.update(data)
    return conf
