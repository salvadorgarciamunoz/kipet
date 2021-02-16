"""
Top Level helper classes for KIPET
"""

class AttrDict(dict):

    "This class lets you use nested dicts like accessing attributes"
    
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setitem__(self, key, item):
        
        if isinstance(item, dict):
            return super(AttrDict, self).__setitem__(key, AttrDict(item))
        else:
            return dict.__setitem__(self, key, item)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DosingPoint():
    """Small class to handle the dosing points in a cleaner manner"""
    
    def __init__(self, component, time, step):
        
        self.component = component
        self.time = time
        self.step = step
    
    def __repr__(self):
        return f'{self.component}: {self.time}, {self.step}'
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def as_list(self):
        return [self.component, self.time, self.step]
