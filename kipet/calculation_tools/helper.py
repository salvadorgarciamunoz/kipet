"""
Top Level helper classes for KIPET
"""


class AttrDict(dict):

    """This class lets you use nested dicts like accessing attributes using a dot notation

    :Methods:

        - :func:`update`

    """
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
        """Method used to update the AttrDict using the same syntax as the dict class

        :param tuple args: The arguments (pass through)
        :param dict kwargs: The keyword arguments (pass through)

        :return: None
        """
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DosingPoint:
    """Small class to handle the dosing points in a cleaner manner

    :param str component: The name of the model component
    :param float time: The time of the dosing
    :param tuple conc: A tuple of the concentration (conc (float), units (str))
    :param tuple vol: A tuple of the dosing volume (vol (float), units (str))

    :Methods:

        - :func:`as_list`

    """
    def __init__(self, component, time, conc, vol):
        
        self.component = component
        self.time = time
        self.conc = conc
        self.vol = vol
    
    def __repr__(self):
        return f'{self.component}: {self.time}, {self.conc} {self.vol}'
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def as_list(self):
        """Return the dosing attributes as a list

        :return: List of dosing point attributes
        :rtype: list

        """
        return [self.component, self.time, self.conc, self.vol]
