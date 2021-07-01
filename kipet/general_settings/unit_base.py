"""
Unit Base - Object for holding units and the unit registry
"""
import pint


class UnitBase:

    """Class to hold universal units and the unit registry

    :param pint.UnitRegistry ur: default unit registry
    :param str time: time base
    :param str volume: volume base
    :param str concentration: concentration base
    """

    def __init__(self):
        """Initialize the UnitBase for the ReactionModel

        :param pint.UnitRegistry ur: default unit registry
        :param str time: time base
        :param str volume: volume base
        :param str concentration: concentration base

        """
        self.ur = pint.UnitRegistry()
        self.time = 'min'
        self.volume = 'L'
        self.concentration = 'M'