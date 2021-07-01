"""
Unit conversion helper functions
"""
# Standard library imports
from enum import Enum

# Third party imports
import attr


class ConversionType(Enum):
    """Simple Enum class for conversion types"""

    volume = 1
    time = 2
    generic = 3


@attr.s
class UnitConversion:
    """Unit conversion class

    :param str unit: The units considered
    :param str dim: The dimensions of the unit
    :param int power: The power of the unit
    :param int dim_power: The power of the dimension

    """
    unit = attr.ib()
    dim = attr.ib()
    power = attr.ib()
    dim_power = attr.ib()


def convert_unit(unit_registry, u_orig, u_goal, scalar=1, power=1, both_powers=False, reverse_power=False):
    """Method to convert units taking the powers into account

    :param unit_registry: The unit registry to use (must be the same)
    :param str u_orig: The original units
    :param str u_goal: The target units to convert to
    :param float scalar: The scalar value to pass to the new units (usually 1)
    :param int power: The power of the unit dimension
    :param bool both_powers: Option to use both powers or a single unit's power
    :param bool reverse_power: Option to reverse the power (invert)

    :return: The new units
    :rtype: pint.Quantity

    """
    c1 = 1*unit_registry(u_orig)
    c2 = 1*unit_registry(u_goal)

    power2 = 1
    if both_powers:
        power2 = power

    if reverse_power:
        power2, power = power, power2

    con = (c1**power).to((c2**power2).u)/c2**power2 * (c2**power2).u/(c1**power).u

    return scalar * con


def convert_single_dimension(unit_registry, unit_from, unit_to):
    """This method converts only a single dimension from the total unit

    :param unit_registry: The required unit registry object
    :param str unit_from: The unit to be converted
    :param str unit_to: The target unit

    :return new_unit: The unit with the updated dimension units
    :rtype: pint.Quantity

    """
    conversion = ConversionType.generic

    if isinstance(unit_to, str):
        unit_to = unit_registry(unit_to)
        is_volume = unit_to.check('[volume]')

    if isinstance(unit_from, str):
        unit_from = unit_registry(unit_from)

    if is_volume:
        conversion = ConversionType.volume

    u_to = unit_to
    u_from = unit_from

    dim_to = {k.replace('[', '').replace(']', ''): v for k, v in dict(u_to.dimensionality).items()}
    dim_from = {k.replace('[', '').replace(']', ''): v for k, v in dict(u_from.dimensionality).items()}

    units_to = {k: v for k, v in u_to._units.items()}
    units_from = {k: v for k, v in u_from._units.items()}
    new_unit = unit_from

    for unit in units_from:
        dims = dict(unit_registry(unit).dimensionality)
        dims = {k.replace('[', '').replace(']', ''): v for k, v in dims.items()}

        try:
            uc_from = UnitConversion(unit=unit,
                                 dim=list(dims.keys())[0],
                                 power=units_from[unit],
                                 dim_power=dim_from[list(dims.keys())[0]]
                                 )
        finally:
            continue

        if (unit_registry(uc_from.unit)**uc_from.power).check("[power]"):
            continue

        if uc_from.unit == uc_to.unit:
            _print('The units are already equal...quitting')
            break

        if uc_from.dim == uc_to.dim:

            if conversion != ConversionType.volume:
                is_volume = (unit_registry(uc_from.unit)**uc_from.power).check("[volume]")
                if is_volume:
                    continue

                reverse_power = False
                if uc_from.power > uc_to.power:
                    reverse_power = True

                power = uc_from.power
                con = convert_unit(unit_registry, unit, uc_to.unit, power=abs(power), reverse_power=reverse_power)
                con = con ** (power/abs(power))
                new_unit = unit_from * con

            elif conversion == ConversionType.volume:
                if unit_registry(f'{unit}**{abs(uc_from.power)}').check('[volume]'):
                    power = 3
                    if abs(uc_from.power) == abs(uc_to.power):
                        power = uc_from.power

                    reverse_power = False
                    if abs(uc_from.power) < abs(uc_to.power):
                        reverse_power = True

                    both = False
                    if abs(uc_from.power) == abs(uc_to.power):
                        both = True

                    con = convert_unit(unit_registry, unit, uc_to.unit, power=power, both_powers=both, reverse_power=reverse_power)
                    con = con ** (uc_from.power/abs(uc_from.power))
                    new_unit = unit_from * con

                else:
                    continue

    return new_unit