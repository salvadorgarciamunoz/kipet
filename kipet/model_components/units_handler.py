"""
Unit conversion helper functions

@author: kevin
"""
import attr
from enum import Enum

from kipet.dev_tools.display import Print

DEBUG = False

_print = Print(verbose=DEBUG)

class ConversionType(Enum):
    
    volume = 1
    time = 2
    generic = 3
    
    
@attr.s
class UnitConversion:

    unit = attr.ib()
    dim = attr.ib()
    power = attr.ib()
    dim_power = attr.ib()
    

def convert_unit(unit_registry, u_orig, u_goal, scalar=1, power=1, both_powers=False, reverse_power=False):
   
    c1 = 1*unit_registry(u_orig)
    c2 = 1*unit_registry(u_goal)
    
    power2 = 1
    if both_powers:
        power2 = power
        
    if reverse_power:
        power2, power = power, power2
        
    con = (c1**power).to((c2**power2).u)/c2**power2 * (c2**power2).u/(c1**power).u
    
    return scalar * con

    #%%
def convert_single_dimension(unit_registry, unit_from, unit_to, power_fixed=False):
    
    #%%
    # unit_registry = r1.unit_base.ur
    # unit_from = 'mol/m**3'
    # unit_to = 'L'
    # # unit_registry = r1.unit_base.ur
    # power_fixed = True
    
    conversion = ConversionType.generic
    
    if isinstance(unit_to, str):
        unit_to = unit_registry(unit_to)
        is_volume = unit_to.check('[volume]')
    
    if isinstance(unit_from, str):
        unit_from = unit_registry(unit_from)
        
    if is_volume:
        conversion = ConversionType.volume
    
    # print(conversion)
    # print(unit_from)
    # print(unit_to)
    
    u_to = unit_to
    u_from = unit_from
    
    _print('#' * 25)
    _print(f'\nCan {unit_from} be converted to {u_to}?')
    
    dim = {}
    dim_to = {k.replace('[', '').replace(']', ''): v for k, v in dict(u_to.dimensionality).items()}
    dim_from = {k.replace('[', '').replace(']', ''): v for k, v in dict(u_from.dimensionality).items()}
    
    
    units_to = {k: v for k, v in u_to._units.items()}
    units_from = {k: v for k, v in u_from._units.items()}
    
    uc_to = UnitConversion(unit=list(units_to.keys())[0],
                           dim=list(dim_to.keys())[0],
                           power=list(units_to.values())[0],
                           dim_power=list(dim_to.values())[0]
                           
                           )
    
    # print(uc_to)
   
    new_unit = unit_from
   
    for unit in units_from:
       
        _print(f'\n\tLooking at {unit}:\n')
        
        dims = dict(ur(unit).dimensionality)
        dims = {k.replace('[', '').replace(']', ''): v for k, v in dims.items()}

        try:
            uc_from = UnitConversion(unit=unit, 
                                 dim=list(dims.keys())[0], 
                                 power=units_from[unit],
                                 dim_power=dim_from[list(dims.keys())[0]]
                                 )
        except:
            _print('\tNo match for this unit found in the reduced set, quitting')
            continue

        _print(f'\t{uc_from}')
        _print(f'\t{uc_to}\n')
        
        
        if (ur(uc_from.unit)**uc_from.power).check("[power]"):
            #print(f'Handling power unit: {(ur(uc_from.unit)**uc_from.power).check("[power]")}')
            continue
        
        if uc_from.unit == uc_to.unit:
            _print('The units are already equal...quitting')
            break
            
        if uc_from.dim == uc_to.dim:
            #print('The dimensions match')
            
            #print(conversion)
            
            if conversion != ConversionType.volume:
                is_volume = (ur(uc_from.unit)**uc_from.power).check("[volume]")
                if is_volume:    
                    continue
    
                reverse_power = False
                if uc_from.power > uc_to.power:
                    reverse_power = True
                
                power = uc_from.power
                con = convert_unit(unit_registry, unit, uc_to.unit, power=abs(power), reverse_power=reverse_power)
                con = con ** (power/abs(power))
                new_unit = unit_from * con
                #print(new_unit)
                
            elif conversion == ConversionType.volume:
                _print(ur(f'{unit}**{uc_from.power}'))
                if ur(f'{unit}**{abs(uc_from.power)}').check('[volume]'):
                    _print('Converting from volume to volume')
                    
                    power = 3 #if uc_from.power > 0 else -3
                    if abs(uc_from.power) == abs(uc_to.power):
                        power = uc_from.power
                    
                    reverse_power = False
                    if abs(uc_from.power) < abs(uc_to.power):
                        reverse_power = True
                    # reverse_power = False
                    
                    both = False
                    if abs(uc_from.power) == abs(uc_to.power):
                        both = True
                    
                    _print(f'convert_unit(ur, {unit}, {uc_to.unit}, power={power}, both_powers={both}, reverse_power={reverse_power}')
                    _print(power)
                    con = convert_unit(unit_registry, unit, uc_to.unit, power=power, both_powers=both, reverse_power=reverse_power)
                    con = con ** (uc_from.power/abs(uc_from.power))
                    _print(unit_from)
                    _print(con)
                    
                    new_unit = unit_from * con
                    _print(new_unit)
            
                else:
                    continue
      
    
    _print('\nFinished checking\nResults:')
    _print(f'\t{new_unit}')
    #%%
    return new_unit
    
    #%%
    
import pint
import pytest

ur = pint.UnitRegistry()

# b = ur('L')

# print(b.check('[volume]'))



#%%
    
def test_time_change():
    
    old_unit = 'hr'
    new_unit = 'min'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(new_unit)
    
def test_reverse_time_change():
    
    old_unit = 'min'
    new_unit = 'hr'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(f'{new_unit}')
    
def test_inverse_time_change():
    #%%
    old_unit = '1/hr'
    new_unit = 'min'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    
#%%
    assert converted_test_unit.units == ur(f'1/{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(f'1/{new_unit}')
    
def test_velocity_time_change():
    #%%
    old_unit = 'm/min'
    new_unit = 'hr'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    
#%%
    assert converted_test_unit.units == ur(f'm/{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(f'm/{new_unit}')
    
def test_inverse_velocity_time_change():
    
    old_unit = 'm/hr'
    new_unit = 's'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'm/{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(f'm/{new_unit}')
    
def test_velocity_length_change():
    
    old_unit = 'm/s'
    new_unit = 'km'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'{new_unit}/s').units
    assert converted_test_unit.m == test_unit.m_as(f'{new_unit}/s')
    
def test_inverse_velocity_length_change():
    
    old_unit = 'km/s'
    new_unit = 'm'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'{new_unit}/s').units
    assert converted_test_unit.m == test_unit.m_as(f'{new_unit}/s')
    
def test_volume_change_l3_to_l3():
    
    old_unit = 'm**3'
    new_unit = 'ft**3'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'{new_unit}').units
    # assert converted_test_unit.m == 1000
    
    
def test_volume_change_l3_to_l1():
    
    old_unit = 'm**3'
    new_unit = 'ft'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)  
    
    assert converted_test_unit.units == ur('m**3').units
    assert converted_test_unit.m == test_unit.m_as(old_unit)    
    
def test_volume_change_vol_to_l1():
    
    old_unit = 'L'
    new_unit = 'ft'
    
    test_unit = 1*ur(old_unit)

    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)  
    
    assert converted_test_unit.units == ur('L').units
    assert converted_test_unit.m == test_unit.m_as(old_unit)
    
def test_volume_change_vol_to_l3():
    
    old_unit = 'L'
    new_unit = 'ft**3'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(new_unit)

def test_concentration_change():
    
    old_unit = 'mol/m**3'
    new_unit = 'L'
    
    test_unit = 1*ur(old_unit)
    
    converted_test_unit = convert_single_dimension(ur, old_unit, new_unit, power_fixed=False)    

    assert converted_test_unit.units == ur(f'mol/{new_unit}').units
    assert converted_test_unit.m == test_unit.m_as(f'mol/{new_unit}')
    
    
    