from pyomo.environ import *
from pyomo.dae import *

BaseModel = AbstractModel()

BaseModel.component_names = Set(doc="Names of the components that are in the mixture")
BaseModel.parameter_names = Set(doc="Names of the kinetic parameters of the reactive system")
BaseModel.fixed_parameter_names = Set(doc="Names of the kinetic parameters to fix")

BaseModel.init_conditions = Param(BaseModel.component_names,within=NonNegativeReals)
BaseModel.fixed_parameters = Param(BaseModel.fixed_parameter_names)
BaseModel.start_time = Param(within = NonNegativeReals, default = 0.0)
BaseModel.end_time = Param(within = NonNegativeReals, default = 1.0)

# Sets
BaseModel.time = ContinuousSet(bounds=(BaseModel.start_time,BaseModel.end_time))

# Variables
BaseModel.C = Var(BaseModel.time,
                    BaseModel.component_names,
                    bounds=(0.0,None),
                    initialize=1)

BaseModel.dCdt = DerivativeVar(BaseModel.C,
                                 wrt=BaseModel.time)

BaseModel.kinetic_parameter = Var(BaseModel.parameter_names,
                                    initialize=1)
# Constraints
def rule_init_conditions(model,k):
    #st = model.start_time
    st = 0
    return model.C[st,k] == model.init_conditions[k]

BaseModel.init_conditions_c = \
    Constraint(BaseModel.component_names,rule=rule_init_conditions)

def rule_fixed_parameters(model,theta):
    return model.kinetic_parameter[theta] == model.fixed_parameters[theta]
BaseModel.fix_parameters = Constraint(BaseModel.fixed_parameter_names,
                                        rule = rule_fixed_parameters)
