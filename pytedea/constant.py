# Composite error term
CET_ADDI = "addi"
"""
CET_ADDI: Additive composite error term.
"""

CET_MULT = "mult"
"""
CET_MULT: Multiplicative composite error term.
"""

CET_Categories = {
    CET_ADDI: "Additive composite error term",
    CET_MULT: "Multiplicative composite error term"
}

CET_Model_Categories = {
    CET_ADDI: "additive model",
    CET_MULT: "multiplicative model"
}

# Frontier
FUN_PROD = "prod"
"""
FUN_PROD: Production frontier.
"""

FUN_COST = "cost"
"""
FUN_COST: Cost frontier.
"""

FUN_Categories = {
    FUN_PROD: "Production frontier",
    FUN_COST: "Cost frontier"
}

# Return to scale
RTS_VRS = "vrs"
"""
RTS_VRS: Variable returns to scale.
"""

RTS_CRS = "crs"
"""
RTS_CRS: Constant returns to scale.
"""

RTS_Categories = {
    RTS_VRS: "Variable returns to scale",
    RTS_CRS: "Constant returns to scale"
}
# Emission abatement factor
EMF_SAME = "same"
"""
EMF_SAME: Same emission abatement factor.
"""

EMF_DIFFERENT = "different"
"""
EMF_DIFFERENT: Different emission abatement factor.
"""

EMF_Categories = {
    EMF_SAME: "Same emission abatement factor",
    EMF_DIFFERENT: "Different emission abatement factor"
}

# Orientation
ORIENT_IO = "io"
"""
ORIENT_IO: Input orientation.
"""

ORIENT_OO = "oo"
"""
ORIENT_OO: Output orientation.
"""

ORIENT_UO = "uo"
"""
ORIENT_UO: Undesirable Output orientation.
"""

ORIENT_HYPERYB = "hyper for y and b"
"""
ORIENT_HYPERYB: Hyperbolic orientation for desirable outputs and undesirable outputs.
"""
ORIENT_HYPERYX = "hyper for y and x"
"""
ORIENT_HYPERYX: Hyperbolic orientation for desirable outputs and inputs.
"""
ORIENT_Categories = {
    ORIENT_IO: "Input orientation",
    ORIENT_OO: "Output orientation",
    ORIENT_UO: "Undesirable Output orientation",
    ORIENT_HYPERYB: "Hyperbolic orientation for desirable outputs and undesirable outputs",
    ORIENT_HYPERYX: "Hyperbolic orientation for desirable outputs and inputs"
}

# left-hand or right-hand Derivative
LEFT = "left-hand Derivative"
"""
LEFT: left-hand Derivative.
"""

RIGHT = "right-hand Derivative"
"""
RIGHT: right-hand Derivative.
"""

Derivative_Categories = {
    LEFT: "left-hand Derivative",
    RIGHT: "right-hand Derivative"
}


# Residual decomposition
RED_MOM = "MOM"
"""
RED_MOM: Method of moments.
"""

RED_QLE = "QLE"
"""
RED_QLE: Quassi-likelihood estimation.
"""

RED_KDE = "KDE"
"""
RED_KDE: Kernel deconvolution estimation.
"""

RED_Categories = {
    RED_MOM: "Method of moments",
    RED_QLE: "Quassi-likelihood estimation",
    RED_KDE: "Kernel deconvolution estimation"
}

# Optimization
OPT_LOCAL = "local"
OPT_DEFAULT = None
