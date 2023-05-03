# import dependencies
from re import compile
import ast
from os import environ
import numpy as np
from pyomo.opt import SolverFactory, SolverManagerFactory
from ..constant import CET_ADDI, CET_MULT, CET_Model_Categories, OPT_LOCAL, OPT_DEFAULT, RTS_CRS
__email_re = compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')


def set_neos_email(address):
    """pass email address to NEOS server

    Args:
        address (String): your own vaild email address.
    """
    if address == OPT_LOCAL:
        print("Optimizing locally.")
        return False
    if not __email_re.match(address):
        raise ValueError("Invalid email address.")
    environ['NEOS_EMAIL'] = address
    return True

def optimize_model2(model, ind, solver=OPT_DEFAULT):
    if solver is not OPT_DEFAULT:
        assert_solver_available_locally(solver)

    solver_instance = SolverFactory(solver)
    # print("Estimating dmu{} locally with {} solver.".format(
    #     ind, solver), flush=True)
    return str(solver_instance.solve(model, tee=False)), 1


def optimize_model3(model, solver=OPT_DEFAULT):
    if solver is not OPT_DEFAULT:
        assert_solver_available_locally(solver)

    solver_instance = SolverFactory(solver)
    # print("Estimating dmu{} locally with {} solver.".format(
    #     ind, solver), flush=True)
    return str(solver_instance.solve(model, tee=False)), 1

def optimize_model(model, email, cet, solver=OPT_DEFAULT):
    if not set_neos_email(email):
        if solver is not OPT_DEFAULT:
            assert_solver_available_locally(solver)
        elif cet == CET_ADDI:
            solver = "mosek"
        elif cet == CET_MULT:
            raise ValueError(
                "Please specify the solver for optimizing multiplicative model locally.")
        solver_instance = SolverFactory(solver)
        print("Estimating the {} locally with {} solver.".format(
            CET_Model_Categories[cet], solver), flush=True)
        return solver_instance.solve(model, tee=True), 1
    else:
        if solver is OPT_DEFAULT and cet is CET_ADDI:
            solver = "mosek"
        elif solver is OPT_DEFAULT and cet == CET_MULT:
            solver = "knitro"
        solver_instance = SolverManagerFactory('neos')
        print("Estimating the {} remotely with {} solver.".format(
            CET_Model_Categories[cet], solver), flush=True)
        return solver_instance.solve(model, tee=True, opt=solver), 1


def trans_list(li):
    if type(li) == list:
        return li
    return li.tolist()


def to_1d_list(li):
    if type(li) == int or type(li) == float:
        return [li]
    if type(li[0]) == list:
        rl = []
        for i in range(len(li)):
            rl.append(li[i][0])
        return rl
    return li


def to_2d_list(li):
    if type(li[0]) != list:
        rl = []
        for value in li:
            rl.append([value])
        return rl
    return li

def assert_valid_deat(sent,data,baseindex,refindex):
    inputvars = sent.split('=')[0].strip(' ').split(' ')
    outputvars = sent.split('=')[1].strip(' ').split(' ')

    if type(baseindex) != type(None):
        varname1 = baseindex.split('=')[0]
        print(baseindex)
        varvalue1 = ast.literal_eval(baseindex.split('=')[1])
        y, x = data.loc[data[varname1].isin(varvalue1), outputvars
        ], data.loc[data[varname1].isin(varvalue1), inputvars]
    else:
        y, x = data.loc[:, outputvars
                         ], data.loc[:, inputvars]

    # print(type(self.varname1),self.varvalue1,self.x,)
    if type(refindex) != type(None):
        varname = refindex.split('=')[0]
        varvalue = ast.literal_eval(refindex.split('=')[1])

        yref, xref = data.loc[data[varname].isin(varvalue), outputvars
        ], data.loc[data[varname].isin(varvalue), inputvars]
    else:
        yref, xref = data.loc[:, outputvars
                               ], data.loc[:, inputvars]

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")


    return outputvars,inputvars,y, x,yref, xref

def assert_valid_ddft(sent,gy,gx):
    inputvars = sent.split('=')[0].strip(' ').split(' ')
    outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')

    if len(outputvars) != len(gy):
        raise ValueError("Number of outputs must be the same in y and gy.")
    if len(inputvars) != len(gx):
        raise ValueError("Number of inputs must be the same in x and gx.")

    return outputvars,inputvars,gy,gx

def assert_valid_ddft2(data,baseindex,refindex,outputvars,inputvars):

    if type(baseindex) != type(None):
        varname1 = baseindex.split('=')[0]
        varvalue1 = ast.literal_eval(baseindex.split('=')[1])
        y, x = data.loc[data[varname1].isin(varvalue1), outputvars
        ], data.loc[data[varname1].isin(varvalue1), inputvars
        ]

    else:
        y, x = data.loc[:, outputvars], data.loc[:, inputvars]

    if type(refindex) != type(None):
        varname=refindex.split('=')[0]
        varvalue=ast.literal_eval(refindex.split('=')[1])

        yref, xref = data.loc[data[varname].isin(varvalue), outputvars
                                            ], data.loc[data[varname].isin(varvalue), inputvars
                                            ]
    else:
        yref, xref = data.loc[:, outputvars], data.loc[:, inputvars]
    return y,x,yref,xref

def assert_valid_dea(sent,data,baseindex,refindex):
    inputvars = sent.split('=')[0].strip(' ').split(' ')
    try:
        outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
        unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    except:
        outputvars = sent.split('=')[1].strip(' ').split(' ')
        unoutputvars = None

    if type(baseindex) != type(None):
        varname1 = baseindex.split('=')[0]
        print(baseindex)
        varvalue1 = ast.literal_eval(baseindex.split('=')[1])
        y, x, b = data.loc[data[varname1].isin(varvalue1), outputvars
        ], data.loc[data[varname1].isin(varvalue1), inputvars
        ], data.loc[data[varname1].isin(varvalue1), unoutputvars
        ]

    else:
        y, x, b = data.loc[:, outputvars
                         ], data.loc[:, inputvars
                         ], data.loc[:, unoutputvars]

    # print(type(self.varname1),self.varvalue1,self.x,)
    if type(refindex) != type(None):
        varname = refindex.split('=')[0]
        varvalue = ast.literal_eval(refindex.split('=')[1])

        yref, xref, bref = data.loc[data[varname].isin(varvalue), outputvars
        ], data.loc[data[varname].isin(varvalue), inputvars
        ], data.loc[data[varname].isin(varvalue), unoutputvars]
    else:
        yref, xref, bref= data.loc[:, outputvars
                               ], data.loc[:, inputvars
                                  ], data.loc[:, unoutputvars]

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")
    return outputvars,inputvars, unoutputvars,y, x,b ,yref, xref,bref

def assert_valid_ddf(sent,gy,gx,gb):
    inputvars = sent.split('=')[0].strip(' ').split(' ')
    try:
        outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
        unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    except:
        outputvars = sent.split('=')[1].strip(' ').split(' ')
        unoutputvars = None

    if len(outputvars) != len(gy):
        raise ValueError("Number of outputs must be the same in y and gy.")
    if len(inputvars) != len(gx):
        raise ValueError("Number of inputs must be the same in x and gx.")
    if len(unoutputvars) != len(gb):
        raise ValueError("Number of undesirable outputs must be the same in b and gb.")
    return outputvars,inputvars,unoutputvars,gy,gx,gb


def assert_valid_ddf2(data, baseindex, refindex, outputvars, inputvars, unoutputvars):

    if type(baseindex) != type(None):
        varname1 = baseindex.split('=')[0]
        print(baseindex)
        varvalue1 = ast.literal_eval(baseindex.split('=')[1])
        y, x, b = data.loc[data[varname1].isin(varvalue1), outputvars
        ], data.loc[data[varname1].isin(varvalue1), inputvars
        ], data.loc[data[varname1].isin(varvalue1), unoutputvars
        ]

    else:
        y, x, b = data.loc[:, outputvars
                         ], data.loc[:, inputvars
                         ], data.loc[:, unoutputvars]

    # print(type(self.varname1),self.varvalue1,self.x,)
    if type(refindex) != type(None):
        varname = refindex.split('=')[0]
        varvalue = ast.literal_eval(refindex.split('=')[1])

        yref, xref, bref = data.loc[data[varname].isin(varvalue), outputvars
        ], data.loc[data[varname].isin(varvalue), inputvars
        ], data.loc[data[varname].isin(varvalue), unoutputvars]
    else:
        yref, xref, bref= data.loc[:, outputvars
                               ], data.loc[:, inputvars
                                  ], data.loc[:, unoutputvars]

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")
    return y, x,b ,yref, xref,bref




def assert_valid_basic_data(y, x, z=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_1d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The multidimensional output data is supported by direciontal based models.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, z


def assert_valid_mupltiple_y_data(y, x):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")
    return y, x


def assert_valid_reference_data(y, x, yref, xref):
    yref = trans_list(yref)
    xref = trans_list(xref)

    yref = to_2d_list(yref)
    xref = to_2d_list(xref)

    yref_shape = np.asarray(yref).shape
    xref_shape = np.asarray(xref).shape

    if yref_shape[0] != xref_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in xref and yref.")
    if yref_shape[1] != np.asarray(y).shape[1]:
        raise ValueError(
            "Number of outputs must be the same in y and yref.")
    if xref_shape[1] != np.asarray(x).shape[1]:
        raise ValueError(
            "Number of inputs must be the same in x and xref.")
    return yref, xref


def assert_valid_reference_data_with_bad_outputs(y, x, b, yref, xref, bref):
    yref, xref = assert_valid_reference_data(y, x, yref, xref)

    if type(b) == type(None):
        return yref, xref, None

    bref = to_2d_list(bref)
    bref_shape = np.asarray(bref).shape

    if bref_shape[0] != np.asarray(yref).shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in yref and bref.")
    if bref_shape[1] != np.asarray(b).shape[1]:
        raise ValueError(
            "Number of undesirable outputs must be the same in b and bref.")

    return yref, xref, bref


def assert_valid_direciontal_data(y, x, b=None, gy=[1], gx=[1], gb=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    gy = to_1d_list(gy)
    gx = to_1d_list(gx)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if y_shape[1] != len(gy):
        raise ValueError("Number of outputs must be the same in y and gy.")

    if x_shape[1] != len(gx):
        raise ValueError("Number of inputs must be the same in x and gx.")

    if type(b) != type(None):
        b = trans_list(b)
        b = to_2d_list(b)
        gb = to_1d_list(gb)
        b_shape = np.asarray(b).shape
        if b_shape[0] != b_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and b.")
        if b_shape[1] != len(gb):
            raise ValueError(
                "Number of undesirable outputs must be the same in b and gb.")

    return y, x, b, gy, gx, gb


def assert_optimized(optimization_status):
    if optimization_status == 0:
        raise Exception(
            "Model isn't optimized. Use optimize() method to estimate the model.")


def assert_contextual_variable(z):
    if type(z) == type(None):
        raise Exception(
            "Estimated coefficient (lambda) cannot be retrieved due to no contextual variable (z variable) included in the model.")

def assert_desirable_output(y):
    if type(y) == type(None):
        raise Exception(
            "Estimated coefficient (gamma) cannot be retrieved due to no desirable output (y variable) included in the model.")

def assert_undesirable_output(b):
    if type(b) == type(None):
        raise Exception(
            "Estimated coefficient (delta) cannot be retrieved due to no undesirable output (b variable) included in the model.")


def assert_various_return_to_scale(rts):
    if rts == RTS_CRS:
        raise Exception(
            "Estimated intercept (alpha) cannot be retrieved due to the constant returns-to-scale assumption.")


def assert_various_return_to_scale_omega(rts):
    if rts == RTS_CRS:
        raise Exception(
            "Omega cannot be retrieved due to the constant returns-to-scale assumption.")


def assert_solver_available_locally(solver):
    if not SolverFactory(solver).available():
        raise ValueError("Solver {} is not available locally.".format(solver))


def assert_valid_direciontal_data_with_z(y, x, b=None,z=None, gy=[1], gx=[1], gb=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    gy = to_1d_list(gy)
    gx = to_1d_list(gx)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if y_shape[1] != len(gy):
        raise ValueError("Number of outputs must be the same in y and gy.")

    if x_shape[1] != len(gx):
        raise ValueError("Number of inputs must be the same in x and gx.")

    if type(b) != type(None):
        b = trans_list(b)
        b = to_2d_list(b)
        gb = to_1d_list(gb)
        b_shape = np.asarray(b).shape
        if b_shape[0] != b_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and b.")
        if b_shape[1] != len(gb):
            raise ValueError(
                "Number of undesirable outputs must be the same in b and gb.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")
    return y, x, b,z, gy, gx, gb

def assert_valid_wp_data_x(y, x, b, z=None):
    y = trans_list(y)
    x = trans_list(x)
    b = trans_list(b)

    y = to_2d_list(y)
    x = to_2d_list(x)
    b = to_2d_list(b)
    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape
    b_shape = np.asarray(b).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The multidimensional output data is supported by direciontal based models.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if x_shape[0] != b_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and b.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, b, z

def assert_valid_wp_data_b(y, x, b, z=None):
    y = trans_list(y)
    x = trans_list(x)
    b = trans_list(b)

    y = to_2d_list(y)
    x = to_2d_list(x)
    b = to_1d_list(b)
    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape
    b_shape = np.asarray(b).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The multidimensional output data is supported by direciontal based models.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if x_shape[0] != b_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and b.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, b, z

def assert_valid_wp_data(y, x, b, z=None):
    y = trans_list(y)
    x = trans_list(x)
    b = trans_list(b)

    y = to_1d_list(y)
    x = to_2d_list(x)
    b = to_2d_list(b)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape
    b_shape = np.asarray(b).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The multidimensional output data is supported by direciontal based models.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if x_shape[0] != b_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and b.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, b, z


def assert_valid_mupltiple_x_y_data(y, x, z=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, z

def assert_valid_yxbz_nog(sent,z):
    inputvars = sent.split('=')[0].strip(' ').split(' ')

    try:
        outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
        unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    except:
        outputvars = sent.split('=')[1].strip(' ').split(' ')
        unoutputvars = None
    zvars=z.strip(' ').split(' ') if type(z)!=type(None) else None

    return outputvars,inputvars,unoutputvars,zvars

def assert_valid_yxb(sent,gy,gx,gb):
    inputvars = sent.split('=')[0].strip(' ').split(' ')

    try:
        outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
        unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    except:
        outputvars = sent.split('=')[1].strip(' ').split(' ')
        unoutputvars = None

    if len(outputvars) != gy.shape[1]:
        raise ValueError("Number of outputs must be the same in y and gy.")
    if len(inputvars) != gx.shape[1]:
        raise ValueError("Number of inputs must be the same in x and gx.")

    if type(gb) != type(None):
        if len(unoutputvars) != gb.shape[1]:
            raise ValueError(
                "Number of undesirable outputs must be the same in b and gb.")
        gb.columns = unoutputvars
    gy.columns,gx.columns ,=outputvars,inputvars,
    return outputvars,inputvars,unoutputvars,gy,gx,gb

def assert_valid_yxb2(baseindex,refindex,data,outputvars,inputvars,unoutputvars):

    if type(baseindex) != type(None):
        varname1 = baseindex.split('=')[0]
        varvalue1 = ast.literal_eval(baseindex.split('=')[1])
        y, x, b = data.loc[data[varname1].isin(varvalue1), outputvars
        ], data.loc[data[varname1].isin(varvalue1), inputvars
        ], data.loc[data[varname1].isin(varvalue1), unoutputvars
        ]

    else:
        y, x, b = data.loc[:, outputvars], data.loc[:, inputvars], data.loc[:, unoutputvars ]

    if type(refindex) != type(None):
        varname=refindex.split('=')[0]
        varvalue=ast.literal_eval(refindex.split('=')[1])

        yref, xref, bref = data.loc[data[varname].isin(varvalue), outputvars
                                            ], data.loc[data[varname].isin(varvalue), inputvars
                                            ], data.loc[data[varname].isin(varvalue), unoutputvars
                                            ]
    else:
        yref, xref, bref = data.loc[:, outputvars], data.loc[:, inputvars], data.loc[:, unoutputvars ]
    return y,x,b,yref,xref,bref




def assert_valid_yxbz2(baseindex,refindex,data,outputvars,inputvars,unoutputvars,zvars):


    if type(baseindex) != type(None):
        varname = baseindex.split('=')[0]
        yr = ast.literal_eval(baseindex.split('=')[1])
        y, x, b,z = data.loc[data[varname].isin(yr), outputvars], \
                    data.loc[data[varname].isin(yr), inputvars], \
                    data.loc[data[varname].isin(yr), unoutputvars], \
                    data.loc[data[varname].isin(yr), zvars] if type(zvars) != type(None) else None
        if type(refindex) != type(None):
            yrref = ast.literal_eval(refindex.split('=')[1])

            if len(set(yr) - set(yrref)) > 0:
                print("ssssssssssssss1111111")
                raise ValueError(
                    "You must specify basic data smaller than reference data.")
            else:
                print("ssssssssssssss22222222")
                yrref2 = list(set(yrref) - set(yr))
                try:
                    print(yrref2[0])
                    yref, xref, bref, zref = data.loc[data[varname].isin(yrref2), outputvars], \
                        data.loc[data[varname].isin(yrref2), inputvars], \
                        data.loc[data[varname].isin(yrref2), unoutputvars], \
                        data.loc[data[varname].isin(yrref2), zvars] if type(zvars) != type(None) else None
                except:
                    yref, xref, bref, zref = None, \
                        None, \
                        None, \
                        None
        elif type(refindex) == type(None):
            yrref = list(data[varname].unique())
            if len(set(yr) - set(yrref)) > 0:
                print("ssssssssssssss1111111")
                raise ValueError(
                    "You must specify basic data smaller than reference data.")
            else:

                print("ssssssssssssss22222222")
                yrref2 = list(set(yrref) - set(yr))
                try:
                    print(yrref2[0])
                    yref, xref, bref, zref = data.loc[data[varname].isin(yrref2), outputvars], \
                        data.loc[data[varname].isin(yrref2), inputvars], \
                        data.loc[data[varname].isin(yrref2), unoutputvars], \
                        data.loc[data[varname].isin(yrref2), zvars] if type(zvars) != type(None) else None
                except:
                    yref, xref, bref, zref = None, \
                        None, \
                        None, \
                        None

    else:
        y, x, b,z = data.loc[:, outputvars], data.loc[:, inputvars], data.loc[:, unoutputvars],\
                    data.loc[:, zvars] if type(zvars) != type(None) else None

        if type(refindex) != type(None):
            varname = refindex.split('=')[0]
            yrref = ast.literal_eval(refindex.split('=')[1])
            yr = list(data[varname].unique())
            if len(set(yr) - set(yrref)) > 0:
                print("ssssssssssssss1111111")
                raise ValueError(
                    "You must specify basic data smaller than reference data.")
            else:
                print("ssssssssssssss22222222")
                yrref2 = list(set(yrref) - set(yr))
                try:
                    print(yrref2[0])
                    yref, xref, bref, zref = data.loc[data[varname].isin(yrref2), outputvars], \
                        data.loc[data[varname].isin(yrref2), inputvars], \
                        data.loc[data[varname].isin(yrref2), unoutputvars], \
                        data.loc[data[varname].isin(yrref2), zvars] if type(zvars) != type(None) else None
                except:
                    yref, xref, bref, zref = None, \
                        None, \
                        None, \
                        None
        elif type(refindex) == type(None):
            yref, xref, bref, zref = None, \
                None, \
                None, \
                None


    if type(yref) != type(None):
        referenceflag = True
    else:
        referenceflag = False

    # print("1",y)
    # print("2",yref)
    # print("3",referenceflag)
    return y,x,b,z,yref,xref,bref,zref,referenceflag

def assert_valid_yxbz(sent,gy,gx,gb,z=None):
    inputvars = sent.split('=')[0].strip(' ').split(' ')

    try:
        outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
        unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    except:
        outputvars = sent.split('=')[1].strip(' ').split(' ')
        unoutputvars = None

    if type(z)!=type(None):
        zvars = z.strip(' ').split(" ")
    else:
        zvars = None
    if len(outputvars) !=  gy.shape[1]:
        raise ValueError("Number of outputs must be the same in y and gy.")
    if len(inputvars) != gx.shape[1]:
        raise ValueError("Number of inputs must be the same in x and gx.")

    if type(gb) != type(None):
        if len(unoutputvars) != gb.shape[1]:
            raise ValueError(
                "Number of undesirable outputs must be the same in b and gb.")
        gb.columns = unoutputvars
    gy.columns,gx.columns ,=outputvars,inputvars,
    return outputvars,inputvars,unoutputvars,zvars,gy,gx,gb





def assert_valid_yxb_drf(sent,fenmu,fenzi):
    inputvars = sent.split('=')[0].strip(' ').split(' ')
    outputvars = sent.split('=')[1].split(':')[0].strip(' ').split(' ')
    unoutputvars = sent.split('=')[1].split(':')[1].strip(' ').split(' ')
    vars=inputvars +outputvars+unoutputvars
    if fenmu not in vars:
        raise ValueError("fenmu must be in sent.")
    if fenzi not in vars:
        raise ValueError("fenzi must be in sent.")

    varslt = {"inputvars": inputvars,
              "outputvars": outputvars,
              "unoutputvars": unoutputvars,
              }
    obj_coeflt = {"xobj_coef": len(inputvars) * [0],
                  "yobj_coef": len(outputvars) * [0],
                  "bobj_coef": len(unoutputvars) * [0]
                  }

    rule4_coeflt = {"xrule4_coef": len(inputvars) * [0],
                    "yrule4_coef": len(outputvars) * [0],
                    "brule4_coef": len(unoutputvars) * [0]
                    }

    for i, j in enumerate(varslt["inputvars"]):
        if fenzi == j:
            obj_coeflt["xobj_coef"][i] = 1
        if fenmu == j:
            rule4_coeflt["xrule4_coef"][i] = 1

    for i, j in enumerate(varslt["outputvars"]):
        if fenzi == j:
            obj_coeflt["yobj_coef"][i] = 1
        if fenmu == j:
            rule4_coeflt["yrule4_coef"][i] = 1
    for i, j in enumerate(varslt["unoutputvars"]):
        if fenzi == j:
            obj_coeflt["bobj_coef"][i] = 1
        if fenmu == j:
            rule4_coeflt["brule4_coef"][i] = 1

    ## 判断分母是x，b or y，是x，b的，目标要加负号。
    if (fenmu in inputvars) or (fenmu in unoutputvars):
        neg_obj = True
    elif fenmu in outputvars:
        neg_obj = False

    return outputvars, inputvars, unoutputvars, obj_coeflt, rule4_coeflt,neg_obj

