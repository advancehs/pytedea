# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.core.expr.numvalue import NumericValue
import pandas as pd
import numpy as np

from . import CNLS
from .constant import CET_ADDI, FUN_COST, FUN_PROD, RTS_VRS,RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class CNLSDDF(CNLS.CNLS):
    """Convex Nonparametric Least Square with directional distance function
    """

    def __init__(self, data,sent = "inputvar=outputvar:unoutputvar",z=None, gy=[1], gx=[1], gb=None, \
                 fun=FUN_PROD, rts=RTS_VRS, baseindex=None,refindex=None):
        """CNLS DDF model

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars: unoutputvars. e.g.: "K L CO2= Y"
            z (float, optional): Contextual variable(s). Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """

        self.outputvars, self.inputvars,  self.zvars, self.gy, self.gx,  \
            = tools.assert_valid_CNLSDDF(sent, gy, gx, z)
        self.y, self.x,  self.z, self.yref, self.xref,  self.zref,self.referenceflag\
            = tools.assert_valid_CNLS2(baseindex, refindex, data, \
                                       self.outputvars, self.inputvars,  self.zvars)
        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.zcol = self.z.columns if type(z) != type(None) else None
        print("xcol,ycol are:",self.x.columns,self.y.columns)

        print("gx,gy are:",self.gx,self.gy)
        print("aaa",self.y,self.x)
        self.fun = fun
        self.rts = rts

        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=self.x.index)  ## I 是 被评价决策单元的数量
        if self.referenceflag:
            self.__model__.I2 = Set(initialize=self.xref.index)  ## I2 是 参考决策单元的数量
        self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))  ## K 是投入个数
        self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))  ## L 是产出个数 被评价单元和参考单元的K，L一样

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.M = Set(initialize=range(len(self.z.iloc[0])))
            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.M, doc='z coefficient')

        # Initialize the variables

        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(
            self.__model__.I, self.__model__.K, bounds=(0.0, None), doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residuals')
        self.__model__.gamma = Var(
            self.__model__.I, self.__model__.L, bounds=(0.0, None), doc='gamma')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self._CNLS__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.translation_rule = Constraint(self.__model__.I,
                                                     rule=self.__translation_property(),
                                                     doc='translation property')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_ADDI, solver)

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.rts == RTS_VRS:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i, self.ycol[l]] for l in model.L) \
                        == model.alpha[i] \
                        + sum(model.beta[i, k] * self.x.loc[i, self.xcol[k]] for k in model.K) \
                        - sum(model.lamda[m] * self.z.loc[i, self.zcol[m]] for m in model.M) \
                        - model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return sum(model.gamma[i, l] * self.y.loc[i, self.ycol[l]] for l in model.L) \
                    == model.alpha[i] \
                    + sum(model.beta[i, k] * self.x.loc[i, self.xcol[k]] for k in model.K) \
                    - model.epsilon[i]

            return regression_rule

        elif self.rts == RTS_CRS:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i, self.ycol[l]] for l in model.L) \
                        == sum(model.beta[i, k] * self.x.loc[i, self.xcol[k]] for k in model.K) \
                        - sum(model.lamda[m] * self.z.loc[i, self.zcol[m]] for m in model.M) \
                        - model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return sum(model.gamma[i, l] * self.y.loc[i, self.ycol[l]] for l in model.L) \
                    == sum(model.beta[i, k] * self.x.loc[i, self.xcol[k]] for k in model.K) \
                    - model.epsilon[i]

            return regression_rule

        raise ValueError("Undefined model parameters.")

    def __translation_property(self):
        """Return the proper translation property"""

        def translation_rule(model, i):
            return sum(model.beta[i, k] * self.gx[k] for k in model.K) \
                + sum(model.gamma[i, l] * self.gy[l] for l in model.L) == 1
        return translation_rule


    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
            __operator = NumericValue.__ge__
        if self.rts == RTS_VRS:
            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(model.alpha[i] \
                  + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                  - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L),
                  model.alpha[h]
                  + sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                  - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]] for l in model.L))
            return afriat_rule

        elif self.rts == RTS_CRS:
            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(model.alpha[i] \
                  + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                  - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L),
                  model.alpha[h]
                  + sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                  - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]] for l in model.L))
            return afriat_rule

        raise ValueError("Undefined model parameters.")

    def display_gamma(self):
        """Display gamma value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.gamma.display()

    def get_gamma(self):
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = pd.Series(self.__model__.gamma.extract_values(),index=self.__model__.gamma.extract_values().keys())
        # if the series is multi-indexed we need to unstack it...
        if type(gamma.index[0]) == tuple:  # it is multi-indexed
            gamma = gamma.unstack(level=1)
        else:
            gamma = pd.DataFrame(gamma)  # force transition from Series -> df
        # multi-index the columns
        gamma.columns = map(lambda x: "beta"+str(x) ,gamma.columns)
        return gamma

    def info(self):
        return self.__model__.pprint()

    def get_frontier(self):
        """Return estimated frontier value by array"""
        raise ValueError("DDF hsa no frontier.")

    def get_gamma2(self):  ## 用于计算效率
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()

    def get_residual2(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)
