# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint,PositiveReals
from pyomo.core.expr.numvalue import NumericValue
from .constant import CET_ADDI, FUN_COST, FUN_PROD, RTS_VRS, RTS_CRS,OPT_DEFAULT, OPT_LOCAL
from .utils import tools
from . import  weakCNLSy
import pandas as pd
import numpy as np


class weakCNLSNDDF_notransfer(weakCNLSy.weakCNLSy):
    """Convex Nonparametric Least Square with non-radial directional distance function
    """

    def __init__(self,data,sent, z=None, gy=[1], gx=[1], gb=[1],weight =None, \
                 fun=FUN_PROD, rts=RTS_VRS, baseindex=None,refindex=None):
        """weakCNLS NDDF model

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars: unoutputvars. e.g.: "K L = Y : CO2"
            tau (float): quantile.
            z (float, optional): Contextual variable(s). Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to [1].
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        self.outputvars, self.inputvars, self.unoutputvars, self.zvars, self.gy, self.gx, self.gb \
            = tools.assert_valid_weakCNLSDDF(sent, gy, gx, gb, z)
        self.y, self.x, self.b, self.z, self.yref, self.xref, self.bref, self.zref,self.referenceflag\
            = tools.assert_valid_weakCNLSDDF2(baseindex, refindex, data, \
                                       self.outputvars, self.inputvars, self.unoutputvars, self.zvars)
        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns
        self.zcol = self.z.columns if type(z) != type(None) else None

        self.fun = fun
        self.rts = rts


        if type(weight) != type(None):
            self.weight= weight
        else:
            self.weight=[]
            xhas1index = abs(np.asarray(self.gx)).sum() > 0
            yhas1index = abs(np.asarray(self.gy)).sum() > 0
            bhas1index = abs(np.asarray(self.gb)).sum() > 0

            fenmu = 1*xhas1index + 1*yhas1index + 1*bhas1index
            print(fenmu)
            for _ in self.gx:
                if _==0:
                    self.weight.append(0)
                else:
                    self.weight.append(1/fenmu/abs(np.asarray(gx)).sum())
            for _ in self.gy:
                if _==0:
                    self.weight.append(0)
                else:
                    self.weight.append(1/fenmu/abs(np.asarray(gy)).sum())
            for _ in self.gb:
                if _ == 0:
                    self.weight.append(0)
                else:
                    self.weight.append(1/fenmu/abs(np.asarray(gb)).sum())


        self.iweight = self.weight[0:len(self.x.iloc[0])]
        self.oweight = self.weight[len(self.x.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])]
        self.bweight = self.weight[len(self.x.iloc[0])+len(self.y.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])+len(self.b.iloc[0])]

        print("xcol,ycol,bcol are:", self.x.columns, self.y.columns, self.b.columns)
        print(self.iweight,self.oweight,self.bweight)
        print("gx,gy,gb are:", self.gx, self.gy, self.gb)
        print("x,y,b are:", self.x, self.y, self.b)

        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=self.x.index)  ## I 是 被评价决策单元的数量
        if self.referenceflag:
            # self.__model__.I2 = Set(initialize=  [*map(lambda x : x[1] ,self.xref.index)]   )  ## I2 是 参考决策单元的数量
            self.__model__.I2 = Set(initialize=self.xref.index)  ## I2 是 参考决策单元的数量
        self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))  ## K 是投入个数
        self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))  ## L 是产出个数 被评价单元和参考单元的K，L一样
        self.__model__.J = Set(initialize=range(len(self.b.iloc[0])))  ## B 是 非期望产出个数

        # Initialize the variables
        if self.rts == RTS_VRS:
            self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(
            self.__model__.I, self.__model__.K, bounds=(0.0, None), doc='beta')
        self.__model__.gamma = Var(
            self.__model__.I, self.__model__.L, bounds=(0.0, None), doc='gamma')
        self.__model__.delta = Var(
            self.__model__.I, self.__model__.J, bounds=(0.0, None),within=PositiveReals, doc='delta')

        self.__model__.epsilon = Var(self.__model__.I,  bounds=(None, None),doc='residuals')

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.M = Set(initialize=range(len(self.z.iloc[0])))
            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.M, doc='z coefficient')
            # self.__model__.lamda.pprint()
            # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        # self.__model__.regression_rule.pprint()
        self.__model__.translation_rule1 = Constraint(self.__model__.I,self.__model__.K,
                                                     rule=self.__translation_property1(),
                                                     doc='translation property1')
        self.__model__.translation_rule2 = Constraint(self.__model__.I,self.__model__.L,
                                                     rule=self.__translation_property2(),
                                                     doc='translation property2')
        self.__model__.translation_rule3 = Constraint(self.__model__.I,self.__model__.J,
                                                     rule=self.__translation_property3(),
                                                     doc='translation property3')
        # self.__model__.translation_rule.pprint()
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')
        # self.__model__.afriat_rule.pprint()
        self.__model__.disposability_rule = Constraint(self.__model__.I,
                                                        self.__model__.I,
                                                        rule=self.__disposability_rule(),
                                                        doc='weak disposibility')
        # self.__model__.disposability_rule.pprint()
        if self.referenceflag:
            self.__model__.afriat_ref_rule = Constraint(self.__model__.I,
                                                       self.__model__.I2,
                                                    rule=self.__afriat_ref_rule(),
                                                    doc='afriat reference inequality')
            # self.__model__.afriat_ref_rule.pprint()

            self.__model__.disposability_ref_rule = Constraint(self.__model__.I,
                                                       self.__model__.I2,
                                                    rule=self.__disposability_ref_rule(),
                                                    doc='weak disposability reference ')
            # self.__model__.disposability_ref_rule.pprint()
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

    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I)
        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.rts == RTS_VRS:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                        == model.alpha[i] \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                        + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                        - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                        - model.epsilon[i]

                return regression_rule
            elif type(self.z) == type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                        == model.alpha[i] \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                        + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                        - model.epsilon[i]
                return regression_rule

        elif self.rts == RTS_CRS:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                        == \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                        + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                        - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                        - model.epsilon[i]
                return regression_rule

            elif type(self.z) == type(None):
                def regression_rule(model, i):
                    return sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                        == \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                        + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                        - model.epsilon[i]
                return regression_rule

        raise ValueError("Undefined model parameters.")

    def __translation_property1(self):
        """Return the proper translation property"""
        def translation_rule1(model, i, k):
            if self.gx[k]==0:
                return Constraint.Skip
            return model.beta[i, k] * self.gx[k] >= self.iweight[k]
        return translation_rule1

    def __translation_property2(self):
        """Return the proper translation property"""
        def translation_rule2(model, i,l):
            if self.gy[l]==0:
                return Constraint.Skip
            return model.gamma[i, l] * self.gy[l] >= self.oweight[l]
        return translation_rule2

    def __translation_property3(self):
        """Return the proper translation property"""
        def translation_rule3(model, i, j):
            if self.gb[j]==0:
                return Constraint.Skip
            return model.delta[i, j] * self.gb[j]  >= self.bweight[j]
        return translation_rule3

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
                                  + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                                  + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                                  - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L),
                                  model.alpha[h]
                                  + sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                                  + sum(model.delta[h, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                                  - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]] for l in model.L))
            return afriat_rule

        elif self.rts == RTS_CRS:
            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(  sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]]  for k in model.K) \
                                  + sum(model.delta[i, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                                  - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L),
                                    sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                                  + sum(model.delta[h, j] * self.b.loc[i,self.bcol[j]] for j in model.J) \
                                  - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]] for l in model.L))
            return afriat_rule

        raise ValueError("Undefined model parameters.")

    def __disposability_rule(self):
        """Return the proper weak disposability constraint"""
        if self.rts == RTS_VRS:
            def disposability_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return model.alpha[i] + sum(model.beta[i, k] * self.x.loc[h,self.xcol[k]] for k in model.K) >= 0
            return disposability_rule

        elif self.rts == RTS_CRS:
            def disposability_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return sum(model.beta[i, k] * self.x.loc[h,self.xcol[k]] for k in model.K) >= 0
            return disposability_rule

        raise ValueError("Undefined model parameters.")

    def __afriat_ref_rule(self):
        """Return the proper afriat inequality constraint for reference"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
            __operator = NumericValue.__ge__
        if self.rts == RTS_VRS:
            def afriat_ref_rule(model, i, h):
                # if i == h:
                #     return Constraint.Skip
                return __operator(sum(0 *model.beta[i, k] for k in model.K),
                  model.alpha[i] \
                  + sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) \
                  + sum(model.delta[i, j] * self.bref.loc[h,self.bcol[j]] for j in model.J) \
                  - sum(model.gamma[i, l] * self.yref.loc[h,self.ycol[l]] for l in model.L))
            return afriat_ref_rule

        elif self.rts == RTS_CRS:
            def afriat_ref_rule(model, i, h):
                # if i == h:
                #     return Constraint.Skip
                return __operator(sum(0 *model.beta[i, k] for k in model.K),
                  sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) \
                  + sum(model.delta[i, j] * self.bref.loc[h,self.bcol[j]] for j in model.J) \
                  - sum(model.gamma[i, l] * self.yref.loc[h,self.ycol[l]] for l in model.L))
            return afriat_ref_rule

        raise ValueError("Undefined model parameters.")

    def __disposability_ref_rule(self):
        """Return the proper weak disposability constraint for reference"""
        if self.rts == RTS_VRS:
            if type(self.z) != type(None):
                def disposability_ref_rule(model, i, h):
                    return model.alpha[i] + sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) >= 0
                return disposability_ref_rule
            elif type(self.z) == type(None):
                def disposability_ref_rule(model, i, h):
                    return model.alpha[i] + sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) >= 0
                return disposability_ref_rule
        elif self.rts == RTS_CRS:
            def disposability_ref_rule(model, i, h):
                return sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) >= 0
            return disposability_ref_rule

        raise ValueError("Undefined model parameters.")

    def info(self):
        return self.__model__.pprint()

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
