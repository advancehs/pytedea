# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, OPT_DEFAULT, RTS_CRS, RTS_VRS, OPT_LOCAL,RED_QLE
from .utils import tools


class weakMetaCNLSb:
    """Convex Nonparametric Least Square with weak disposability (weakCNLSb)
        lnb=ln(\gamma y -\beta x -\alpha) - \epsilon(\epsilon<0)
    """

    def __init__(self, data,GCE,sent, z=None, cet=CET_ADDI, \
                 fun=FUN_PROD, rts=RTS_VRS,baseindex=None,refindex=None):
        """weakCNLSb model

        Args:
            GCE(pd.DataFrame): 分组后求的GCE.
            sent (str): inputvars=outputvars: unoutputvars. e.g.: "K L = Y:CO2 "
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        self.outputvars,self.inputvars,self.unoutputvars,self.zvars = tools.assert_valid_weakCNLS(sent, z)

        self.y, self.x, self.b, self.z, self.yref, self.xref, self.bref, self.zref, self.referenceflag\
            = tools.assert_valid_weakCNLSDDF2(baseindex,refindex,data,\
                                       self.outputvars,self.inputvars,self.unoutputvars,self.zvars)
        self.x["one"] = 1
        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns
        self.zcol = self.z.columns if type(z) != type(None) else None

        print("xcol,ycol,bcol are:",self.x.columns,self.y.columns,self.b.columns)

        # print("xcol,ycol,bcol are:",self.xcol,self.ycol[0],self.bcol)

        self.cet = cet
        self.fun = fun
        self.rts = rts


        self.GCE = GCE
        print("GCE is:",self.GCE)

        self.GCE_C = pd.DataFrame(self.GCE.iloc[:,0].to_numpy() *self.b.iloc[:,0].to_numpy(),index=self.x.index)
        print("GCE_C",self.GCE_C)

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()


        # Initialize the sets
        self.__model__.I = Set(initialize=self.x.index)  ## I 是 被评价决策单元的数量
        if self.referenceflag:
            self.__model__.I2 = Set(initialize=self.xref.index)  ## I2 是 参考决策单元的数量
        self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))  ## K 是投入个数
        # self.__model__.J = Set(initialize=range(len(self.b.iloc[0])))  ## B 是 非期望产出个数
        self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))  ## L 是产出个数 被评价单元和参考单元的K，L一样
        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.M = Set(initialize=range(len(self.z.iloc[0])))
            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.M, doc='z coefficient')
        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.K,
                                  bounds=(0.0, None),  ## i行 j列x
                                  doc='beta')
        self.__model__.gamma = Var(self.__model__.I,
                                   self.__model__.L,
                                   bounds=(0.0, None),
                                   doc='gamma')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')
        self.__model__.frontier = Var(self.__model__.I,
                                      bounds=(0.0, None),
                                      doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        if self.cet == CET_MULT:
            self.__model__.log_rule = Constraint(self.__model__.I,
                                                 rule=self.__log_rule(),
                                                 doc='log-transformed regression equation')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')
        self.__model__.disposability_rule = Constraint(self.__model__.I,
                                                        self.__model__.I,
                                                        rule=self.__disposability_rule(),
                                                        doc='weak disposibility')
        if self.referenceflag:
            self.__model__.afriat_ref_rule = Constraint(self.__model__.I,
                                                       self.__model__.I2,
                                                    rule=self.__afriat_ref_rule(),
                                                    doc='afriat reference inequality')

            self.__model__.disposability_ref_rule = Constraint(self.__model__.I,
                                                       self.__model__.I2,
                                                    rule=self.__disposability_ref_rule(),
                                                    doc='weak disposability reference ')
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
            self.__model__, email, self.cet, solver)

    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I)
        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.cet == CET_ADDI:
            if self.rts == RTS_VRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):

                        return np.array(1/ self.GCE_C.loc[i,]) == model.alpha[i] \
                                + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                                - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                                - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                                - model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return np.array(1/ self.GCE_C.loc[i,]) == -model.alpha[i] \
                            + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                            - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                            - model.epsilon[i]

                return regression_rule
            elif self.rts == RTS_CRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return np.array(1/ self.GCE_C.loc[i,]) == \
                                sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                                - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]] for l in model.L) \
                                - sum(model.lamda[m] * self.z.loc[i,self.ycol[m]] for m in model.M) \
                                - model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return np.array(1/ self.GCE_C.loc[i,]) == \
                            sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                            - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]]  for l in model.L) \
                            - model.epsilon[i]

                return regression_rule

        elif self.cet == CET_MULT:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(np.array(1/ self.GCE_C.loc[i,])) == log(model.frontier[i] + 1) \
                            - sum(model.lamda[m] * self.z.loc[i,self.ycol[m]]  for m in model.M) \
                            - model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return log(np.array(1/ self.GCE_C.loc[i,])) == log(model.frontier[i] + 1) \
                        - model.epsilon[i]

            return regression_rule

        raise ValueError("Undefined model parameters.")

    def __log_rule(self):
        """Return the proper log constraint"""
        if self.cet == CET_MULT:
            if self.rts == RTS_VRS:

                def log_rule(model, i):
                    return model.frontier[i] == model.alpha[i] + sum(
                        model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                            - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]]  for l in model.L) - 1

                return log_rule
            elif self.rts == RTS_CRS:

                def log_rule(model, i):
                    return model.frontier[i] == sum(
                        model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                            - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]]  for l in model.L) - 1

                return log_rule

        raise ValueError("Undefined model parameters.")

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
                return __operator(
                    model.alpha[i] + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K)
                                  - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]]  for l in model.L),
                    model.alpha[h] + sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K)
                                   - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]]  for l in model.L) )

            return afriat_rule
        elif self.rts == RTS_CRS:

            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                     sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K)
                    - sum(model.gamma[i, l] * self.y.loc[i,self.ycol[l]]  for l in model.L),
                     sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K)
                    - sum(model.gamma[h, l] * self.y.loc[i,self.ycol[l]]  for l in model.L))

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
                  - sum(model.gamma[i, l] * self.yref.loc[h,self.ycol[l]] for l in model.L))
            return afriat_ref_rule

        elif self.rts == RTS_CRS:
            def afriat_ref_rule(model, i, h):
                # if i == h:
                #     return Constraint.Skip
                return __operator(sum(0 *model.beta[i, k] for k in model.K),
                  sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) \
                  - sum(model.gamma[i, l] * self.yref.loc[h,self.ycol[l]] for l in model.L))
            return afriat_ref_rule

        raise ValueError("Undefined model parameters.")

    def __disposability_ref_rule(self):
        """Return the proper weak disposability constraint"""
        if self.rts == RTS_VRS:
            def disposability_ref_rule(model, i, h):
                # if i == h:
                #     return Constraint.Skip
                return model.alpha[i] + sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) >= 0
            return disposability_ref_rule

        elif self.rts == RTS_CRS:
            def disposability_ref_rule(model, i, h):
                # if i == h:
                #     return Constraint.Skip
                return sum(model.beta[i, k] * self.xref.loc[h,self.xcol[k]] for k in model.K) >= 0
            return disposability_ref_rule

        raise ValueError("Undefined model parameters.")


    def display_status(self):
        """Display the status of problem"""
        tools.assert_optimized(self.optimization_status)
        print(self.display_status)

    def display_alpha(self):
        """Display alpha value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.beta.display()

    def display_lamda(self):
        """Display lamda value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        self.__model__.lamda.display()

    def display_residual(self):
        """Dispaly residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon.display()

    def display_delta(self):
        """Display delta value"""
        raise ValueError("weakCNLSb has no delta.")


    def get_status(self):
        """Return status"""
        return self.optimization_status


    def get_alpha(self):
        """Return alpha value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        alpha = pd.Series(self.__model__.alpha.extract_values())
        return alpha

    def get_beta(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        beta = pd.Series(self.__model__.beta.extract_values(),index=self.__model__.beta.extract_values().keys())
        # if the series is multi-indexed we need to unstack it...
        if type(beta.index[0]) == tuple:  # it is multi-indexed
            beta = beta.unstack(level=1)
        else:
            beta = pd.DataFrame(beta)  # force transition from Series -> df
        # multi-index the columns
        beta.columns = map(lambda x: "beta"+str(x) ,beta.columns)
        return beta

    def get_residual(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = pd.Series(self.__model__.epsilon.extract_values())
        return residual

    def get_lamda(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        lamda = pd.Series(self.__model__.lamda.extract_values(),index=self.__model__.lamda.extract_values().keys())
        # if the series is multi-indexed we need to unstack it...
        if type(lamda.index[0]) == tuple:  # it is multi-indexed
            lamda = lamda.unstack(level=1)
        else:
            lamda = pd.DataFrame(lamda)  # force transition from Series -> df
        # multi-index the columns
        lamda.columns = map(lambda x: "beta"+str(x) ,lamda.columns)
        return lamda

    def get_delta(self):
        """Return delta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_undesirable_output(self.b)
        delta = pd.Series(self.__model__.delta.extract_values(),index=self.__model__.delta.extract_values().keys())
        # if the series is multi-indexed we need to unstack it...
        if type(delta.index[0]) == tuple:  # it is multi-indexed
            delta = delta.unstack(level=1)
        else:
            delta = pd.DataFrame(delta)  # force transition from Series -> df
        # multi-index the columns
        delta.columns = map(lambda x: "beta"+str(x) ,delta.columns)
        return delta

    def display_gamma(self):
        """Display delta value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_desirable_output(self.y)
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
        tools.assert_optimized(self.optimization_status)
        if self.cet == CET_MULT and type(self.z) == type(None):
            frontier = np.asarray(list(self.__model__.frontier[:].value)) + 1
        elif self.cet == CET_MULT and type(self.z) != type(None):
            frontier = list(np.divide(np.exp(
                self.get_residual() + self.get_lamda() * np.asarray(self.z)[:, 0]), self.b) - 1)
        elif self.cet == CET_ADDI:
            frontier = -np.asarray(self.b) + self.get_residual()
        return np.asarray(frontier)

    def get_residual2(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)

    def get_gamma2(self):  ## 用于计算效率
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()
