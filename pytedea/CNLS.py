# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, OPT_DEFAULT, RTS_CRS, RTS_VRS, OPT_LOCAL
from .utils import tools, interpolation

class CNLS:
    """Convex Nonparametric Least Square (CNLS)
    """

    def __init__(self, data,sent, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS,baseindex=None,refindex=None):
        """CNLS model

        Args:
            sent (str): inputvars=outputvars:unoutputvars. e.g.: "K L CO2 = Y "
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist

        self.outputvars,self.inputvars,self.zvars = tools.assert_valid_CNLS(sent, z)

        self.y, self.x,  self.z, self.yref, self.xref,  self.zref,self.referenceflag\
            = tools.assert_valid_CNLS2(baseindex,refindex,data,\
                                       self.outputvars,self.inputvars,self.zvars)

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.zcol = self.z.columns if type(z) != type(None) else None
        print("xcol,ycol are:",self.x.columns,self.y.columns)

        self.cet = cet
        self.fun = fun
        self.rts = rts

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=self.x.index)  ## I 是 被评价决策单元的数量
        self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))  ## K 是投入个数
        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.M = Set(initialize=range(len(self.z.iloc[0])))
            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.M, doc='z coefficient')
        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.K,
                                  bounds=(0.0, None),
                                  doc='beta')
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
                        return np.array((self.y.loc[i,])) \
                            == model.alpha[i] \
                                + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                                - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                                - model.epsilon[i]
                    return regression_rule


                def regression_rule(model, i):
                    return np.array((self.y.loc[i,]))   == model.alpha[i] \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                        - model.epsilon[i]
                return regression_rule

            elif self.rts == RTS_CRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return np.array((self.y.loc[i,]))   == \
                                sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                                - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                                - model.epsilon[i]
                    return regression_rule

                def regression_rule(model, i):
                    return np.array((self.y.loc[i,])) == \
                        sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) \
                        - model.epsilon[i]
                return regression_rule

        elif self.cet == CET_MULT:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(np.array(self.y.loc[i,:]) ) == log(model.frontier[i] + 1) \
                            - sum(model.lamda[m] * self.z.loc[i,self.zcol[m]] for m in model.M) \
                            - model.epsilon[i]
                return regression_rule


            def regression_rule(model, i):
                return log(np.array(self.y.loc[i,:]) ) ==log(model.frontier[i] + 1) \
                            - model.epsilon[i]
            return regression_rule

        raise ValueError("Undefined model parameters.")

    def __log_rule(self):
        """Return the proper log constraint"""
        if self.cet == CET_MULT:
            if self.rts == RTS_VRS:
                def log_rule(model, i):
                    return model.frontier[i] == model.alpha[i]  \
                        + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) - 1
                return log_rule

            elif self.rts == RTS_CRS:
                def log_rule(model, i):
                    return model.frontier[i] ==\
                         sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K) - 1
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
                    model.alpha[i] \
                    + sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K ),
                    model.alpha[h]\
                    + sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K) )
            return afriat_rule

        elif self.rts == RTS_CRS:
            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                    sum(model.beta[i, k] * self.x.loc[i,self.xcol[k]] for k in model.K),
                    sum(model.beta[h, k] * self.x.loc[i,self.xcol[k]] for k in model.K))
            return afriat_rule

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

    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        if self.cet == CET_MULT and type(self.z) == type(None):
            frontier = np.asarray(list(self.__model__.frontier[:].value)) + 1
        elif self.cet == CET_MULT and type(self.z) != type(None):
            frontier = list(np.multiply(self.y, np.exp(
                self.get_residual() + self.get_lamda() * np.asarray(self.z)[:, 0])) - 1)
        elif self.cet == CET_ADDI:
            frontier = np.asarray(self.y) + self.get_residual()
        return np.asarray(frontier)

    def info(self):
        return self.__model__.pprint()

    def get_residual2(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)


    def get_adjusted_residual(self):
        """Return the shifted residuals(epsilon) tern by CCNLS"""
        tools.assert_optimized(self.optimization_status)
        return self.get_residual() - np.amax(self.get_residual())

    def get_adjusted_alpha(self):
        """Return the shifted constatnt(alpha) term by CCNLS"""
        tools.assert_optimized(self.optimization_status)
        return self.get_alpha() + np.amax(self.get_residual())

    def get_predict(self, x_test):
        """Return the estimated function in testing sample"""
        tools.assert_optimized(self.optimization_status)
        return interpolation.interpolation(self.get_alpha(), self.get_beta(), x_test, fun=self.fun)
