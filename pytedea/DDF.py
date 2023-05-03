"""Main module."""
# import dependencies

from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Reals,PositiveReals
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_HYPERYB, RTS_VRS, RTS_CRS,EMF_SAME, EMF_DIFFERENT,OPT_DEFAULT, OPT_LOCAL
from .utils import tools
from .DEA import DEA
import ast

class DDF(DEA):
    def __init__(self, data,sent = "inputvar=outputvar",  \
                 gy=[1], gx=[1],gb=[1],rts=RTS_VRS,emf=EMF_SAME, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L CO2= Y"
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to [1].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            emf (String): EMF_SAME (same emission abatement factor) or EMF_DIFFERENT (different emission abatement factor)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.outputvars, self.inputvars,self.unoutputvars,  self.gy, self.gx, self.gb= \
            tools.assert_valid_ddf(sent,gy,gx,gb)
        self.y,self.x,self.b,self.yref,self.xref,self.bref = \
            tools.assert_valid_ddf2(data, baseindex, refindex, self.outputvars, self.inputvars, self.unoutputvars)

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns
        self.rts = rts
        self.emf = emf

        self.I = self.x.index          ## I 是 被评价决策单元的索引
        self.__modeldict = {}
        for i in self.I:
            # print(i)
            self.I0 = i                                                 ## I 是 被评价决策单元的数量

            self.__model__ = ConcreteModel()
            # Initialize sets
            self.__model__.I2 = Set(initialize= self.xref.index)                     ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))   ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))   ## L 是产出个数 被评价单元和参考单元的K，L一样
            self.__model__.J = Set(initialize=range(len(self.b.iloc[0])))   ## J 是非期望产出个数


            # Initialize variable
            self.__model__.beta = Var(Set(initialize=range(1)),bounds=(0.0, None), \
                                      within=Reals,doc='directional distance')
            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')
            if self.rts == RTS_VRS:
                if self.emf == EMF_SAME:
                    if abs(np.asarray(self.gx).sum())>=1:
                        self.__model__.beta2 = Var(Set(initialize=range(1)), bounds=(0.0, None), \
                                                  within=Reals, doc='beta*theta')
                    self.__model__.theta = Var(Set(initialize=range(1)), bounds=(0.0, 1.0), \
                                              within=Reals, doc='theta')
                elif self.emf == EMF_DIFFERENT:
                    self.__model__.phi = Var(self.__model__.I2, bounds=(0.0, None), within=Reals, doc='phi_i')
                    self.__model__.mu = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='mu_i')
                else:
                    raise ValueError("Undefined model parameters.")

            # Setup the objective function and constraints
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(
                self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(
                self.__model__.L,  rule=self.__output_rule(), doc='output constraint')
            self.__model__.undesirable_output = Constraint(
                self.__model__.J,  rule=self.__undesirable_output_rule(), doc='undesirable output constraint')


            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                if abs(np.asarray(self.gx).sum()) >= 1  :
                    def objective_rule(model):
                        return model.beta2[0] * 1
                else:
                    def objective_rule(model):
                        return model.beta[0]*1
            elif self.emf == EMF_DIFFERENT:
                def objective_rule(model):
                    return model.beta[0] * 1
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            def objective_rule(model):
                return model.beta[0] * 1
        else:
            raise ValueError("Undefined model parameters.")
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        if self.rts == RTS_VRS: # gx=1,0 gy=1,0 gb=1,0
            if self.emf == EMF_SAME:
                if abs(np.asarray(self.gx).sum()) >= 1:
                    def input_rule(model, k):
                        return sum(model.lamda[i2] * self.xref.loc[i2, self.xcol[k]] for i2 in model.I2
                                   ) - model.beta2[0] * self.gx[k] * self.x.loc[self.I0, self.xcol[k]] <= \
                            model.theta[0]*self.x.loc[self.I0, self.xcol[k]]
                else:
                    def input_rule(model, k):
                        return sum(model.lamda[i2] * self.xref.loc[i2, self.xcol[k]] for i2 in model.I2
                                   ) - model.beta[0] * self.gx[k] * self.x.loc[self.I0, self.xcol[k]] <= \
                            model.theta[0] * self.x.loc[self.I0, self.xcol[k]]
            elif self.emf == EMF_DIFFERENT:
                def input_rule(model, k):
                    return sum((model.phi[i2]+ model.mu[i2])* self.xref.loc[i2, self.xcol[k]] for i2 in model.I2
                               ) - model.beta[0] * self.gx[k] * self.x.loc[self.I0, self.xcol[k]] <= \
                        self.x.loc[self.I0, self.xcol[k]]
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            def input_rule(model, k):
                return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                        ) - model.beta[0]*self.gx[k]*self.x.loc[self.I0,self.xcol[k]] <= \
                            self.x.loc[self.I0,self.xcol[k]]
        else:
            raise ValueError("Undefined model parameters.")
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                def output_rule(model, l):
                    return -sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                            ) + model.beta[0]*self.gy[l] *self.y.loc[self.I0,self.ycol[l]]<=\
                                -self.y.loc[self.I0,self.ycol[l]]
            elif self.emf == EMF_DIFFERENT:
                def output_rule(model, l):
                    return -sum(model.phi[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                            ) + model.beta[0]*self.gy[l] *self.y.loc[self.I0,self.ycol[l]]<=\
                                -self.y.loc[self.I0,self.ycol[l]]
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            def output_rule(model, l):
                return -sum(model.lamda[i2] * self.yref.loc[i2, self.ycol[l]] for i2 in model.I2
                            ) + model.beta[0] * self.gy[l] * self.y.loc[self.I0, self.ycol[l]] <= \
                    -self.y.loc[self.I0, self.ycol[l]]

        else:
            raise ValueError("Undefined model parameters.")
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                def undesirable_output_rule(model, b):
                    return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                            ) -model.beta[0]*self.gb[b]*self.b.loc[self.I0,self.bcol[b]]== \
                        self.b.loc[self.I0,self.bcol[b]]
            elif self.emf == EMF_DIFFERENT:
                def undesirable_output_rule(model, b):
                    return sum(model.phi[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                            ) -model.beta[0]*self.gb[b]*self.b.loc[self.I0,self.bcol[b]]== \
                        self.b.loc[self.I0,self.bcol[b]]
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            def undesirable_output_rule(model, b):
                return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[b]] for i2 in model.I2
                           ) - model.beta[0] * self.gb[b] * self.b.loc[self.I0, self.bcol[b]] == \
                    self.b.loc[self.I0, self.bcol[b]]
        else:
            raise ValueError("Undefined model parameters.")
        return undesirable_output_rule

    def __vrs_rule(self):
        if self.emf == EMF_SAME:
            def vrs_rule(model):
                return sum(model.lamda[ i2] for i2 in model.I2) == model.theta[0]
        elif self.emf == EMF_DIFFERENT:
            def vrs_rule(model):
                return sum((model.phi[i2]+ model.mu[i2]) for i2 in model.I2) == 1
        else:
            raise ValueError("Undefined model parameters.")
        return vrs_rule


    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                if abs(np.asarray(self.gx).sum()) >= 1:
                    data2,beta,theta,beta2 = pd.DataFrame(),{},{}, {}
                    for ind, problem in self.__modeldict.items():
                        _, data2.loc[ind, "optimization_status"] = tools.optimize_model2(problem, ind, solver)
                        beta2[ind], = np.asarray(list(problem.beta2[:].value))
                        theta[ind], = np.asarray(list(problem.theta[:].value))

                    theta = pd.DataFrame(theta, index=["theta"]).T
                    beta2 = pd.DataFrame(beta2, index=["beta2"]).T
                    print("aaa",beta2)
                    print("bbb",theta)

                    data3 = pd.concat([data2, theta], axis=1)
                    data3 = pd.concat([data3, beta2], axis=1)
                    data3['beta'] = data3['beta2'] /data3['theta']
                    print("ddd",data3)

                else:
                    data2,beta, theta= pd.DataFrame(),{},{}
                    for ind, problem in self.__modeldict.items():
                        _, data2.loc[ind,"optimization_status"]= tools.optimize_model2(problem, ind, solver)
                        beta[ind],= np.asarray(list(problem.beta[:].value))
                        theta[ind],= np.asarray(list(problem.theta[:].value))
                    beta = pd.DataFrame(beta, index=["beta"]).T
                    theta = pd.DataFrame(theta, index=["theta"]).T

                    data3 = pd.concat([data2, beta], axis=1)
                    data3 = pd.concat([data3, theta], axis=1)
            elif self.emf == EMF_DIFFERENT:
                data2, beta = pd.DataFrame(), {}
                for ind, problem in self.__modeldict.items():
                    _, data2.loc[ind, "optimization_status"] = tools.optimize_model2(problem, ind, solver)
                    beta[ind], = np.asarray(list(problem.beta[:].value))
                beta = pd.DataFrame(beta, index=["beta"]).T

                data3 = pd.concat([data2, beta], axis=1)
            else:
                raise ValueError("Undefined model parameters.")
        else:
            data2,beta,lamda,= pd.DataFrame(),{},{}
            for ind, problem in self.__modeldict.items():
                _, data2.loc[ind,"optimization_status"]= tools.optimize_model2(problem, ind, solver)
                beta[ind],= np.asarray(list(problem.beta[:].value))
            beta = pd.DataFrame(beta,index=["beta"]).T
            data3 = pd.concat([data2,beta],axis=1)
        print("aaa", data3)

        return data3

    def info(self, dmu = "all"):
        """Show the infomation of the lp model

        Args:
            dmu (string): The solver chosen for optimization. Default is "all".
        """
        if dmu =="all":
            for ind, problem in self.__modeldict.items():
                print(ind,"\n",problem.pprint())

        print(self.__modeldict[int(dmu)].pprint())

