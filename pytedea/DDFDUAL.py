from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Reals, PositiveReals
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools
import ast


class DDFDUAL():

    def __init__(self, data,sent = "inputvar=outputvar:unoutputvar",
                 gy=[1], gx=[1], gb=[1], rts=RTS_VRS, baseindex=None,refindex=None):
        """DDFDUAL: Dual of Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            gy (list): output directional vector. Defaults to [1].
            gx (list): input directional vector. Defaults to [1].
            gb (list): undesirable output directional vector. Defaults to None.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.outputvars, self.inputvars,self.unoutputvars,  self.gy, self.gx, self.gb= \
            tools.assert_valid_ddf(sent,gy,gx,gb)
        self.y,self.x,self.b,self.yref,self.xref,self.bref = \
            tools.assert_valid_ddf2(data, baseindex, refindex, self.outputvars, self.inputvars, self.unoutputvars)

        self.rts = rts
        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns

        print("xcol,ycol,bcol are:",self.x.columns,self.y.columns,self.b.columns)

        print("gx,gy,gb are:",self.gx,self.gy,self.gb)

        self.I = self.x.index          ## I 是 被评价决策单元的索引
        self.__modeldict = {}
        for i in self.I:
            # print(i)
            self.I0 = i                                                 ## I 是 被评价决策单元的数量

            self.__model__ = ConcreteModel()
            # Initialize sets
            self.__model__.I2 = Set(initialize=self.xref.index)      ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样
            self.__model__.J = Set(initialize=range(len(self.b.iloc[0])))   ## B 是 非期望产出个数


            # Initialize variable

            self.__model__.beta = Var(
                self.__model__.K,initialize=1,bounds=(0.0, None), doc='beta')
            self.__model__.gamma = Var(
                self.__model__.L, initialize=1,bounds=(0.0, None),doc='gamma')
            self.__model__.delta = Var(
                self.__model__.J,bounds=(None, None), doc='delta')
            if self.rts == RTS_VRS:
                self.__model__.alpha = Var(
                    Set(initialize=range(1)), bounds=(None, None),doc='alpha')

            # Setup the objective function and constraints
            self.__model__.objective = Objective(
                rule=self.__objective_rule(), sense=minimize, doc='objective function')
            self.__model__.first = Constraint(
                self.__model__.I2,  rule=self.__first_rule(), doc='first constraint')
            self.__model__.second = Constraint(
                self.__model__.I2,  rule=self.__disposability_rule(), doc='weak disposibility')
            self.__model__.third = Constraint(
                rule=self.__translation_property(), doc='translation property')


            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):

                return sum(model.beta[k]*self.x.loc[self.I0,self.xcol[k]] for k in model.K
                    ) - sum(model.gamma[l]*self.y.loc[self.I0,self.ycol[l]] for l in model.L
                    ) + sum(model.delta[j]*self.b.loc[self.I0,self.bcol[j]] for j in model.J
                    ) + (model.alpha[0]*1 if self.rts == RTS_VRS else 0)
        return objective_rule

    def __first_rule(self):
        """Return the proper first constraint"""
        def first_rule(model, i2):

            return sum(model.beta[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                ) - sum(model.gamma[l] * self.yref.loc[i2,self.ycol[l]] for l in model.L
                ) + sum(model.delta[j] * self.bref.loc[i2,self.bcol[j]] for j in model.J
                ) + (model.alpha[0]*1 if self.rts == RTS_VRS else 0)   >=0
        return first_rule

    def __disposability_rule(self):
        """Return the proper weak disposibility constraint"""
        def disposability_rule(model, i2):

            return sum(model.beta[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                ) + (model.alpha[0]*1 if self.rts == RTS_VRS else 0)   >=0
        return disposability_rule

    def __translation_property(self):
        """Return the proper translation property"""
        def translation_rule(model):
            return sum(model.beta[ k] * self.gx[k] for k in model.K) \
                + sum(model.gamma[ l] * self.gy[l] for l in model.L) \
                + sum(model.delta[ j] * self.gb[j] for j in model.J) == 1

        return translation_rule


    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization


        if self.rts == RTS_CRS:

            data2,beta,gamma,delta= pd.DataFrame(),{},{},{}
            for ind, problem in self.__modeldict.items():
                _, data2.loc[ind,"optimization_status"]= tools.optimize_model2(problem, ind, solver)
                beta[ind]= np.asarray(list(problem.beta[:].value))
                gamma[ind]= np.asarray(list(problem.gamma[:].value))
                delta[ind]= np.asarray(list(problem.delta[:].value))

            beta = pd.DataFrame(beta, index="beta"+self.xcol).T
            gamma = pd.DataFrame(gamma,index="gamma"+self.ycol ).T
            delta = pd.DataFrame(delta,index="delta"+self.bcol ).T

            data3 = pd.concat([data2,beta],axis=1)
            data3 = pd.concat([data3,gamma],axis=1)
            data3 = pd.concat([data3,delta],axis=1)
        else:
            data2,alpha,beta,gamma,delta= pd.DataFrame(),{},{},{},{}
            for ind, problem in self.__modeldict.items():
                _, data2.loc[ind,"optimization_status"]= tools.optimize_model2(problem, ind, solver)
                alpha[ind]= np.asarray(list(problem.alpha[:].value))
                beta[ind]= np.asarray(list(problem.beta[:].value))
                gamma[ind]= np.asarray(list(problem.gamma[:].value))
                delta[ind]= np.asarray(list(problem.delta[:].value))

            alpha = pd.DataFrame(alpha, index=["alpha"]).T
            beta = pd.DataFrame(beta, index="beta"+self.xcol).T
            gamma = pd.DataFrame(gamma,index="gamma"+self.ycol ).T
            delta = pd.DataFrame(delta,index="delta"+self.bcol ).T

            data3 = pd.concat([data2,alpha],axis=1)
            data3 = pd.concat([data3,beta],axis=1)
            data3 = pd.concat([data3,gamma],axis=1)
            data3 = pd.concat([data3,delta],axis=1)

        # print("aaa", data3)
        print("与期望产出有关的边际减排成本的计算方法为：-1*期望产出价格*delta/gamma")
        print("与投入有关的边际减排成本的计算方法为：投入价格*delta/beta")
        print("自己计算一下")

        return data3


    def info(self, dmu = "all"):
        """Show the infomation of the lp model

        Args:
            dmu (string): The solver chosen for optimization. Default is "all".
        """
        if dmu =="all":
            for ind, problem in list(self.__modeldict.items()):
                print(ind,"\n",problem.pprint())

        print(self.__modeldict[int(dmu)].pprint())

