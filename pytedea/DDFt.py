"""Main module."""
# import dependencies

from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Reals,PositiveReals
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_HYPERYB, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools
from .DEAt import DEAt
import ast

class DDFt(DEAt):
    def __init__(self, data,sent = "inputvar=outputvar",  gy=[1], gx=[1],rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L CO2= Y"
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.outputvars, self.inputvars,  self.gy, self.gx = tools.assert_valid_ddft(sent,gy,gx)
        self.y,self.x,self.yref,self.xref = tools.assert_valid_ddft2(data, baseindex, refindex, self.outputvars, self.inputvars)

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.rts = rts

        # if orient in [ORIENT_IO, ORIENT_OO, ORIENT_HYPER]:
        #     self.orient = orient
        # else:
        #     self.orient = None
        #     if orient in self.xcol:
        #         self.xindexs = list(self.xcol).index(orient)
        #         self.yindexs = None
        #     elif orient in self.ycol:
        #         self.yindexs = list(self.ycol).index(orient)
        #         self.xindexs = None
        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引
        self.__modeldict = {}
        for i in self.I:
            # print(i)
            self.I0 = i                                                 ## I 是 被评价决策单元的数量

            self.__model__ = ConcreteModel()
            # Initialize sets
            self.__model__.I2 = Set(initialize= self.xref.index)                     ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))          ## L 是产出个数 被评价单元和参考单元的K，L一样


            # Initialize variable
            self.__model__.theta = Var(Set(initialize=range(1)),bounds=(0.0, None), within=Reals,doc='directional distance')
            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')

            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return model.theta[0]*1
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) - model.theta[0]*self.gx[k]*self.x.loc[self.I0,self.xcol[k]] <= self.x.loc[self.I0,self.xcol[k]]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return -sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) + model.theta[0]*self.gy[l] *self.y.loc[self.I0,self.ycol[l]]<= -self.y.loc[self.I0,self.ycol[l]]
        return output_rule

    def __vrs_rule(self):

        def vrs_rule(model):
            return sum(model.lamda[ i2] for i2 in model.I2) == 1

        return vrs_rule


    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,theta,lamda,= pd.DataFrame(),{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"]= tools.optimize_model2(problem, ind, solver)
            theta[ind],= np.asarray(list(problem.theta[:].value))
            # lamda[ind] =  np.asarray(list(problem.lamda[:].value))
        # print(theta)
        theta = pd.DataFrame(theta,index=["theta"]).T
        # lamda2 = pd.DataFrame(lamda).T
        # lamda2.columns = map(lambda x: "lamda"+str(x) ,lamda2.columns)
        data3 = pd.concat([data2,theta],axis=1)
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

