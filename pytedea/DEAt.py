"""Main module."""
# import dependencies

from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Reals,PositiveReals
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_HYPERYX,ORIENT_HYPERYB, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools
import ast

class DEAt:
    """traditional Data Envelopment Analysis (DEA)
    """
    def __init__(self, data,sent = "inputvar=outputvar",  orient=ORIENT_IO, rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L CO2= Y"
            orient(str): ORIENT_IO ORIENT_OO ORIENT_HYPER, or choose some variables in sent
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model

        self.rts = rts
        self.outputvars, self.inputvars ,self.y, self.x,self.yref, self.xref= tools.assert_valid_deat(sent,data,baseindex,refindex )

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        if orient in [ORIENT_IO, ORIENT_OO, ORIENT_HYPERYX]:
            self.orient = orient
            self.xindexs = None
            self.yindexs = None
        else:
            self.orient = None
            if orient in self.xcol:
                self.xindexs = list(self.xcol).index(orient)
                self.yindexs = None
            elif orient in self.ycol:
                self.yindexs = list(self.ycol).index(orient)
                self.xindexs = None

        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i
            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize=  self.xref.index)       ## I2 是 参考决策单元的数量

            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样

            # Initialize variable
            self.__model__.theta = Var(Set(initialize=range(1)),bounds=(None, None), doc='efficiency')
            if self.orient == ORIENT_HYPERYX:
                self.__model__.delta = Var(Set(initialize=range(1)),bounds=(None, None), doc='efficiency**2')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), doc='intensity variables')

            # Setup the objective function and constraints
            if self.orient == ORIENT_IO:
                self.__model__.objective = Objective(
                    rule=self.__objective_rule(), sense=minimize, doc='objective function')
            elif self.orient == ORIENT_OO:
                self.__model__.objective = Objective(
                    rule=self.__objective_rule(), sense=maximize, doc='objective function')
            else:
                if type(self.xindexs)!=type(None):
                    self.__model__.objective = Objective(
                        rule=self.__objective_rule(), sense=minimize, doc='objective function')
                else:
                    self.__model__.objective = Objective(
                        rule=self.__objective_rule(), sense=maximize, doc='objective function')

            self.__model__.input = Constraint(
                self.__model__.K, rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(
                self.__model__.L, rule=self.__output_rule(), doc='output constraint')


            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(
                     rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

            # self.__model__.objective.pprint()
            # self.__model__.input.pprint()
            # self.__model__.output.pprint()
            # self.__model__.vrs.pprint()


        # # Optimize model


    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return model.theta[0]*1  + sum(model.lamda[i2] *0 for i2 in model.I2)
        return objective_rule


    def __input_rule(self):
        """Return the proper input constraint"""
        if self.orient == ORIENT_OO:
            def input_rule(model, k):
                return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                    self.x.loc[self.I0,self.xcol[k]]
        elif self.orient == ORIENT_IO:
            def input_rule(model, k):
                return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                    model.theta * self.x.loc[self.I0,self.xcol[k]]

        else:
            if type(self.xindexs)!=type(None):
                def input_rule(model, k):
                    if k != self.xindexs:
                        return Constraint.Skip
                    return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                        model.theta * self.x.loc[self.I0,self.xcol[k]]
            else:
                def input_rule(model, k):
                    return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= self.x.loc[self.I0,self.xcol[k]]

        return input_rule



    def __output_rule(self):
        """Return the proper output constraint"""
        if self.orient == ORIENT_OO:
            def output_rule(model, l):
                return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                    >=model.theta * self.y.loc[self.I0,self.ycol[l]]
        elif self.orient == ORIENT_IO:
            def output_rule(model, l):
                return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                    >= self.y.loc[self.I0,self.ycol[l]]

        else:
            if type(self.yindexs)!=type(None):
                def output_rule(model, l):
                    if l != self.yindexs:
                        return Constraint.Skip
                    return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                        >= model.theta * self.y.loc[self.I0,self.ycol[l]]
            else:
                def output_rule(model, l):
                    return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                        >= self.y.loc[self.I0,self.ycol[l]]

        return output_rule

    def __vrs_rule(self):
        def vrs_rule(model):
            return sum(model.lamda[i2] for i2 in model.I2) == 1
        return vrs_rule

    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,lamda = pd.DataFrame(),{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model2(problem, ind, solver)
            data2.loc[ind,"theta"] = np.asarray(list(problem.theta[:].value))

        if self.orient==ORIENT_OO:
            data2["te"] = 1/  data2["theta"]
        else:
            data2["te"] = data2["theta"]
        return data2

    def info(self, dmu = "all"):
        """Show the infomation of the lp model

        Args:
            dmu (string): The solver chosen for optimization. Default is "all".
        """
        if dmu =="all":
            for ind, problem in self.__modeldict.items():
                print(ind,"\n",problem.pprint())

        print(self.__modeldict[int(dmu)].pprint())




class RM(DEAt):
    def __init__(self, data,year,sent = "inputvar=outputvar",  orient=ORIENT_IO, rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            orient(str): ORIENT_IO ORIENT_OO
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.orient = orient
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.rts = rts


        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars]
        else:
            self.y, self.x = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars]


        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns

        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i

            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize=  self.xref.index)       ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样

            # Initialize variable
            if self.orient == ORIENT_OO:
                self.__model__.theta = Var(self.__model__.L,bounds=(None, None), doc='efficiency')
            else:
                self.__model__.theta = Var(self.__model__.K,bounds=(None, None), doc='efficiency')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), doc='intensity variables')

            # Setup the objective function
            if self.orient == ORIENT_OO:
                self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')
            else:
                self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')

            # Setup the Constraint function
            self.__model__.input = Constraint(self.__model__.K, rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L, rule=self.__output_rule(), doc='output constraint')

            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # # Optimize model



    def __objective_rule(self):
        """Return the proper objective function"""
        if self.orient == ORIENT_OO:
            def objective_rule(model):
                return sum(model.theta[l]*1 for l in model.L) / len(model.L)
        else:
            def objective_rule(model):
                return sum(model.theta[k]*1 for k in model.K) / len(model.K)
        return objective_rule


    def __input_rule(self):
        """Return the proper input constraint"""
        if self.orient == ORIENT_OO:
            def input_rule(model, k):
                return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= self.x.loc[self.I0,self.xcol[k]]
        elif self.orient == ORIENT_IO:
            def input_rule(model, k):
                return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= model.theta[k] * self.x.loc[self.I0,self.xcol[k]]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        if self.orient == ORIENT_OO:
            def output_rule(model, l):
                return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) >=model.theta[l] * self.y.loc[self.I0,self.ycol[l]]
        elif self.orient == ORIENT_IO:
            def output_rule(model, l):
                return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) >= self.y.loc[self.I0,self.ycol[l]]
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

        data2,obj,theta,lamda = pd.DataFrame(),{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)
            obj[ind]= problem.objective()
            theta[ind] = np.asarray(list(problem.theta[:].value))
            # lamda[ind] = np.asarray(list(problem.lamda[:].value))
        theta = pd.DataFrame(theta).T
        if self.orient== ORIENT_OO:
            theta.columns = theta.columns.map(lambda x : "Output"+ str(x)+"'s theta" )
        else:
            theta.columns = theta.columns.map(lambda x : "Input"+ str(x)+"'s theta" )

        obj = pd.DataFrame(obj,index=["obj"]).T
        lamda =pd.DataFrame(lamda).T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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







class NDDF:

    def __init__(self, data,year,sent = "inputvar=outputvar",  gy=[1], gx=[1], gb=None, weight =None, rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            weight(list, optional): weght matrix
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.gy, self.gx, self.gb = gy,gx,gb
        self.rts = rts

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:
            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        if type(weight) != type(None):
            self.weight= weight


        else:
            self.weight=[]
            if type(self.b) != type(None):
                fenmu = 1*int(self.gx[0]!=0) + 1*int(self.gy[0]!=0) + 1*int(self.gb[0]!=0)
                print(fenmu)
                for _ in range(len(self.x.iloc[0])):
                    self.weight.append(1/fenmu/len(self.x.iloc[0]))
                for _ in range(len(self.y.iloc[0])):
                    self.weight.append(1/fenmu/len(self.y.iloc[0]))
                for _ in range(len(self.b.iloc[0])):
                    self.weight.append(1/fenmu/len(self.b.iloc[0]))
            else:
                fenmu = 1*int(self.gx[0]!=0) + 1*int(self.gy[0]!=0)

                for _ in range(len(self.x.iloc[0])):
                    self.weight.append(1/fenmu/len(self.x.iloc[0]))
                for _ in range(len(self.y.iloc[0])):
                    self.weight.append(1/fenmu/len(self.y.iloc[0]))

        self.iweight = self.weight[0:len(self.x.iloc[0])]
        self.oweight = self.weight[len(self.x.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])]
        if type(self.b) != type(None):
            self.bweight = self.weight[len(self.x.iloc[0])+len(self.y.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])+len(self.b.iloc[0])]


        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.b) != type(None) else None

        print(self.iweight,self.oweight,self.bweight)
        print(self.gx,self.gy,self.gb)

        self.I = self.x.index        ## I 是 被评价决策单元的索引
        self.__modeldict = {}
        for i in self.I:
            self.I0 = i                                                 ## I 是 被评价决策单元的数量

            self.__model__ = ConcreteModel()
            # Initialize sets
            self.__model__.I2 = Set(initialize=self.xref.index)      ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样
            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))   ## B 是 非期望产出个数

            # Initialize variable

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None),within=Reals, doc='scale factor x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), within=Reals,doc='scale factor y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), within=Reals,doc='scale factor B')
            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), within=Reals,doc='intensity variables')

            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')

            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), doc='undesirable output constraint')

            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        if type(self.b) != type(None):
            def objective_rule(model):
                return -sum( self.iweight[k]*self.gx[k]* model.thetax[k] for k in model.K
                    ) + sum(self.oweight[l]*self.gy[l]* model.thetay[l] for l in model.L
                    ) - sum(self.bweight[b]*self.gb[b]* model.thetab[b] for b in model.B)
        else:
            def objective_rule(model):
                return sum( self.iweight[k]*self.gx[k]* model.thetax[k] for k in model.K
                    ) + sum(self.oweight[l]*self.gy[l]* model.thetay[l] for l in model.L)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                ) - self.gx[k]*self.x.loc[self.I0,self.xcol[k]]*model.thetax[k]<=  self.x.loc[self.I0,self.xcol[k]]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return -1* sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                ) + self.gy[l]*self.y.loc[self.I0,self.ycol[l]]*model.thetay[l] <= -1*self.y.loc[self.I0,self.ycol[l]]

        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                ) - self.gb[b]*self.b.loc[self.I0,self.bcol[b]]*model.thetab[b] == self.b.loc[self.I0,self.bcol[b]]
        return undesirable_output_rule


    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind]= problem.objective()
                thetax[ind]=np.asarray(list(problem.thetax[:].value))
                thetay[ind]=np.asarray(list(problem.thetay[:].value))
                thetab[ind]=np.asarray(list(problem.thetab[:].value))
                lamda[ind]=np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind]= problem.objective()
                thetax[ind]=np.asarray(list(problem.thetax[:].value))
                thetay[ind]=np.asarray(list(problem.thetay[:].value))
                lamda[ind]=np.asarray(list(problem.lamda[:].value))


        obj = pd.DataFrame(obj,index=["obj"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )
        theta = pd.concat([thetax,thetay],axis=1)

        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta = pd.concat([theta,thetab],axis=1)
        dropcol=[]
        for colnum,g in enumerate(self.gx+self.gy+self.gb if type(self.b) != type(None) else self.gx+self.gy):
            # print(g)
            if g==0:
                dropcol.append(theta.columns[colnum])

        theta.drop(columns = dropcol,inplace=True)
        print("dropcol:",dropcol)
        lamda =pd.DataFrame(lamda).T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x))
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta],axis=1)

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




class WRM(NDDF):

    def __init__(self, data,year,sent = "K L=Y:CO2",  rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.rts = rts


        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:
            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

        # print(self.xcol)
        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i

            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize=  self.xref.index)                  ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样
            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))   ## B 是 非期望产出个数


            # Initialize variable

            self.__model__.thetax = Var(self.__model__.K,initialize=1,bounds=(0.0, None), doc='scale factor x')
            self.__model__.thetay = Var(self.__model__.L,initialize=1,bounds=(0.0, None), doc='scale factor y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,initialize=1,bounds=(0.0, None), doc='scale factor B')
            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), doc='intensity variables')

            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__NDDF__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')

            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), doc='undesirable output constraint')

            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __NDDF__objective_rule(self):
        """Return the proper objective function"""
        if type(self.b) != type(None):
            def NDDF__objective_rule(model):
                return sum( model.thetax[k] for k in model.K) /len(model.K
                    )+ sum( model.thetay[l] for l in model.L) /len(model.L
                    ) + sum(model.thetab[b] for b in model.B) /len(model.B)
        else:
            def NDDF__objective_rule(model):
                return sum( model.thetax[k] for k in model.K) /len(model.K
                    )+ sum( model.thetay[l] for l in model.L) /len(model.L)
        return NDDF__objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                ) <= (1-model.thetax[k] )*self.x.loc[self.I0,self.xcol[k]]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                ) >= (1+model.thetay[l]) *self.y.loc[self.I0,self.ycol[l]]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                ) == (1 - model.thetab[b])*self.b.loc[self.I0,self.bcol[b]]
        return undesirable_output_rule

    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)
            if type(self.b) != type(None):
                obj[ind]=problem.objective()
                thetax[ind]=np.asarray(list(problem.thetax[:].value))
                thetay[ind]=np.asarray(list(problem.thetay[:].value))
                thetab[ind]=np.asarray(list(problem.thetab[:].value))
                lamda[ind]=np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind]=problem.objective()
                thetax[ind]=np.asarray(list(problem.thetax[:].value))
                thetay[ind]=np.asarray(list(problem.thetay[:].value))
                lamda[ind]=np.asarray(list(problem.lamda[:].value))
                # print(problem.objective() )

        obj = pd.DataFrame(obj,index=["obj"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )
        theta = pd.concat([thetax,thetay],axis=1)

        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta = pd.concat([theta,thetab],axis=1)

        lamda =pd.DataFrame(lamda).T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)

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


class SBM(DEAt):

    def __init__(self, data,year,sent = "K L=Y:CO2",  rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.rts = rts


        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            # print(self.data)
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:
            # print(self.data)

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None



        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

        # print(self.xcol,self.ycol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}
        # print("111111111")

        for i in self.I:
            self.I0 = i

            self.__model__ = ConcreteModel()
            # print("222222222")
            self.__model__.I2 = Set(initialize=  self.xref.index)       ## I2 是 参考决策单元的数量
            # print("3333333")

            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            # print("4444")

            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样
            if type(self.b) != type(None):
                # print("55555")

                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))   ## B 是 非期望产出个数

            # Initialize variable
            self.__model__.t = Var(Set(initialize=range(1)),domain =PositiveReals,bounds=(0, None), doc='CC object')

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), doc='intensity variables')

            # Setup the objective function and constraints
            # print("66666")

            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            # self.__model__.cctrans1 = Constraint( expr= self.__model__.t[0] >0, doc='cctrans1')
            # print("7777777")
            self.__model__.cctrans = Constraint(rule=self.__cctrans_rule(), doc='cctrans')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            # print("88888888")

            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')

            if type(self.b) != type(None):
                # print("99999999")

                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), doc='undesirable output constraint')

            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return model.t[0]-sum(model.thetax[k]/self.x.loc[self.I0,self.xcol[k]] for k in model.K) / len(model.K)
        return objective_rule

    def __cctrans_rule(self):
        """Return the cctrans  constraint"""
        if type(self.b) != type(None):
            def cctrans_rule(model):
                return 1==model.t[0]+ (sum(model.thetay[l]/self.y.loc[self.I0,self.ycol[l]] for l in model.L
                    ) +sum(model.thetay[b]/self.b.loc[self.I0,self.bcol[b]] for b in model.B))/(len(model.L)+len(model.B))
        else:
            def cctrans_rule(model):
                return 1==model.t[0]+ (sum(model.thetay[l]/self.y.loc[self.I0,self.ycol[l]] for l in model.L) )/(len(model.L))
        return cctrans_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            # print(self.x[self.I0][0]-model.thetax[0])
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                ) == self.x.loc[self.I0,self.xcol[k]]*model.t[0]-model.thetax[k]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                ) == self.y.loc[self.I0,self.ycol[l]] * model.t[0]+model.thetay[l]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                ) == self.b.loc[self.I0,self.bcol[b]] * model.t[0]-model.thetab[b]
        return undesirable_output_rule

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

        data2,obj,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()/np.asarray(list(problem.t[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))/np.asarray(list(problem.t[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))/np.asarray(list(problem.t[:].value))
                thetab[ind]= np.asarray(list(problem.thetab[:].value))/np.asarray(list(problem.t[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))/np.asarray(list(problem.t[:].value))
            else:
                obj[ind] = problem.objective()/np.asarray(list(problem.t[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))/np.asarray(list(problem.t[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))/np.asarray(list(problem.t[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))/np.asarray(list(problem.t[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj,index=["obj"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )
        theta=pd.concat([thetax,thetay],axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta = pd.concat([theta,thetab],axis=1)

        lamda =pd.DataFrame(lamda) .T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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


class MB():
    def __init__(self, data,year,sent = "inputvar_p + inputvar_np =outputvar_p + outputvar_np:unoutputvar",  \
                 sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], level=5 ,rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L+ E = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            level(int, optional): 返回求的层级。1：只变量化b；2：再变量化x_p；3：再变量化x_np；\
                                                4：再变量化y_p（没有则变量化y_np）；5：再变量化y_np
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.rts = rts
        self.baseindex = baseindex
        self.refindex = refindex

        self.inputvars_np, self.inputvars_p,self.outputvars_np,\
            self.outputvars_p,self.unoutputvars, self.sx, self.sy,self.level = tools.split_MB(self.sent, sx, sy,level)

    def optimize(self, solver=OPT_DEFAULT,dmu="1"):

        if type(self.outputvars_np) != type(None):
            if type(self.outputvars_p) != type(None):

                model1111 = MB_.MB1111(  ## K L + E = Y + Y2 : CO2
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_np,self.outputvars_p,\
                               self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                data3 = model1111.optimize(solver=solver)
                info = model1111.info(dmu=dmu)
            elif type(self.outputvars_p) == type(None):
                model1110 = MB_.MB1110(  ## K L + E = Y  : CO2
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_np,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                data3 = model1110.optimize(solver=solver)
                info = model1110.info(dmu=dmu)
        elif type(self.outputvars_np) == type(None):
            if type(self.outputvars_p) != type(None):
                model1101 = MB_.MB1101(  ## K L + E = + Y2 : CO2
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_p,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                data3 = model1101.optimize(solver=solver)
                info = model1101.info(dmu=dmu)
            elif type(self.outputvars_p) == type(None):
                model1100 = MB_.MB1100(  ## K L + E =   : CO2
                               self.data, self.inputvars_np, self.inputvars_p,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                data3 = model1100.optimize(solver=solver)
                info = model1100.info(dmu=dmu)
        return data3,info




    def info(self, dmu="all"):
        if type(self.outputvars_np) != type(None):
            if type(self.outputvars_p) != type(None):
                model1111 = MB_.MB1111(
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_np,\
                               self.outputvars_p,self.unoutputvars, self.sx, self.sy, \
                                self.rts,self.level,self.baseindex,self.refindex)
                return model1111.info(dmu=dmu)
            elif type(self.outputvars_p) == type(None):
                model1110 = MB_.MB1110(  ## K L + E = Y  : CO2
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_np,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                return model1110.info(dmu=dmu)
        elif type(self.outputvars_np) == type(None):
            if type(self.outputvars_p) != type(None):
                model1101 = MB_.MB1101(  ## K L + E = + Y2 : CO2
                               self.data, self.inputvars_np, self.inputvars_p,self.outputvars_p,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                return model1101.info(dmu=dmu)
            elif type(self.outputvars_p) == type(None):
                model1100 = MB_.MB1100(  ## K L + E =   : CO2
                               self.data, self.inputvars_np, self.inputvars_p,\
                                self.unoutputvars, self.sx, self.sy,self.rts,self.level,self.baseindex,self.refindex)
                return model1100.info(dmu=dmu)



class MBx(MB):
    def __init__(self, data,year,sent = "inputvar=outputvar",  sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.sx, self.sy = sx, sy
        self.rts = rts
        print(self.sx, self.sy)

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(self.b)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

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

            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))      ## B 是 非期望产出个数

            # Initialize variable

            self.__model__.objx = Var(self.__model__.K,bounds=(0.0, None), within=Reals,doc='object x')

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')
                self.__model__.theta = Var(self.__model__.B,bounds=(0.0, None), within=Reals,doc='object b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')


            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), \
                                                               doc='undesirable output constraint')
                self.__model__.mb = Constraint(self.__model__.B,  rule=self.__mb_rule(), \
                                                                doc='material balance constraint')
            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.theta[b]*1 for b in model.B)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) + model.thetax[k] == model.objx[k]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) - model.thetay[l] == self.y.loc[self.I0,self.ycol[l]]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                    ) + model.thetab[b] == model.theta[b]*1
        return undesirable_output_rule

    def __mb_rule(self):
        """Return the proper undesirable output constraint"""
        def mb_rule(model, b):
            return sum(self.sx[b][k] * (model.thetax[k] + self.x.loc[self.I0,self.xcol[k]] - model.objx[k] ) for k in model.K) \
                    + sum(self.sy[b][l] * model.thetay[l] for l in model.L) \
                    == self.b.loc[self.I0,self.bcol[b]] - model.theta[b]
        return mb_rule

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

        data2,obj,theta,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                thetab[ind]= np.asarray(list(problem.thetab[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj,index=["obj"]).T
        theta = pd.DataFrame(theta,index=["var Undesirable"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )

        theta_=pd.concat([theta,thetax],axis=1)
        theta_=pd.concat([theta_,thetay],axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta_ = pd.concat([theta_,thetab],axis=1)

        lamda =pd.DataFrame(lamda) .T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta_],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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


class MBx2(MB):
    def __init__(self, data,year,sent = "inputvar=outputvar",  sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.sx, self.sy = sx, sy
        self.rts = rts
        print(self.sx, self.sy)

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(self.b)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

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

            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))      ## B 是 非期望产出个数

            # Initialize variable

            self.__model__.objx = Var(self.__model__.K,bounds=(0.0, None), within=Reals,doc='object x')

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')
                self.__model__.theta = Var(self.__model__.B,bounds=(0.0, None), within=Reals,doc='object b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')


            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), \
                                                               doc='undesirable output constraint')
                self.__model__.mb = Constraint(self.__model__.B,  rule=self.__mb_rule(), \
                                                                doc='material balance constraint')
            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.theta[b]*1 for b in model.B)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) + model.thetax[k] == model.objx[k]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) - model.thetay[l] == self.y.loc[self.I0,self.ycol[l]]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                    ) + model.thetab[b] == model.theta[b]*1
        return undesirable_output_rule

    def __mb_rule(self):
        """Return the proper undesirable output constraint"""
        def mb_rule(model, b):
            return sum(self.sx[b][k] * (model.thetax[k] + self.x.loc[self.I0,self.xcol[k]] - model.objx[k] ) for k in model.K) \
                    + sum(self.sy[b][l] * model.thetay[l] for l in model.L) \
                    == self.b.loc[self.I0,self.bcol[b]] - model.theta[b]
        return mb_rule

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

        data2,obj,theta,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                thetab[ind]= np.asarray(list(problem.thetab[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj,index=["obj"]).T
        theta = pd.DataFrame(theta,index=["var Undesirable"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )

        theta_=pd.concat([theta,thetax],axis=1)
        theta_=pd.concat([theta_,thetay],axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta_ = pd.concat([theta_,thetab],axis=1)

        lamda =pd.DataFrame(lamda) .T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta_],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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


class MBxy(MB):
    def __init__(self, data,year,sent = "inputvar=outputvar",  sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.sx, self.sy = sx, sy
        self.rts = rts
        print(self.sx, self.sy)

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(self.b)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

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

            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))      ## B 是 非期望产出个数

            # Initialize variable

            self.__model__.objx = Var(self.__model__.K,bounds=(0.0, None), within=Reals,doc='object x')
            self.__model__.objy = Var(self.__model__.L,bounds=(0.0, None), within=Reals,doc='object y')

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')
                self.__model__.theta = Var(self.__model__.B,bounds=(0.0, None), within=Reals,doc='object b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')


            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), \
                                                               doc='undesirable output constraint')
                self.__model__.mb = Constraint(self.__model__.B,  rule=self.__mb_rule(), \
                                                                doc='material balance constraint')
            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.theta[b]*1 for b in model.B)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) + model.thetax[k] == model.objx[k]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) - model.thetay[l] == model.objy[l]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                    ) + model.thetab[b] == model.theta[b]*1
        return undesirable_output_rule

    def __mb_rule(self):
        """Return the proper undesirable output constraint"""
        def mb_rule(model, b):
            return sum(self.sx[b][k] * (model.thetax[k] + self.x.loc[self.I0,self.xcol[k]] - model.objx[k]) for k in model.K) \
                 + sum(self.sy[b][l] * (model.thetay[l] + model.objy[l] - self.y.loc[self.I0,self.ycol[l]]) for l in model.L) \
                == model.thetab[b]+self.b.loc[self.I0,self.bcol[b]] - model.theta[b]
        return mb_rule


    def __vrs_rule(self):
        def vrs_rule(model):
            return sum(model.lamda[i2] for i2 in model.I2) == 1

        return vrs_rule

    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,theta,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                thetab[ind]= np.asarray(list(problem.thetab[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind] = problem.objective()
                theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj,index=["obj"]).T
        theta = pd.DataFrame(theta,index=["var Undesirable"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )

        theta_=pd.concat([theta,thetax],axis=1)
        theta_=pd.concat([theta_,thetay],axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta_ = pd.concat([theta_,thetab],axis=1)

        lamda =pd.DataFrame(lamda) .T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta_],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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



class MBxy2(MB):
    def __init__(self, data,year,sent = "inputvar=outputvar",  sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.sx, self.sy = sx, sy
        self.rts = rts
        print(self.sx, self.sy)

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(self.b)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

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

            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))      ## B 是 非期望产出个数

            # Initialize variable

            self.__model__.objx = Var(self.__model__.K,bounds=(0.0, None), within=Reals,doc='object x')
            self.__model__.objy = Var(self.__model__.L,bounds=(0.0, None), within=Reals,doc='object y')

            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')


            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), \
                                                               doc='undesirable output constraint')
                self.__model__.mb = Constraint(self.__model__.B,  rule=self.__mb_rule(), \
                                                                doc='material balance constraint')
            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.thetab[b] for b in model.B)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) + model.thetax[k] == model.objx[k]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) - model.thetay[l] == model.objy[l]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                    ) + model.thetab[b] == self.b.loc[self.I0,self.bcol[b]]
        return undesirable_output_rule

    def __mb_rule(self):
        """Return the proper undesirable output constraint"""
        def mb_rule(model, b):
            return sum(self.sx[b][k] * (model.thetax[k] + self.x.loc[self.I0,self.xcol[k]] - model.objx[k]) for k in model.K) \
                 + sum(self.sy[b][l] * (model.thetay[l] + model.objy[l] - self.y.loc[self.I0,self.ycol[l]]) for l in model.L) \
                == model.thetab[b]
        return mb_rule


    def __vrs_rule(self):
        def vrs_rule(model):
            return sum(model.lamda[i2] for i2 in model.I2) == 1

        return vrs_rule

    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,thetax,thetay,thetab,lamda = pd.DataFrame(),{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()
                # theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                thetab[ind]= np.asarray(list(problem.thetab[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind] = problem.objective()
                # theta[ind], = np.asarray(list(problem.theta[:].value))
                thetax[ind]= np.asarray(list(problem.thetax[:].value))
                thetay[ind]= np.asarray(list(problem.thetay[:].value))
                lamda[ind]= np.asarray(list(problem.lamda[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj,index=["obj"]).T
        # theta = pd.DataFrame(theta,index=["var Undesirable"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x : "Input"+ str(x)+"'s slack" )
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y : "Output"+ str(y)+"'s slack" )

        # theta_=pd.concat([theta,thetax],axis=1)
        theta_=pd.concat([thetax,thetay],axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s slack" )
            theta_ = pd.concat([theta_,thetab],axis=1)

        lamda =pd.DataFrame(lamda) .T
        lamda.columns = lamda.columns.map(lambda x : "lamda"+ str(x) )
        data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([data3,theta_],axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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

class MB2(MB):
    def __init__(self, data,year,sent = "inputvar=outputvar",  sx=[[1,1,1],[1,1,1]], sy=[[1],[1]], rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            sx (list): 投入包含污染物质系数. Defaults to [[1,1,1],[1,1,1]].
            sy (list, optional): 期望产出包含污染物质系数. Defaults to [[1],[1]].
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.sx, self.sy = sx, sy
        self.rts = rts
        print(self.sx, self.sy)

        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(self.b)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

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

            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))      ## B 是 非期望产出个数

            # Initialize variable
            self.__model__.thetax = Var(self.__model__.K,bounds=(0.0, None), doc='slack x')
            self.__model__.thetay = Var(self.__model__.L,bounds=(0.0, None), doc='slack y')
            if type(self.b) != type(None):
                self.__model__.thetab = Var(self.__model__.B,bounds=(0.0, None), doc='slack b')

            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None),within=Reals, doc='intensity variables')


            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')
            self.__model__.input = Constraint(self.__model__.K,  rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(self.__model__.L,  rule=self.__output_rule(), doc='output constraint')


            if type(self.b) != type(None):
                self.__model__.undesirable_output = Constraint(self.__model__.B, rule=self.__undesirable_output_rule(), \
                                                               doc='undesirable output constraint')
                self.__model__.mb = Constraint(self.__model__.B,  rule=self.__mb_rule(), \
                                                                doc='material balance constraint')
            if self.rts == RTS_VRS:
                self.__model__.vrs = Constraint(rule=self.__vrs_rule(), doc='various return to scale rule')

            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return sum(model.thetab[b]*1 for b in model.B)
        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""
        def input_rule(model, k):
            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2
                    ) + model.thetax[k] == self.x.loc[self.I0,self.xcol[k]]
        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""
        def output_rule(model, l):
            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2
                    ) - model.thetay[l] == self.y.loc[self.I0,self.ycol[l]]
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        def undesirable_output_rule(model, b):
            return sum(model.lamda[i2] * self.bref.loc[i2,self.bcol[b]] for i2 in model.I2
                    ) + model.thetab[b] == self.b.loc[self.I0,self.bcol[b]]
        return undesirable_output_rule

    def __mb_rule(self):
        """Return the proper undesirable output constraint"""
        def mb_rule(model, b):
            return sum(self.sx[b][k] * model.thetax[k] for k in model.K) \
                    + sum(self.sy[b][l] * model.thetay[l] for l in model.L) \
                    == model.thetab[b]
        return mb_rule

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

        data2, obj, thetax, thetay, thetab, lamda = pd.DataFrame(), {}, {}, {}, {}, {}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind, "optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind] = problem.objective()
                thetax[ind] = np.asarray(list(problem.thetax[:].value))
                thetay[ind] = np.asarray(list(problem.thetay[:].value))
                thetab[ind] = np.asarray(list(problem.thetab[:].value))
                lamda[ind] = np.asarray(list(problem.lamda[:].value))
            else:
                obj[ind] = problem.objective()
                thetax[ind] = np.asarray(list(problem.thetax[:].value))
                thetay[ind] = np.asarray(list(problem.thetay[:].value))
                lamda[ind] = np.asarray(list(problem.lamda[:].value))

                # print(list(problem.thetax[:].value ),list(problem.t[:].value ))
        obj = pd.DataFrame(obj, index=["obj"]).T
        thetax = pd.DataFrame(thetax).T
        thetax.columns = thetax.columns.map(lambda x: "Input" + str(x) + "'s slack")
        thetay = pd.DataFrame(thetay).T
        thetay.columns = thetay.columns.map(lambda y: "Output" + str(y) + "'s slack")

        theta_ = pd.concat([thetax, thetay], axis=1)
        if type(self.b) != type(None):
            thetab = pd.DataFrame(thetab).T
            thetab.columns = thetab.columns.map(lambda b: "Undesirable Output" + str(b) + "'s slack")
            theta_ = pd.concat([theta_, thetab], axis=1)

        lamda = pd.DataFrame(lamda).T
        lamda.columns = lamda.columns.map(lambda x: "lamda" + str(x))
        data3 = pd.concat([data2, obj], axis=1)
        data3 = pd.concat([data3, theta_], axis=1)
        # data3 = pd.concat([data3,lamda],axis=1)
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





class DEADUAL:
    """Dual of Data Envelopment Analysis (DEA)
    """

    def __init__(self, data,year,sent = "inputvar=outputvar",  orient=ORIENT_IO, rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            orient(str): ORIENT_IO ORIENT_OO
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.orient = orient
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
        self.rts = rts


        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars]
        else:
            self.y, self.x = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars]


        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars]
        else:
            self.yref, self.xref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars]

        self.xcol = self.x.columns
        self.ycol = self.y.columns

        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i

            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize= self.xref.index)       ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))          ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样

            # Initialize variable
            self.__model__.px = Var(self.__model__.K, bounds=(0.0, None), doc='shadow price of x')
            self.__model__.py = Var(self.__model__.L, bounds=(0.0, None), doc='shadow price of y')
            if self.rts == RTS_VRS:
                self.__model__.pomega = Var(Set(initialize=range(1)), bounds=(None, None), within=Reals,doc='shadow price of 1')

            # Setup the objective function and constraints
            # print(self.orient)
            if self.orient == ORIENT_OO:
                self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            elif self.orient == ORIENT_IO:
                self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')

            self.__model__.first = Constraint(self.__model__.I2, rule=self.__first_rule(), doc='first constraint')
            self.__model__.second = Constraint( rule=self.__second_rule(), doc='second constraint')


            self.__modeldict[i] = self.__model__

        # # Optimize model


    def __objective_rule(self):
        """Return the proper objective function"""
        if self.orient == ORIENT_OO:
            def objective_rule(model):
                return sum(model.px[k] *self.x.loc[self.I0,self.xcol[k]] for k in model.K
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0)

        elif self.orient == ORIENT_IO:
            def objective_rule(model):
                return sum(model.py[l] *self.y.loc[self.I0,self.ycol[l]] for l in model.L
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0)
        return objective_rule


    def __first_rule(self):
        """Return the proper input constraint"""
        if self.orient == ORIENT_OO:
            def first_rule(model, i2):
                return sum(model.px[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                    ) - sum(model.py[l] * self.yref.loc[i2,self.ycol[l]] for l in model.L
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0) >= 0
        elif self.orient == ORIENT_IO:
            def first_rule(model, i2):
                return sum(model.py[l] * self.yref.loc[i2,self.ycol[l]] for l in model.L
                    ) - sum(model.px[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0) <= 0
        return first_rule



    def __second_rule(self):
        """Return the proper output constraint"""
        if self.orient == ORIENT_OO:
            def second_rule(model):
                return sum(model.py[l] * self.y.loc[self.I0,self.ycol[l]] for l in model.L)==1
        elif self.orient == ORIENT_IO:
            def second_rule(model):
                return sum(model.px[k] * self.x.loc[self.I0,self.xcol[k]] for k in model.K)==1
        return second_rule



    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,px,py,pomega = pd.DataFrame(),{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if self.rts == RTS_VRS:
                obj[ind]=problem.objective()
                px[ind]=np.asarray(list(problem.px[:].value))
                py[ind]=np.asarray(list(problem.py[:].value))
                pomega[ind]=np.asarray(list(problem.pomega[:].value))
            else:
                obj[ind]=problem.objective()
                px[ind]=np.asarray(list(problem.px[:].value))
                py[ind]=np.asarray(list(problem.py[:].value))
        px = pd.DataFrame(px).T
        px.columns = px.columns.map(lambda x : "Input"+ str(x)+"'s shadow price" )
        py = pd.DataFrame(py).T
        py.columns = py.columns.map(lambda y : "Output"+ str(y)+"'s shadow price" )
        p=pd.concat([px,py],axis=1)
        data3 = pd.concat([data2,p],axis=1)

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




class SBMDUAL:
    """Dual of Data Envelopment Analysis (DEA)
    """

    def __init__(self, data,year,sent = "K L=Y:CO2",  baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None




        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.unoutputvars) != type(None) else None

        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i

            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize=  self.xref.index)       ## I2 是 参考决策单元的数量
            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))           ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样
            if type(self.b) != type(None):
                self.__model__.B = Set(initialize=range(len(self.b.iloc[0])))           ## B 是坏产出个数 被评价单元和参考单元的K，L，B一样

            # Initialize variable
            self.__model__.px = Var(self.__model__.K, bounds=(None, None), within=Reals, doc='shadow price of x')
            self.__model__.py = Var(self.__model__.L, bounds=(None, None), within=Reals, doc='shadow price of y')
            if type(self.b) != type(None):
                self.__model__.pb = Var(self.__model__.B, bounds=(None, None), within=Reals,doc='shadow price of 1')
            self.__model__.pomega = Var(Set(initialize=range(1)), bounds=(None, None), within=Reals,doc='shadow price of 1')

            # Setup the objective function and constraints

            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=maximize, doc='objective function')

            self.__model__.first = Constraint(self.__model__.I2, rule=self.__first_rule(), doc='first constraint')
            self.__model__.second = Constraint(self.__model__.K, rule=self.__second_rule(), doc='second constraint')
            self.__model__.third = Constraint(self.__model__.L, rule=self.__third_rule(), doc='third constraint')
            if type(self.b) != type(None):
                self.__model__.forth = Constraint(self.__model__.B, rule=self.__forth_rule(), doc='forth constraint')
            self.__model__.fifth = Constraint( rule=self.__fifth_rule(), doc='fifth constraint')

            self.__modeldict[i] = self.__model__

        # # Optimize model


    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            return model.pomega[0]*1
        return objective_rule

    def __first_rule(self):
        """Return the proper input constraint"""

        def first_rule(model, i2):
            return -sum(model.px[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                ) + sum(model.py[l] * self.yref.loc[i2,self.ycol[l]] for l in model.L
                ) - (sum(model.pb[b] * self.bref.loc[i2,self.bcol[b]] for b in model.B) if type(self.b) != type(None) else 0)   <= 0
        return first_rule

    def __second_rule(self):
        """Return the proper second constraint"""
        def second_rule(model,k):
            return -model.px[k] <= -1 / (len(model.K)*self.x.loc[self.I0,self.xcol[k]] )
        return second_rule

    def __third_rule(self):
        """Return the proper third constraint"""
        def third_rule(model,l):
            return model.pomega[0] / (len(model.L)+len(model.B) if type(self.b) != type(None) else len(model.L)
                    ) /self.y.loc[self.I0,self.ycol[l]]-  model.py[l]<=0
        return third_rule

    def __forth_rule(self):
        """Return the proper forth constraint"""
        def forth_rule(model,b):
            return model.pomega[0] / (len(model.L)+len(model.B)) /self.b.loc[self.I0,self.bcol[b]] - model.pb[b]<=0
        return forth_rule

    def __fifth_rule(self):
        """Return the proper fifth constraint"""
        def fifth_rule(model):
            return model.pomega[0] + sum(model.px[k] * self.x.loc[self.I0,self.xcol[k]] for k in model.K
                ) - sum(model.py[l] * self.y.loc[self.I0,self.ycol[l]] for l in model.L
                ) + (sum(model.pb[b] * self.b.loc[self.I0,self.bcol[b]] for b in model.B) if type(self.b) != type(None) else 0 )<=1
        return fifth_rule


    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,px,py,pb,pomega = pd.DataFrame(),{},{},{},{},{}
        for ind, problem in self.__modeldict.items():
            _, data2.loc[ind,"optimization_status"] = tools.optimize_model(problem, ind, solver)

            if type(self.b) != type(None):
                obj[ind]=problem.objective()
                px[ind]=np.asarray(list(problem.px[:].value))
                py[ind]=np.asarray(list(problem.py[:].value))
                pb[ind]=np.asarray(list(problem.pb[:].value))
                pomega[ind]= np.asarray(list(problem.pomega[:].value))
            else:
                obj[ind]=problem.objective()
                px[ind]=np.asarray(list(problem.px[:].value))
                py[ind]=np.asarray(list(problem.py[:].value))
                pomega[ind]= np.asarray(list(problem.pomega[:].value))

        obj = pd.DataFrame(obj,index=["obj"]).T
        px = pd.DataFrame(px).T
        px.columns = px.columns.map(lambda x : "Input"+ str(x)+"'s shadow price" )
        py = pd.DataFrame(py).T
        py.columns = py.columns.map(lambda y : "Output"+ str(y)+"'s shadow price" )
        p=pd.concat([px,py],axis=1)
        if type(self.b) != type(None):
            pb = pd.DataFrame(pb).T
            pb.columns = pb.columns.map(lambda b : "Output"+ str(b)+"'s shadow price" )
            p=pd.concat([p,pb],axis=1)
        data3=  pd.concat([data2,obj],axis=1)
        data3=  pd.concat([data3,p],axis=1)


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




class NDDFDUAL(NDDF):

    def __init__(self, data,year,sent = "inputvar=outputvar",  gy=[1], gx=[1], gb=None, weight =None, rts=RTS_VRS, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L = Y : CO2"
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            weight(list, optional): weght matrix
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model
        self.data=data
        self.year = year
        self.sent = sent
        self.tlt=pd.Series(self.year).drop_duplicates().sort_values()
        self.inputvars = self.sent.split('=')[0].strip(' ').split(' ')
        try:
            self.outputvars = self.sent.split('=')[1]   .split(':')[0].strip(' ').split(' ')
            self.unoutputvars = self.sent.split('=')[1]   .split(':')[1].strip(' ').split(' ')
        except:
            self.outputvars = self.sent.split('=')[1]    .strip(' ').split(' ')
            self.unoutputvars=None
        self.gy, self.gx, self.gb = gy,gx,gb
        self.rts = rts


        self.baseindex = baseindex
        if type(baseindex) != type(None):
            self.varname1=self.baseindex.split('=')[0]
            print(self.baseindex)
            self.varvalue1=ast.literal_eval(self.baseindex.split('=')[1])
            self.y, self.x, self.b = self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.outputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.inputvars
                                        ], self.data.loc[self.data[self.varname1].isin(self.varvalue1), self.unoutputvars
                                        ]if type(self.unoutputvars) != type(None) else None

        else:

            self.y, self.x, self.b = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None


        # print(type(self.varname1),self.varvalue1,self.x,)
        self.refindex = refindex
        if type(refindex) != type(None):
            self.varname=self.refindex.split('=')[0]
            self.varvalue=ast.literal_eval(self.refindex.split('=')[1])

            self.yref, self.xref, self.bref = self.data.loc[self.data[self.varname].isin(self.varvalue), self.outputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.inputvars
                                                ], self.data.loc[self.data[self.varname].isin(self.varvalue), self.unoutputvars
                                                ] if type(self.unoutputvars) != type(None) else None
        else:
            self.yref, self.xref, self.bref = self.data.loc[:, self.outputvars
                                        ], self.data.loc[:, self.inputvars
                                        ], self.data.loc[:, self.unoutputvars
                                        ] if type(self.unoutputvars) != type(None) else None

        if type(weight) != type(None):
            self.weight= weight


        else:
            self.weight=[]
            if type(self.b) != type(None):
                fenmu = 1*int(self.gx[0]!=0) + 1*int(self.gy[0]!=0) + 1*int(self.gb[0]!=0)
                print(fenmu)
                for _ in range(len(self.x.iloc[0])):
                    self.weight.append(1/fenmu/len(self.x.iloc[0]))
                for _ in range(len(self.y.iloc[0])):
                    self.weight.append(1/fenmu/len(self.y.iloc[0]))
                for _ in range(len(self.b.iloc[0])):
                    self.weight.append(1/fenmu/len(self.b.iloc[0]))
            else:
                fenmu = 1*int(self.gx[0]!=0) + 1*int(self.gy[0]!=0)

                for _ in range(len(self.x.iloc[0])):
                    self.weight.append(1/fenmu/len(self.x.iloc[0]))
                for _ in range(len(self.y.iloc[0])):
                    self.weight.append(1/fenmu/len(self.y.iloc[0]))

        self.iweight = self.weight[0:len(self.x.iloc[0])]
        self.oweight = self.weight[len(self.x.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])]
        if type(self.b) != type(None):
            self.bweight = self.weight[len(self.x.iloc[0])+len(self.y.iloc[0]):len(self.x.iloc[0])+len(self.y.iloc[0])+len(self.b.iloc[0])]


        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns if type(self.b) != type(None) else None

        print(self.iweight,self.oweight,self.bweight)
        print(self.gx,self.gy,self.gb)

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
            if type(self.b) != type(None):
                self.__model__.J = Set(initialize=range(len(self.b.iloc[0])))   ## B 是 非期望产出个数


            # Initialize variable

            self.__model__.px = Var(self.__model__.K,initialize=1,bounds=(0.0, None), within=Reals,doc='shadow price of x')
            self.__model__.py = Var(self.__model__.L, initialize=1,bounds=(0.0, None),within=Reals, doc='shadow price of y')
            if type(self.b) != type(None):
                self.__model__.pb = Var(self.__model__.J,initialize=1,within=Reals, doc='shadow price of b')
            if self.rts == RTS_VRS:
                self.__model__.pomega = Var(Set(initialize=range(1)),  within=Reals,doc='shadow price of 1')

            # Setup the objective function and constraints
            self.__model__.objective = Objective(rule=self.__objective_rule(), sense=minimize, doc='objective function')
            self.__model__.first = Constraint(self.__model__.I2,  rule=self.__first_rule(), doc='first constraint')
            self.__model__.second = Constraint(self.__model__.K,  rule=self.__second_rule(), doc='second constraint')
            self.__model__.third = Constraint(self.__model__.L,  rule=self.__third_rule(), doc='third constraint')

            if type(self.b) != type(None):
                self.__model__.forth = Constraint(self.__model__.J,  rule=self.__forth_rule(), doc='forth constraint')


            self.__modeldict[i] = self.__model__

        # Optimize model
    def __objective_rule(self):
        """Return the proper objective function"""
        def objective_rule(model):
            if type(self.b) != type(None):
                return sum(model.px[k]*self.x.loc[self.I0,self.xcol[k]] for k in model.K
                    ) - sum(model.py[l]*self.y.loc[self.I0,self.ycol[l]] for l in model.L
                    ) + sum(model.pb[j]*self.b.loc[self.I0,self.bcol[j]] for j in model.J
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0)
            else:
                return sum(model.px[k]*self.x.loc[self.I0,self.xcol[k]] for k in model.K
                    ) - sum(model.py[l]*self.y.loc[self.I0,self.ycol[l]] for l in model.L
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0 )
        return objective_rule

    def __first_rule(self):
        """Return the proper first constraint"""
        def first_rule(model, i2):
            if type(self.b) != type(None):
                return sum(model.px[k] * self.xref.loc[i2,self.xcol[k]] for k in model.K
                    ) - sum(model.py[l] * self.yref.loc[i2,self.ycol[l]] for l in model.L
                    ) + sum(model.pb[j] * self.bref.loc[i2,self.bcol[j]] for j in model.J
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0)   >=0
            else:
                return sum(model.px[k] *self.xref.loc[i2,self.xcol[k]]   for k in model.K
                    ) - sum(model.py[l] *self.yref.loc[i2,self.ycol[l]]  for l in model.L
                    ) + (model.pomega[0]*1 if self.rts == RTS_VRS else 0 )  >=0
        return first_rule

    def __second_rule(self):
        """Return the proper second constraint"""
        def second_rule(model, k):
            if self.gx[k]==0:
                return Constraint.Skip
            return  -self.gx[k]*self.x.loc[self.I0,self.xcol[k]] * model.px[k] >= self.iweight[k]
        return second_rule

    def __third_rule(self):
        """Return the proper third constraint"""
        def third_rule(model, l):
            if self.gy[l]==0:
                return Constraint.Skip
            return  self.gy[l]*self.y.loc[self.I0,self.ycol[l]]*model.py[l] >= self.oweight[l]

        return third_rule

    def __forth_rule(self):
        """Return the proper forth constraint"""
        def forth_rule(model, j):
            if self.gb[j]==0:
                return Constraint.Skip
            return  -self.gb[j]*self.b.loc[self.I0,self.bcol[j]]*model.pb[j] >= self.bweight[j]
        return forth_rule

    def optimize(self,  solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization

        data2,obj,px,py,pb,  = pd.DataFrame,{},{},{},{},
        for ind, problem in self.__modeldict.items():
            _, _ = tools.optimize_model(problem, ind, solver)
            if type(self.b) != type(None):
                obj[ind]= problem.objective()
                px[ind]= np.asarray(list(problem.px[:].value))
                py[ind]= np.asarray(list(problem.py[:].value))
                pb[ind]= np.asarray(list(problem.pb[:].value))
                # pomega[ind]=
            else:
                obj[ind]= problem.objective()
                px[ind]= np.asarray(list(problem.px[:].value))
                py[ind]= np.asarray(list(problem.py[:].value))
                # pomega[ind]=
        obj = pd.DataFrame(obj,index=["obj"]).T
        px = pd.DataFrame(px).T
        px.columns = px.columns.map(lambda x : "Input"+ str(x)+"'s shadow price" )
        py = pd.DataFrame(py).T
        py.columns = py.columns.map(lambda y : "Output"+ str(y)+"'s shadow price" )
        pb = pd.DataFrame(pb).T
        pb.columns = pb.columns.map(lambda b : "Undesirable Output"+ str(b)+"'s shadow price" )
        p=pd.concat([px,py],axis=1)
        p=pd.concat([p,pb],axis=1)
        # data3 = pd.concat([data2,obj],axis=1)
        data3 = pd.concat([obj,p],axis=1)
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

