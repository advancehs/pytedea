"""Main module."""
# import dependencies

from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint, Reals,PositiveReals
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_UO, ORIENT_HYPERYB,ORIENT_HYPERYX, \
    RTS_VRS, RTS_CRS, EMF_SAME,EMF_DIFFERENT,OPT_DEFAULT, OPT_LOCAL
from .utils import tools
import ast

class DEA:
    """Data Envelopment Analysis (DEA)
    """
    def __init__(self, data,sent = "inputvar=outputvar",  \
                 orient=ORIENT_IO, rts=RTS_VRS,emf=EMF_SAME, baseindex=None,refindex=None):
        """DEA: Directional distance function

        Args:
            data (pandas.DataFrame): input pandas.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L= Y:CO2"
            orient(str): ORIENT_IO ORIENT_OO ORIENT_HYPER, or choose some variables in sent
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            emf (String): EMF_SAME (same emission abatement factor) or EMF_DIFFERENT (different emission abatement factor)
            baseindex (String, optional): estimate index. Defaults to None. e.g.: "Year=[2010]"
            refindex (String, optional): reference index. Defaults to None. e.g.: "Year=[2010]"
        """
        # Initialize DEA model

        self.rts = rts
        self.emf = emf
        self.outputvars, self.inputvars ,self.unoutputvars ,self.y, self.x,self.b,self.yref, self.xref,self.bref\
            = tools.assert_valid_dea(sent,data,baseindex,refindex )

        self.xcol = self.x.columns
        self.ycol = self.y.columns
        self.bcol = self.b.columns
        if orient in [ORIENT_IO, ORIENT_OO,ORIENT_UO]:
            self.orient = orient
            self.xindexs = None
            self.yindexs = None
            self.bindexs = None
        else:
            self.orient = None
            if orient in self.xcol:
                self.xindexs = list(self.xcol).index(orient)
                self.yindexs = None
                self.bindexs = None
            elif orient in self.ycol:
                self.yindexs = list(self.ycol).index(orient)
                self.xindexs = None
                self.bindexs = None
            elif orient in self.bcol:
                self.bindexs = list(self.bcol).index(orient)
                self.xindexs = None
                self.yindexs = None

        # print(self.xcol)

        self.I = self.x.index          ## I 是 被评价决策单元的索引      ## 当前被评价决策单元的序号 self.x[I0]
        self.__modeldict = {}

        for i in self.I:
            self.I0 = i
            self.__model__ = ConcreteModel()

            self.__model__.I2 = Set(initialize=  self.xref.index)       ## I2 是 参考决策单元的数量

            self.__model__.K = Set(initialize=range(len(self.x.iloc[0])))   ## K 是投入个数
            self.__model__.L = Set(initialize=range(len(self.y.iloc[0])))   ## L 是产出个数 被评价单元和参考单元的K，L一样
            self.__model__.J = Set(initialize=range(len(self.b.iloc[0]))) ## J 是非期望产出个数 被评价单元和参考单元的K，L一样

            # Initialize variable
            if (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                self.__model__.beta = Var(Set(initialize=range(1)),bounds=(0, 1), doc='efficiency')
            elif self.orient == ORIENT_OO:
                self.__model__.beta = Var(Set(initialize=range(1)),bounds=(1, None), doc='efficiency')
            elif type(self.orient) == type(None):
                if (type(self.xindexs)!=type(None)) | (type(self.bindexs)!=type(None)):
                    self.__model__.beta = Var(Set(initialize=range(1)), bounds=(0, 1), doc='efficiency')
                elif type(self.yindexs)!=type(None):
                    self.__model__.beta = Var(Set(initialize=range(1)), bounds=(1, None), doc='efficiency')


            self.__model__.lamda = Var(self.__model__.I2, bounds=(0.0, None), doc='intensity variables')

            if self.rts == RTS_VRS:
                if self.emf == EMF_SAME:
                    if (self.orient == ORIENT_IO) | (type(self.xindexs)!=type(None) ):
                        self.__model__.beta2 = Var(Set(initialize=range(1)), bounds=(0.0, 1), \
                                                  within=Reals, doc='beta*theta')
                    self.__model__.theta = Var(Set(initialize=range(1)), bounds=(0.0, 1.0), \
                                              within=Reals, doc='theta')
                elif self.emf == EMF_DIFFERENT:
                    self.__model__.phi = Var(self.__model__.I2,bounds=(0.0, None), doc='phi_i')
                    self.__model__.mu = Var(self.__model__.I2,bounds=(0.0, None), doc='mu_i')

                else:
                    raise ValueError("Undefined model parameters.")

            # Setup the objective function and constraints
            if (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                self.__model__.objective = Objective(
                    rule=self.__objective_rule(), sense=minimize, doc='objective function')
            elif self.orient == ORIENT_OO:
                self.__model__.objective = Objective(
                    rule=self.__objective_rule(), sense=maximize, doc='objective function')
            else:
                if (type(self.xindexs)!=type(None)) | (type(self.bindexs)!=type(None)):
                    self.__model__.objective = Objective(
                        rule=self.__objective_rule(), sense=minimize, doc='objective function')
                elif type(self.yindexs)!=type(None):
                    self.__model__.objective = Objective(
                        rule=self.__objective_rule(), sense=maximize, doc='objective function')

            self.__model__.input = Constraint(
                self.__model__.K, rule=self.__input_rule(), doc='input constraint')
            self.__model__.output = Constraint(
                self.__model__.L, rule=self.__output_rule(), doc='output constraint')
            self.__model__.undesirable_output = Constraint(
                self.__model__.J, rule=self.__undesirable_output_rule(),
                                                           doc='undesirable output constraint')

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
        if self.emf == EMF_SAME:
            if (self.orient == ORIENT_IO) | (type(self.xindexs)!=type(None)) :
                def objective_rule(model):
                    return model.beta2[0]*1  + sum(model.lamda[i2] *0 for i2 in model.I2)
            else:
                def objective_rule(model):
                    return model.beta[0]*1  + sum(model.lamda[i2] *0 for i2 in model.I2)
        elif self.emf == EMF_DIFFERENT:
            def objective_rule(model):
                return model.beta[0] * 1 + sum(model.lamda[i2] * 0 for i2 in model.I2)
        else:
            raise ValueError("Undefined model parameters.")
        return objective_rule


    def __input_rule(self):
        """Return the proper input constraint"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                if (self.orient == ORIENT_OO) | (self.orient == ORIENT_UO):
                    def input_rule(model, k):
                        return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                            model.theta * self.x.loc[self.I0,self.xcol[k]]
                elif self.orient == ORIENT_IO:
                    def input_rule(model, k):
                        return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                            model.beta2 * self.x.loc[self.I0,self.xcol[k]]
                else:
                    if type(self.xindexs) != type(None):
                        def input_rule(model, k):
                            if k != self.xindexs:
                                return Constraint.Skip
                            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                                model.beta2 * self.x.loc[self.I0,self.xcol[k]]
                    else: # type(self.yindexs) != type(None) | type(self.bindexs) != type(None)
                        def input_rule(model, k):
                            return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                                model.theta * self.x.loc[self.I0,self.xcol[k]]
            elif self.emf == EMF_DIFFERENT:
                if (self.orient == ORIENT_OO) | (self.orient == ORIENT_UO):
                    def input_rule(model, k):
                        return sum((model.phi[i2] +model.mu[i2])* self.xref.loc[i2, self.xcol[k]] for i2 in model.I2) <= \
                            self.x.loc[self.I0, self.xcol[k]]
                elif self.orient == ORIENT_IO:
                    def input_rule(model, k):
                        return sum((model.phi[i2] +model.mu[i2])* self.xref.loc[i2, self.xcol[k]] for i2 in model.I2) <= \
                            model.beta * self.x.loc[self.I0, self.xcol[k]]
                else:
                    if type(self.xindexs) != type(None):
                        def input_rule(model, k):
                            if k != self.xindexs:
                                return Constraint.Skip
                            return sum(
                                (model.phi[i2] + model.mu[i2]) * self.xref.loc[i2, self.xcol[k]] for i2 in model.I2) <= \
                                model.beta * self.x.loc[self.I0, self.xcol[k]]
                    else:  # type(self.yindexs) != type(None) | type(self.bindexs) != type(None)
                        def input_rule(model, k):
                            return sum(
                                (model.phi[i2] + model.mu[i2]) * self.xref.loc[i2, self.xcol[k]] for i2 in model.I2) <= \
                                self.x.loc[self.I0, self.xcol[k]]

        else:
            if (self.orient == ORIENT_OO) | (self.orient == ORIENT_UO):
                def input_rule(model, k):
                    return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                        self.x.loc[self.I0,self.xcol[k]]
            elif self.orient == ORIENT_IO:
                def input_rule(model, k):
                    return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                        model.beta * self.x.loc[self.I0,self.xcol[k]]
            else:
                if type(self.xindexs) != type(None):
                    def input_rule(model, k):
                        if k != self.xindexs:
                            return Constraint.Skip
                        return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                            model.beta * self.x.loc[self.I0,self.xcol[k]]
                else: # type(self.yindexs) != type(None) | type(self.bindexs) != type(None)
                    def input_rule(model, k):
                        return sum(model.lamda[i2] * self.xref.loc[i2,self.xcol[k]] for i2 in model.I2) <= \
                             self.x.loc[self.I0,self.xcol[k]]
        return input_rule



    def __output_rule(self):
        """Return the proper output constraint"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                if self.orient == ORIENT_OO:
                    def output_rule(model, l):
                        return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                            >=model.beta * self.y.loc[self.I0,self.ycol[l]]
                elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                    def output_rule(model, l):
                        return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                            >= self.y.loc[self.I0,self.ycol[l]]
                else:
                    if type(self.yindexs) != type(None):
                        def output_rule(model, l):
                            if l != self.yindexs:
                                return Constraint.Skip
                            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                                >=model.beta * self.y.loc[self.I0,self.ycol[l]]
                    else:
                        def output_rule(model, l):
                            return sum(model.lamda[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                                >= self.y.loc[self.I0,self.ycol[l]]
            elif self.emf == EMF_DIFFERENT:
                if self.orient == ORIENT_OO:
                    def output_rule(model, l):
                        return sum(model.phi[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                            >=model.beta * self.y.loc[self.I0,self.ycol[l]]
                elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                    def output_rule(model, l):
                        return sum(model.phi[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                            >= self.y.loc[self.I0,self.ycol[l]]
                else:
                    if type(self.yindexs) != type(None):
                        def output_rule(model, l):
                            if l != self.yindexs:
                                return Constraint.Skip
                            return sum(model.phi[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                                >=model.beta * self.y.loc[self.I0,self.ycol[l]]
                    else:
                        def output_rule(model, l):
                            return sum(model.phi[i2] * self.yref.loc[i2,self.ycol[l]] for i2 in model.I2) \
                                >= self.y.loc[self.I0,self.ycol[l]]
            else:
                raise ValueError("Undefined model parameters.")

        elif self.rts == RTS_CRS:
            if self.orient == ORIENT_OO:
                def output_rule(model, l):
                    return sum(model.lamda[i2] * self.yref.loc[i2, self.ycol[l]] for i2 in model.I2) \
                        >= model.beta * self.y.loc[self.I0, self.ycol[l]]
            elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                def output_rule(model, l):
                    return sum(model.lamda[i2] * self.yref.loc[i2, self.ycol[l]] for i2 in model.I2) \
                        >= self.y.loc[self.I0, self.ycol[l]]
            else:
                if type(self.yindexs) != type(None):
                    def output_rule(model, l):
                        if l != self.yindexs:
                            return Constraint.Skip
                        return sum(model.lamda[i2] * self.yref.loc[i2, self.ycol[l]] for i2 in model.I2) \
                            >= model.beta * self.y.loc[self.I0, self.ycol[l]]
                else:
                    def output_rule(model, l):
                        return sum(model.lamda[i2] * self.yref.loc[i2, self.ycol[l]] for i2 in model.I2) \
                            >= self.y.loc[self.I0, self.ycol[l]]
        else:
            raise ValueError("Undefined model parameters.")
        return output_rule

    def __undesirable_output_rule(self):
        """Return the proper undesirable output constraint"""
        if self.rts == RTS_VRS:
            if self.emf == EMF_SAME:
                if self.orient == ORIENT_UO:
                    def undesirable_output_rule(model, j):
                        return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == model.beta * self.b.loc[self.I0, self.bcol[j]]
                elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                    def undesirable_output_rule(model, j):
                        return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == self.b.loc[self.I0, self.bcol[j]]
                else:
                    if type(self.bindexs) != type(None):
                        def undesirable_output_rule(model, j):
                            if j != self.bindexs:
                                return Constraint.Skip
                            return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                                == model.beta * self.b.loc[self.I0, self.bcol[j]]
                    else:
                        def undesirable_output_rule(model, j):
                            return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                                == self.b.loc[self.I0, self.bcol[j]]
            elif self.emf == EMF_DIFFERENT:
                if self.orient == ORIENT_UO:
                    def undesirable_output_rule(model, j):
                        return sum(model.phi[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == model.beta * self.b.loc[self.I0, self.bcol[j]]
                elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                    def undesirable_output_rule(model, j):
                        return sum(model.phi[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == self.b.loc[self.I0, self.bcol[j]]
                else:
                    if type(self.bindexs) != type(None):
                        def undesirable_output_rule(model, j):
                            if j != self.bindexs:
                                return Constraint.Skip
                            return sum(model.phi[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                                == model.beta * self.b.loc[self.I0, self.bcol[j]]
                    else:
                        def undesirable_output_rule(model, j):
                            return sum(model.phi[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                                == self.b.loc[self.I0, self.bcol[j]]
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            if self.orient == ORIENT_UO:
                def undesirable_output_rule(model, j):
                    return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                        == model.beta * self.b.loc[self.I0, self.bcol[j]]
            elif (self.orient == ORIENT_IO) | (self.orient == ORIENT_UO):
                def undesirable_output_rule(model, j):
                    return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                        == self.b.loc[self.I0, self.bcol[j]]
            else:
                if type(self.bindexs) != type(None):
                    def undesirable_output_rule(model, j):
                        if j != self.bindexs:
                            return Constraint.Skip
                        return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == model.beta * self.b.loc[self.I0, self.bcol[j]]
                else:
                    def undesirable_output_rule(model, j):
                        return sum(model.lamda[i2] * self.bref.loc[i2, self.bcol[j]] for i2 in model.I2) \
                            == self.b.loc[self.I0, self.bcol[j]]
        else:
            raise ValueError("Undefined model parameters.")
        return undesirable_output_rule

    def __vrs_rule(self):
        if self.emf == EMF_SAME:
            def vrs_rule(model):
                return sum(model.lamda[ i2] for i2 in model.I2) == model.theta[0] *1
        elif self.emf == EMF_DIFFERENT:
            def vrs_rule(model):
                return sum( (model.phi[i2] + model.mu[i2]) for i2 in model.I2) == 1
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
            if self.emf==EMF_SAME:
                data2 = pd.DataFrame()
                for ind, problem in self.__modeldict.items():
                    _, data2.loc[ind,"optimization_status"] = tools.optimize_model2(problem, ind, solver)
                    data2.loc[ind, "theta"] = np.asarray(list(problem.theta[:].value))

                    if (self.orient == ORIENT_IO) | (type(self.xindexs)!=type(None) ):
                        data2.loc[ind,"beta2"] = np.asarray(list(problem.beta2[:].value))
                        data2["beta"] = data2["beta2"] / data2["theta"]
                    else:
                        data2.loc[ind,"beta"] = np.asarray(list(problem.beta[:].value))
                if self.orient == ORIENT_OO:
                    data2["te"] = 1 / data2["beta"]
                else:
                    data2["te"] = data2["beta"]
            elif self.emf==EMF_DIFFERENT:
                data2 = pd.DataFrame()
                for ind, problem in self.__modeldict.items():
                    _, data2.loc[ind, "optimization_status"] = tools.optimize_model2(problem, ind, solver)
                    data2.loc[ind, "beta"] = np.asarray(list(problem.beta[:].value))
                if self.orient == ORIENT_OO:
                    data2["te"] = 1 / data2["beta"]
                else:
                    data2["te"] = data2["beta"]
            else:
                raise ValueError("Undefined model parameters.")
        elif self.rts == RTS_CRS:
            data2 = pd.DataFrame()
            for ind, problem in self.__modeldict.items():
                _, data2.loc[ind,"optimization_status"] = tools.optimize_model2(problem, ind, solver)
                data2.loc[ind,"beta"] = np.asarray(list(problem.beta[:].value))
            if self.orient == ORIENT_OO:
                data2["te"] = 1 / data2["beta"]
            else:
                data2["te"] = data2["beta"]
        else:
            raise ValueError("Undefined model parameters.")
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


