"""Main module."""
# import dependencies

import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_UO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL,\
            TOTAL,CONTEMPORARY,EMF_SAME,EMF_DIFFERENT,MAL,LUE
from .utils import tools
from .DDFt import DDFt



class MQDDFt:
    """Malmquist production index (MQPI)
    """

    def __init__(self, data,year,sent = "inputvar=outputvar", gy = [1], gx = [1],  rts=RTS_VRS,\
                 tech = TOTAL,dynamic = MAL,solver="mosek"):
        """DEA: Envelopment problem

        Args:
            data (pandas.DataFrame): input pandas.
            year (str): column name to specify time.
            sent (str): inputvars=outputvars[: unoutputvars]. e.g.: "K L CO2= Y"
            orient (String): ORIENT_IO (input orientation) or ORIENT_OO (output orientation)
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            tech (str): TOTAL or CONTEMPORARY
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        # Initialize DEA model

        self.tech=tech
        self.tlt = pd.Series(data[year]).drop_duplicates().sort_values() ## 生成时间的列表

        if self.tech== TOTAL:
            dataz11 = pd.DataFrame()      # D11
            for tindex in self.tlt.index:

                model =DDFt(data,sent = sent, gy =gy, gx =gx,\
                                 rts=rts, baseindex="{}=[{}]".format(year,self.tlt.iloc[tindex]))

                data11  = model.optimize(solver)#.drop(self.data.columns)
                dataz11 = pd.concat([dataz11,data11])
            print("22222222")

            datazz = pd.concat([data,dataz11],axis=1)


            datazz["mqpi"] = datazz["D11"] / datazz["D11"].shift(1)
            datazz.drop(columns = ["D11"],inplace = True)
            print("11111111111111111111111")

        elif self.tech== CONTEMPORARY:
            print("3333333333")

            dataz11 = pd.DataFrame()      # D11
            for tindex in self.tlt.index:
                model = DDFt(data=data,sent=sent, gy =gy, gx =gx, rts=rts,\
                                baseindex="{}=[{}]".format(year,self.tlt.iloc[tindex]),\
                                refindex="{}=[{}]".format(year,self.tlt.iloc[tindex]))
                data11  = model.optimize(solver)
                dataz11 = pd.concat([dataz11,data11])

            datazz = pd.concat([data,dataz11[["beta"]]],axis=1).rename(columns = {"beta":"D11"} )

            dataz12 = pd.DataFrame()      # D12
            for tindex in self.tlt.index[1:]:
                model = DDFt(data=data,sent=sent, gy =gy, gx =gx, rts=rts,\
                                baseindex="{}=[{}]".format(year,self.tlt.iloc[tindex]), \
                                refindex="{}=[{}]".format(year,self.tlt.iloc[tindex-1]))
                data12  = model.optimize(solver)
                dataz12 = pd.concat([dataz12,data12])

            datazz = pd.concat([datazz,dataz12[["beta"]]],axis=1).rename(columns = {"beta":"D12"} )

            dataz21 = pd.DataFrame()      # D21
            for tindex in self.tlt.index[:-1]:
                model = DDFt(data=data,sent=sent, gy =gy, gx =gx, rts=rts,\
                                baseindex="{}=[{}]".format(year,self.tlt.iloc[tindex]), \
                                refindex="{}=[{}]".format(year,self.tlt.iloc[tindex+1]))
                data21  = model.optimize(solver)#.drop(self.data.columns)
                dataz21 = pd.concat([dataz21,data21])
            datazz = pd.concat([datazz,dataz21[["beta"]]],axis=1).rename(columns = {"beta":"D21"} )
            if dynamic==LUE:
                datazz["MQ"] = 0.5*((datazz["D11"].shift(1)-datazz["D12"])\
                                       +(datazz["D21"].shift(1)- datazz["D11"]))
                # datazz.drop(columns = ["D11","D12","D21"],inplace = True)
                datazz["MEFFCH"] = (datazz["D11"].shift(1))-(datazz["D11"])
                datazz["MTECHCH"] = 0.5*((datazz["D11"]-datazz["D12"])+\
                                            (datazz["D21"].shift(1)-datazz["D11"].shift(1)))
            elif dynamic==MAL:
                if (abs(np.asarray(gy).sum())>=1) : # gy>0
                    if (abs(np.asarray(gx).sum())==0):# gy>0 # gx=0
                        datazz["MQ"] = np.sqrt((1+datazz["D11"].shift(1))/(1+datazz["D12"])\
                                               *(1+datazz["D21"].shift(1))/ (1+datazz["D11"]))
                        # datazz.drop(columns = ["D11","D12","D21"],inplace = True)
                        datazz["MEFFCH"] = (1+datazz["D11"].shift(1))/(1+datazz["D11"])
                        datazz["MTECHCH"] = np.sqrt((1+datazz["D11"])/(1+datazz["D12"])*\
                                                    (1+datazz["D21"].shift(1))/(1+datazz["D11"].shift(1)))
                    else: # gy>0 # gx>0
                        raise ValueError("Undefined model parameters.")
                else:# gy=0
                    if (abs(np.asarray(gx).sum())==0):# gy=0 gx=0
                        raise ValueError("Undefined model parameters.")
                    else: # gy=0 gx>0
                        datazz["MQ"] = np.sqrt((1-datazz["D12"])/(1-datazz["D11"].shift(1))\
                                               *(1-datazz["D11"])/ (1-datazz["D21"].shift(1)))
                        # datazz.drop(columns = ["D11","D12","D21"],inplace = True)
                        datazz["MEFFCH"] = (1-datazz["D11"])/(1-datazz["D11"].shift(1))
                        datazz["MTECHCH"] = np.sqrt((1-datazz["D12"])/(1-datazz["D11"])*\
                                                    (1-datazz["D11"].shift(1))/(1-datazz["D21"].shift(1)))

            datazz = datazz.fillna(1)

            self.datazz = datazz
    def optimize(self):

        return self.datazz










