# import packages
from pytedea import MQDDF
from pytedea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_UO, RTS_VRS, RTS_CRS, \
                        OPT_LOCAL,EMF_SAME,EMF_DIFFERENT,TOTAL,CONTEMPORARY,MAL,LUE

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2016').reset_index(drop=True)

# define and solve the test_DEAt model
def test_MQDDF():
    model = MQDDF.MQDDF(data,year='year',sent = "K L E=Y:CO2",  gy=[0], gx=[0,0,0],gb=[-1], rts=RTS_VRS,emf=EMF_SAME,\
                        tech = CONTEMPORARY,dynamic=MAL,solver="mosek")
    print(model.optimize()[["MQ","MTECHCH","MEFFCH"]])
test_MQDDF()
