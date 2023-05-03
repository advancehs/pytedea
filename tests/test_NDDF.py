# import packages
from pytedea import NDDF
from pytedea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL,\
            EMF_SAME, EMF_DIFFERENT

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)

# define and solve the test_DEAt model
def test_NDDF():

    model = NDDF.NDDF(data,sent = "K L E=Y:CO2",  gy=[1], gx=[-1,-1,-1],gb=[0],rts=RTS_VRS,emf=EMF_DIFFERENT, baseindex=None,refindex=None)
    res = model.optimize(solver="mosek")
    print(model.info(1))
    # res.to_excel("sss.xlsx")
test_NDDF()
