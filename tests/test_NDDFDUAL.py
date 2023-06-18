# import packages
from pytedea import NDDFDUAL
from pytedea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL,\
            EMF_SAME, EMF_DIFFERENT

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\BaiduNetdiskDownload\china data.xlsx").query('year>2018').reset_index(drop=True)

# define and solve the test_DEAt model
def test_NDDFDUAL():

    model = NDDFDUAL.NDDFDUAL(data,sent = "K L E=Y:CO2",
                            gy=[1], gx=[0,0,0],gb=[-1],rts=RTS_CRS, baseindex=None,refindex=None)
    res = model.optimize(solver="mosek")
    print(model.info(1))

test_NDDFDUAL()


