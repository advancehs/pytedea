# import packages
from pytedea import DEA
from pytedea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_UO, RTS_VRS, RTS_CRS, \
                        OPT_LOCAL,EMF_SAME,EMF_DIFFERENT

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)

# define and solve the test_DEAt model
def test_DEA():

    # model = DEA.DEA(data,sent = "K L E=Y:CO2",  orient="L", rts=RTS_VRS, baseindex=None,refindex=None)
    model = DEA.DEA(data,sent = "K L E=Y:CO2",  orient="Y", rts=RTS_VRS, emf=EMF_SAME, baseindex=None,refindex=None)
    res = model.optimize(solver="mosek")
    print(res)

test_DEA()
