# import packages
from pydea import DEAt
from pydea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_HYPER, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)

# define and solve the test_DEAt model
def test_DEAt():
    # model = DEAt.DEAt(data,sent = "K L E CO2=Y",  orient=ORIENT_IO, rts=RTS_VRS, baseindex=None,refindex=None)
    # res = model.optimize(solver="mosek")
    # print(model.info(1))
    model = DEAt.DEAt(data,sent = "K L E CO2=Y",  orient=ORIENT_HYPER, rts=RTS_VRS, baseindex=None,refindex=None)
    res = model.optimize(solver="mosek")
    print(model.info(1))

test_DEAt()
