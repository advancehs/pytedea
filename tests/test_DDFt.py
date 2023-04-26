# import packages
from pydea import DDFt
from pydea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_HYPER, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)

# define and solve the test_DEAt model
def test_DDFt():

    model = DDFt.DDFt(data,sent = "K L E CO2=Y",  gy=[1], gx=[1,0,0,0],rts=RTS_VRS, baseindex=None,refindex=None)
    res = model.optimize(solver="mosek")
    print(model.info(1))

test_DDFt()
