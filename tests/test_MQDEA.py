# import packages
from pytedea import MQDEA
from pytedea.constant import CET_ADDI, ORIENT_IO, ORIENT_OO,ORIENT_UO, RTS_VRS, RTS_CRS, \
                        OPT_LOCAL,EMF_SAME,EMF_DIFFERENT,TOTAL,CONTEMPORARY

import pandas as pd
# import all data (including the contextual varibale)
data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2016').reset_index(drop=True)

# define and solve the test_DEAt model
def test_MQDEA():
    model = MQDEA.MQDEA(data,year='year',sent = "K L E=Y:CO2",  orient="Y", rts=RTS_VRS,\
                        tech = CONTEMPORARY,solver="mosek")
    print(model.optimize()[["MQ","MTECHCH","MEFFCH"]])
test_MQDEA()
