# import packages
import pandas as pd
from pytedea import weakCNLSDDF,StoNED
from pytedea.constant import FUN_PROD, OPT_LOCAL,RTS_VRS,RED_MOM,RED_QLE,RTS_CRS

data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)


def test_weakCNLSDDF():

    model = weakCNLSDDF.weakCNLSDDF(data,sent = "K L E=Y:CO2", gy=[1], gx=[1,1,1],gb=[1],  \
                               fun=FUN_PROD, rts = RTS_VRS, \
                                      )

    model.optimize( solver="mosek")

    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_QLE))

test_weakCNLSDDF()

# RTS_VRS  有
# RTS_CRS  没有
