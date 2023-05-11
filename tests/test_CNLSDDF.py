# import packages
import pandas as pd
from pytedea import CNLSDDF,StoNED
from pytedea.constant import FUN_PROD, OPT_LOCAL,RTS_VRS,RED_MOM,RED_QLE,RTS_CRS

data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)


def test_CNLSDDF():

    model = CNLSDDF.CNLSDDF(data,sent = "K L E CO2=Y", gy=[1], gx=[1,1,1,1],  \
                               fun=FUN_PROD, rts = RTS_CRS, \
                                      )

    model.optimize( solver="mosek")

    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_QLE))

test_CNLSDDF()
# 有无大于1的非效率
# RTS_VRS  没有
# RTS_CRS  没有
