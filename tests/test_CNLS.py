# import packages
from pytedea import CNLS,StoNED
from pytedea.constant import FUN_PROD, OPT_LOCAL,RTS_VRS ,RTS_CRS,CET_ADDI, CET_MULT,RED_QLE
import pandas as pd

data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)


def test_CNLS():
    model = CNLS.CNLS(data,sent = "K L E CO2=Y", \
                              # cet=CET_MULT, rts = RTS_VRS, fun=FUN_PROD, \
                                cet=CET_ADDI, rts=RTS_CRS, fun=FUN_PROD, \
                                # baseindex="Year=[2010,2011]",refindex="Year=[2010,2011]"
                              )
    # model.optimize( email= '1019753743@qq.com',)
    model.optimize(solver="mosek",)

    rd = StoNED.StoNED(model)
    print(rd.get_technical_inefficiency(RED_QLE))


test_CNLS()
# 有无大于1的非效率
# RTS_VRS  有
# RTS_CRS  没有