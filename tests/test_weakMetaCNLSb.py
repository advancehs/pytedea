# import packages
from pytedea import weakMetaCNLSb,StoNED,weakCNLSb
from pytedea.constant import FUN_PROD, OPT_LOCAL,RTS_VRS ,RTS_CRS,CET_ADDI, CET_MULT,RED_QLE
import pandas as pd

data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\china data.xlsx").query('year>2018').reset_index(drop=True)
# data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\oecd data.xlsx").query('year>2015').reset_index(drop=True)


def test_weakMetaCNLSb():
    model = weakCNLSb.weakCNLSb(data,sent = "K L E=Y:CO2", \
                              # cet=CET_MULT, rts = RTS_VRS, fun=FUN_PROD, \
                              cet=CET_ADDI, rts = RTS_VRS, fun=FUN_PROD, \
                              )
    # model.optimize( email= '1019753743@qq.com',)
    model.optimize( solver="ipopt")

    rd = StoNED.StoNED(model)
    GCE = rd.get_technical_efficiency(RED_QLE)
    GCE = pd.DataFrame(GCE)
    print("GCE",GCE)

    model2 = weakMetaCNLSb.weakMetaCNLSb(data,GCE,sent = "K L E=Y:CO2", \
                                cet=CET_MULT, rts=RTS_VRS, fun=FUN_PROD, \
                              )
    # model2.optimize( email= '1019753743@qq.com',)
    model2.optimize(solver="ipopt")  #   mosek

    rd2 = StoNED.StoNED(model2)
    TGR = rd2.get_technical_efficiency(RED_QLE)

    print("TGR",TGR)

test_weakMetaCNLSb()

# RTS_VRS  没有
# RTS_CRS  没有
