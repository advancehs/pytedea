from pytedea import MB
from pytedea.constant import RTS_CRS,RTS_VRS, ORIENT_OO ,ORIENT_IO
import pandas as pd

data = pd.read_excel(r"D:\Pythonwork\一带一路\处理一带一路数据\power plant data.xlsx").query('year>2014').reset_index(drop=True)
print(data.head())
data.columns = ['province', 'K', 'L', 'F', 'E', 'CO2', 'year', '省份']

# 含有CO2的F生产包含CO2的CO2
sx = data['CO2'].sum()/data['F'].sum()
print(sx)
sy = 0
model =MB.MB(data,sent = "K L+F=E:CO2", sx=[[0,0,sx]], sy=[[sy]], \
                 rts=RTS_CRS,baseindex=None,refindex=None,level=3)
data3,info =  model.optimize("mosek","1")

print(data3 )
# print(info)


# print("###########################################")
# model2 =teddf.MBxy(data,"Year",sent = "HRSN CPNK=VALK:GHG", sx=[[0,0.0001360080088453017]], sy=[[0.00000360080088453017]], \
#                  rts=RTS_VRS,baseindex=None,refindex=None)
# data32 =  model2.optimize("mosek")
# print(data32 )
# print(model2.info("29"))
