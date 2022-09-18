from datetime import datetime
from datetime import date
import vnquant.data as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

today = date.today()




three_month_ago = today - relativedelta(months=3)

today = str(today)

three_month_ago = str(three_month_ago)



def get_data():
    data_loader = dt.DataLoader(symbols="FPT",
           start=three_month_ago,
           end=today,
           minimal=True,
           data_source="vnd")
    data = data_loader.download()

    data.columns = ['high', 'low', 'open', 'close', 'avg', 'volume']
    data = data.tail(30)
    Feature= []
    i = 0
    temp=[]
    for j in range(i, i+30):
        for k in range(0, 6):
            temp.append(data.iloc[j,k])
    Feature.append(temp)
    Feature = pd.DataFrame(Feature)
    Feature.to_csv('data_to_test.csv')
    return Feature


