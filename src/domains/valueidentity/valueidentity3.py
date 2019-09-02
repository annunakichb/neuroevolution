# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def createSamples():
    df = pd.read_excel("/Volumes/data/shared/perfume_data.xlsx", header=None)
    rownum, colnum = df.shape
    values = []
    lables = []
    for index in df.index:
        classname = df.loc[index].values[0]
        print(classname)
        row = df.loc[index].values[1:-1]
        data1 = [v // 1000 for v in row]
        data1 = (data1 - data1.min()) / (data1.max() - data1.min())
        data2 = [v % 1000 for v in row]
        data2 = (data2 - data2.min()) / (data2.max() - data2.min())
        datas = zip(data1,data2)
        values.append(datas)
        lables.append(classname)
        #plt.scatter(data1, data2)
    return values,lables

if __name__ == '__main__':
    pass
