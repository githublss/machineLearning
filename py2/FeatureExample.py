#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/2/18

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def main():
    data1 = pd.DataFrame({'天气':['晴','晴','阴','雨','雨','雨','阴','晴','晴','雨','晴','阴','阴','雨'],
                         '温度':['高','高','高','低','低','低','低','低','低','低','低','低','高','低'],
                         '湿度':['高','低','高','高','高','低','低','高','低','高','低','高','低','高'],
                         '起风':[False,True,False,False,False,True,True,False,False,False,True,True,False,True],
                         '打球':['NO','NO','YES','YES','YES','NO','YES','NO','YES','YES','YES','YES','YES','NO']})
    # print data1[['天气','温度','湿度','起风','打球']]
    # 计算法熵
    def ent(data):
        infor_entropy = pd.value_counts(data)/len(data)   # value_counts()，Return a Series containing counts of unique values.
        return sum(np.log2(infor_entropy) * infor_entropy * (-1))
    def gain(data,str1,str2):
        # 计算条件信息熵
        e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
        p1 = pd.value_counts(data[str1]) / len(data[str1])     # 所占比例
        e2 = sum(e1*p1)
        strGain = ent(data[str2])-e2
        print strGain

    # 计算信息增益率
    def gainRatio(data, str1, str2):
        e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
        p1 = pd.value_counts(data[str1]) / len(data[str1])  # 每个str1在str1中所占比例
        e2 = sum(e1 * p1)
        strGain = ent(data[str2]) - e2
        IV = -sum(map((lambda x: np.log2(x) * x), p1))
        strGainRatio = strGain / IV
        return strGainRatio

    print gainRatio(data1,'天气', '打球')
    print data1[['天气','湿度']]
    print data1.columns.values.tolist()[1]
    for i in range(len(weightFeatureGainRatios)):
        for j in range(len(weightFeatureGainRatios[i])):
            weightFeatureGainRatios[i][j]=(weightFeatureGainRatios[i][j]-np.min(weightFeatureGainRatios))/(np.max(weightFeatureGainRatios)-np.min(weightFeatureGainRatios))
    print weightFeatureGainRatios


if __name__ == '__main__':
    main()