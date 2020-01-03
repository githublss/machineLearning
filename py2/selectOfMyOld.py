#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/4/25

import pandas as pd
import numpy as np
import datetime
import os
import copy

# 计算熵的函数,传入的是一个列向量
def ent(data):
    infor_entropy = pd.value_counts(data)/len(data)   # value_counts()，Return a Series containing counts of unique values.
    return sum(np.log2(infor_entropy) * infor_entropy * (-1))

# 计算信息增益
def gain(data,str1,str2):
    # 计算条件信息熵str1为条件，str2为分类结果
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
    p1 = pd.value_counts(data[str1]) / len(data[str1])     # 所占比例
    e2 = sum(e1*p1)
    strGain = ent(data[str2]) - e2
    return strGain


# 计算信息增益率
def gainRatio(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
    p1 = pd.value_counts(data[str1]) / len(data[str1])  # 每个str1在str1中所占比例
    e2 = sum(e1 * p1)
    strGain = ent(data[str2]) - e2
    IV = -sum(map((lambda x: np.log2(x) * x), p1))  # 求出属性str1的固有值。
    strGainRatio = strGain / IV
    return strGainRatio

# 获取fpath文件夹下的type类型文件的列表
def findFileInFiles(fpath, type):
    files = os.listdir(fpath)
    F = []
    for file in files:
        if type in file:
            F.append(file)
    return F

# 将fpach文件夹下的.arff文件转换为csv文件
def arff_to_csv(fpath):
    fileNp=pd.read_csv(fpath,comment='@',header=None)
    fileNp=fileNp.replace('?', '0')     # 对数据集进行替换的替换规则
    fileNp = fileNp.replace('Y', '1')
    fileNp = fileNp.replace('N', '0')
    fpath=fpath.replace('.arff','.csv')
    print(fpath)
    fileNp.to_csv(fpath,index=False)
    print('OK')


# 对fpath文件夹下的数据进行处理，从原始数据得到后面可以用到的数据
def dealData(fpath):
    arffFiles = findFileInFiles(fpath, '.arff')
    for file in arffFiles:
        arff_to_csv(fpath+file)
    csvFiles = findFileInFiles(fpath, 'allclass.csv')
    for csvFile in csvFiles:
        df = pd.read_csv(fpath + csvFile)
        df = df.ix[:, :-1]  # 前面是选择行，后面是选择列
        rankFeatureIndex = pd.read_csv('rankFeatureIndex.csv')  # 读取从weka中获得的特征排序的索引
        rankFeatureIndex['index'] = rankFeatureIndex['index'] - 1
        rankFeature = []  # 存放排好序的特征值
        for i in rankFeatureIndex['index']:
            rankFeature.append(df[str(i)])
        rankFeature = np.array(rankFeature).T  # 将list转化为numpy中的数组
        rankFeaturePd = pd.DataFrame(rankFeature, columns=rankFeatureIndex['index'])
        rankFeatureName = fpath + filter(str.isdigit, csvFile) + 'rankFeature.csv'
        rankFeaturePd.to_csv(rankFeatureName, index=False)  # 生成按照weka生成索引序列顺序的特征表
        print(rankFeatureName + 'save OK!')

def getFeatureByGainRatio(fpath):
    rankFiles = findFileInFiles(fpath, 'rankFeature.csv')[0:1]
    print('will deal file is:',rankFiles)
    FeatureGainRatios = []
    if True:
        for rankFile in rankFiles:  # 遍历所有排好序的文件
            # rankFile = 'wekaTemp.csv'
            print(rankFile)
            df = pd.read_csv(fpath + rankFile)  # 将排好序的特征值赋值给df
            featured = df.columns.values.tolist()  # 先将所有特征的列名称都添加到待选特征里面
            print(featured)
            selectFeatureIndex = []  # 选取到的特征的索引
            deleteFeature = []  # 被剔除的特征的索引
            FeatureGainRatios = []   # 用来存放依次以每个特征作为分类时，与其后面的个个特征之间的信息增益率
            corr = df.corr()  # 计算出相关系数
            corr = np.array(corr)
            for feature1 in featured:  # 对文件中的每个特征进行遍历
                if feature1 in deleteFeature:
                    continue
                selectFeatureIndex.append(feature1)  # 将set集合中的第一个与特征值添加到选择列表中去
                deleteFeature.append(feature1)  # 将特征选择后就记录到删除列表中
                gainSum = 0  # 特征增益的和
                tempFeatureGainRatio = []  # 存放feature1与其他特征间的信息增益列表
                feature2list = []  # 存放feature1与其他特征间的信息增益列表的索引
                for feature2 in featured:
                    if (feature2 in selectFeatureIndex) or (feature2 in deleteFeature):  # 为了不重复计算已经选择的属性之间的信息熵
                        continue
                    feature2list.append(feature2)
                    featureGainRatio = gainRatio(df, feature2, feature1)  # 计算出feature1与feature2之间的信息增益率
                    tempFeatureGainRatio.append(featureGainRatio)
                    gainSum = gainSum + featureGainRatio
                    print(feature1, feature2, featureGainRatio)
                if tempFeatureGainRatio == []:
                    continue
                FeatureGainRatios.append(tempFeatureGainRatio)   # 将所求的单个的信息信息增益率都记录到总表中
                feature1Index = featured.index(feature1)
                weightFeatureGainRatioMulCorr = np.array(FeatureGainRatios[-1]) * corr[feature1Index+1:,feature1Index]
                for i in range(weightFeatureGainRatioMulCorr):
                    if weightFeatureGainRatioMulCorr[i] >= (np.max(weightFeatureGainRatioMulCorr)+np.min(weightFeatureGainRatioMulCorr))/2:
                        deleteFeature.append(featured[i+feature1Index+1])



                # tempFeatureGain = pd.Series(tempFeatureGain, index=feature2list)
                # gainMean = tempFeatureGain.mean()
                # print('feature Gain list is:\n', tempFeatureGain)
                # print('delete feature is):'
                # for x in tempFeatureGain[tempFeatureGain > gainMean].index:  # 剔除的条件是信息增益是否大于平均值
                #     print(x)
                #     featured.remove(str(x))  # 将信息增益大于平均值的特征剔除掉
                #     deleteFeature.append(x)  # 将剔除的特征添加到剔除列表中
                # print('will select:', featured)
                # print('selectd:', selectFeatureIndex)
            print(FeatureGainRatios)
            pd.DataFrame(FeatureGainRatios).to_csv(fpath+filter(str.isdigit,rankFile)+'FeatureGainRatios.csv',index=False)
    rankFile = rankFiles[0]
    # FeatureGainRatiosTemp = pd.read_csv(fpath+filter(str.isdigit,rankFile)+'FeatureGainRatios.csv',)
    # FeatureGainRatios = np.loadtxt(fpath+filter(str.isdigit,rankFile)+'FeatureGainRatios.csv',skiprows=1,dtype=float,delimiter=',')

    # # 将保存的信息增益信息读出来
    # count =1
    # for temp in FeatureGainRatiosTemp:
    #     print(temp)
    #     # FeatureGainRatios.append(FeatureGainRatiosTemp.iloc[:,:])
    # print(FeatureGainRatiosTemp)


    weightFeatureGainRatios = []
    for FeatureGainRatio in FeatureGainRatios:
        tempWeightFeatureGainRatio = []
        for GainRatio in FeatureGainRatio:
            if sum(FeatureGainRatio) == 0:
                tempWeightFeatureGainRatio.append(0)
                continue
            tempWeightFeatureGainRatio.append(GainRatio*len(FeatureGainRatio)/sum(FeatureGainRatio))    # 根据所给公式计算每一个元素与后面元素的信息增益率权重
        weightFeatureGainRatios.append(tempWeightFeatureGainRatio)   # 将特定特征带权重的信息增益率列表添加到总表中
    # 数据的归一化
    normWeightFeatureGainRatios = copy.deepcopy(weightFeatureGainRatios)    # 存放WeightFeatureGainRatios归一化后的数据
    for i in range(len(weightFeatureGainRatios)):
        for j in range(len(weightFeatureGainRatios[i])):
            normWeightFeatureGainRatios[i][j]=(weightFeatureGainRatios[i][j]-np.min(np.min(weightFeatureGainRatios,axis=0)))/\
                                          (np.max(np.max(weightFeatureGainRatios,axis=0))-np.min(np.min(weightFeatureGainRatios,axis=0)))
            # print(weightFeatureGainRatios[i][j],np.max(np.max(weightFeatureGainRatios,axis=0)),np.min(np.min(weightFeatureGainRatios,axis=0)))
    print(normWeightFeatureGainRatios)

    corr = df.corr()    # 计算出相关系数
    corr = np.array(corr)
    print(corr)


    weightFeatureGainRatioMulCorrs = []     # 存放相关系数乘以带权重的特征增益率的值
    count = 1
    for weightFeatureGainRatio in normWeightFeatureGainRatios:
        weightFeatureGainRatioMulCorrs.append(corr[count:,count-1] * weightFeatureGainRatio)  # 每次取corr中一列数据中的后count行
        count = count + 1
    selectFeatureIndex = []
    for i in range(len(weightFeatureGainRatioMulCorrs)):    # i表示是第几行的特征值
        if i in selectFeatureIndex:
            continue
        selectFeatureIndex.append(i)
        for j in range(len(weightFeatureGainRatioMulCorrs[i])):
            if weightFeatureGainRatioMulCorrs[i][j] >= ((np.max(weightFeatureGainRatioMulCorrs[i]) + np.min(weightFeatureGainRatioMulCorrs[i]))/2):
                selectFeatureIndex.append(i+j+1)

    print(selectFeatureIndex)
    pd.DataFrame(selectFeatureIndex).to_csv(fpath + filter(str.isdigit, rankFile) + 'selectFeatureByRatios.csv', index=False)
    # pd.Series(selectFeatureIndex).to_csv(fpath + filter(str.isdigit, rankFile) + 'selectFeatureIndex.csv',
    #                                      encoding='utf-8', index=False)

# 按照设计的特征选择算法，进行特征的选择
def getBestFeature(fpath):
    rankFiles = findFileInFiles(fpath,'rankFeature.csv')
    for rankFile in rankFiles:  # 遍历所有排好序的文件
        df = pd.read_csv(fpath+rankFile)  # 将排好序的特征值赋值给df
        featured = df.columns.values.tolist()  # 先将所有特征都添加到待选特征里面
        selectFeatureIndex = []  # 选取到的特征的索引
        deleteFeature = []  # 被剔除的特征的索引
        for feature1 in featured:   # 对文件中的每个特征进行遍历
            selectFeatureIndex.append(feature1)
            gainSum = 0  # 特征增益的和
            tempFeatureGain = []  # 存放feature1与其他特征间的信息增益列表
            feature2list = []  # 存放feature1与其他特征间的信息增益列表的索引
            for feature2 in featured:
                if feature2 in selectFeatureIndex:  # 为了不重复计算已经选择的属性之间的信息熵
                    continue
                feature2list.append(feature2)
                featureGain = gain(df, feature2, feature1)  # 计算出feature1与feature2之间的信息增益
                tempFeatureGain.append(featureGain)
                gainSum = gainSum + featureGain
                print(feature1, feature2, featureGain)
            tempFeatureGain = pd.Series(tempFeatureGain, index=feature2list)
            gainMean = tempFeatureGain.mean()
            print('feature Gain list is:\n', tempFeatureGain)
            print('delete feature is:')
            for x in tempFeatureGain[tempFeatureGain > gainMean].index:  # 剔除的条件是信息增益是否大于平均值
                print(x)
                featured.remove(str(x))  # 将信息增益大于平均值的特征剔除掉
                deleteFeature.append(x)  # 将剔除的特征添加到剔除列表中
            print('will select:', featured)
            print('selectd:', selectFeatureIndex)
        pd.Series(selectFeatureIndex).to_csv(fpath+filter(str.isdigit,rankFile)+'selectFeatureIndex.csv', encoding='utf-8', index=False)

# 通过selectFeatureIndex.csv文件将特征值序列提取出来
def selectFeature(fpath):
    type = 'selectFeatureIndexCustom.csv'   # 抽取出来的特征索引顺序表
    toType = 'selectFeatureCustom.csv'      # 要保存成的文件名
    selectIndexs=findFileInFiles(fpath,type)
    for selectIndex in selectIndexs:
        rankFeaturePd = pd.read_csv(fpath+filter(str.isdigit,selectIndex)+'rankFeature.csv')
        selectFeatureIndex = pd.read_csv(fpath+filter(str.isdigit,selectIndex)+type, header=None, names=['index'])  # 将列名指定为index
        print(selectFeatureIndex)
        if 'Custom' in type:    # 自己定义的索引值从1开始的，所以减一
            selectFeatureIndex = selectFeatureIndex-1
        index = selectFeatureIndex['index']
        selectFeature = []  # 存放排好序的特征值
        for i in index:
            selectFeature.append(rankFeaturePd[str(i)])
        selectFeature = np.array(selectFeature).T  # 将list转化为numpy中的数组
        selectFeaturePd = pd.DataFrame(selectFeature, columns=selectFeatureIndex['index'])
        selectFeaturePd['target'] = pd.read_csv(fpath+'entry'+filter(str.isdigit,selectIndex)+'.weka.allclass.csv', encoding='utf-8')[
            '248']  # 将目标值添加到selectFeature中
        selectFeaturePd.to_csv(fpath+filter(str.isdigit,selectIndex)+toType, index=False)
        print('ok')
def main():
    fpath = 'FeatureData/'
    # fpath = ''
    # dealData(fpath)           # 将原始数据按照weka中获得特征索引整体进行排序
    # getBestFeature(fpath)     # 通过信息增益来选特征的索引值
    getFeatureByGainRatio(fpath)    # 通过信息增益率来选特征的索引值
    # selectFeature(fpath)        # 通过上一步抽取出来的索引值，得到一个特征数据表


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print('花费时间：'.decode('utf-8') , str(end - start))