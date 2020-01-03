#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/2/24

# 通过使用sklearn中的集成算法中的SVM，来对前面筛选出来的特征值进行分类
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report
from selectOfMy import findFileInFiles


# 将目标值从字符转换为浮点型数据
def target_type(s):
    it = {'WWW':0, 'SERVICES':1, 'P2P':2, 'MULTIMEDIA':3, 'MAIL':4, 'INTERACTIVE':5, 'GAMES':6, 'FTP-PASV':7,
          'FTP-DATA':8, 'FTP-CONTROL':9, 'DATABASE':10, 'ATTACK':11}
    return it[s]

# 通过网格计算来选出最优的参数
def GridSearchCVToSelect(files,fpath):
    print files
    for file in files:
        start = datetime.datetime.now()
        path = file
        temp = pd.read_csv(fpath + path)
        # print temp.shape[1]
        target = temp.shape[1] - 1  # 分类目标值所在的列号
        data = np.loadtxt(fpath + path, dtype=float, delimiter=',', skiprows=1,
                          converters={target: target_type})  # 将第66列的分类结果用数字来表示,忽略第一行的列名称
        x, y = np.split(data, (target,), axis=1)  # axis=1表示在水平方向上进行分割
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)  # 把样本占比设为0.7
        # cRang=range(1, 100, 5)
        gammaRange = np.arange(0, 1, 0.05)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammaRange, 'C': [50]}]  # 设置组合参数
        scores = ['precision','recall'] # 定义评分方法
        for score in scores:
            print '********Score to:', filter(str.isdigit, file), '*******'
            print 'Tuning hyper-parameters for:', score
            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(x_train, y_train.ravel())
            print "Best parameters set found on development set:"
            print clf.best_params_

            print "Grid scores on development set:"

            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print ("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            y_true, y_pred = y_test, clf.predict(x_test)

            # 打印在测试集上的预测结果与真实值的分数
            print(classification_report(y_true, y_pred, digits=4))
            print 'accuracy:', accuracy_score(y_true, y_pred)
def main():
    fpath = 'FeatureData/'  # 数据集所在的路径
    useData = 'newData.csv'   # 使用的数据集
    files = findFileInFiles(fpath,useData)[:6]  # 只计算前六个数据集
    GridSearchCVToSelect(files,fpath)
    # print files
    # for file in files:
    #     start = datetime.datetime.now()
    #     path =file
    #     temp = pd.read_csv(fpath+path)
    #     # print temp.shape[1]
    #     target=temp.shape[1]-1   # 分类目标值所在的列号
    #     data = np.loadtxt(fpath + path, dtype=float, delimiter=',', skiprows=1,
    #                       converters={target: target_type})  # 将第66列的分类结果用数字来表示,忽略第一行的列名称
    #     x, y = np.split(data, (target,), axis=1)  # axis=1表示在水平方向上进行分割
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)  # 把样本占比设为0.7
    #     C = 50
    #     gamma = 0.5
    #     clf = svm.SVC(C=C, kernel='rbf', gamma=gamma, decision_function_shape='ovr')
    #     clf.fit(x_train, y_train.ravel())
    #
    #     y_pred = clf.predict(x_test)
    #     print filter(str.isdigit,file),':'
    #     print '{Parameter:C=', C, 'gamma=', gamma, '}'
    #     print '选取特征数量：'.decode('utf-8'),temp.shape[1]
    #     print accuracy_score(y_pred, y_test)
    #     print classification_report(y_test,y_pred,digits=4)
    #     end = datetime.datetime.now()
    #     print filter(str.isdigit,file)+'花费时间：'.decode('utf-8'), str(end - start)
    # print data

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print '花费时间：'.decode('utf-8'), str(end - start)