# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:14:07 2019
采用SVR方法进行预测
@author: XMM
"""

from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run APR.")

    parser.add_argument('--input_trainFile', nargs='?', default='features/train_features_k25_cas50.csv',
                        help='Input train cascadaes features file path')

    parser.add_argument('--input_testFile', nargs='?', default='features/validation1_features_k25_cas10.csv',
                        help='Input test cascadaes features file path')

    parser.add_argument('--output', nargs='?', default='../dataset/features/result.csv',
                        help='Features path')
    
    parser.add_argument('--k', type=int, default=25,
                        help='Number of early adopters. Default is 25.')
    
    parser.add_argument('--myModel', type=int, default=0,
                        help='1 denotes user MyModel featuers. 0 denotes APR features.')
    return parser.parse_args()


def read_features_csv(file, My_Model):
    #需要注意第一行数据未读取
    df = pd.read_csv(file)
    columns_size = df.columns.size
    if My_Model == 0:
        X = df.iloc[:,1: columns_size - args.k - 2]
    else:
        X = pd.concat([df.iloc[:,1: columns_size - 2* args.k -2] ,df.iloc[:, columns_size - args.k -2: -2]], axis=1)
    Y = df.iloc[:,-1]
    return X, Y
    
def get_data(train_file, test_file, myModel):
    train_x, train_y = read_features_csv(train_file, myModel)
    test_x, test_y = read_features_csv(test_file, myModel)
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    return train_x, train_y, test_x, test_y    


def set_threshold(pre_reg, threshold):
    res = []
    for i in range(len(pre_reg)):
        if pre_reg[i] > threshold:
           res.append(1)
        else:
            res.append(0)
    return np.array(res)

    
def get_result(y, y_pre):
    accuracy = accuracy_score(y, y_pre)
    precision = precision_score(y,y_pre)
    recall = recall_score(y, y_pre)
    F1 = f1_score(y, y_pre)
    print("accuracy, precision, recall, F1:", accuracy, precision, recall, F1)



        
def Grid_Logistic():
    param_grid = ParameterGrid({'degree':[1, 3, 5, 7], 'C':[.1, 1, 10]})    
    train_x, train_y, test_x, test_y = get_data(args.input_trainFile, args.input_testFile, args.myModel)
    threshold = 0.42
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                     tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                     cache_size=200, verbose=False, max_iter=-1)

    svr.fit(train_x, train_y)
    y_pre = svr.predict(test_x)
    y_pre = set_threshold(y_pre, threshold)
    get_result(test_y, y_pre)
        

def main(args):
    Grid_Logistic()
    
if __name__ == "__main__":
	args = parse_args()
	main(args)