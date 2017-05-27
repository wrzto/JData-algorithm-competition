#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import pickle
import os
from features_generator import get_action_data
import random
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

'''文件路径定义'''
ACTION_FILE = './datasets/JData_Action.csv'
COMMENT_FILE = './datasets/JData_Comment.csv'
PRODUCT_FILE = './datasets/JData_Product.csv'
USER_FILE = './datasets/JData_User.csv'


def compute_str_time(str_t, delta, add = False):
    t = pd.to_datetime(str_t)
    if add:
        return str(t + timedelta(delta))
    else:
        return str(t - timedelta(delta))


def eval_f11(pred, real):
    '''
    param:
        pred --> series
        real --> series
    '''
    p = np.mean(pred.isin(real))
    r = np.sum(pred.isin(real)) / real.shape[0]

    f11 = (6 * p * r) / (5 * r + p)

    print('<---------------我是分隔线--------------->')
    print('f11准确率--->: {0}'.format(p))
    print('f11召回率--->: {0}'.format(r))
    print('f11分数-->: {0}'.format(f11))

    return f11

def eval_f12(pred, real):
    '''
    param:
        pred --> dataframe
        real --> dataframe
    '''
    real['label'] = 1
    pred = pd.merge(pred, real, on=['user_id', 'sku_id'], how='left')
    pred.fillna(0, inplace=True)
    p = pred.label.mean()
    r = np.sum(pred.label) / real.shape[0]

    f12 = (5 * p * r) / (2 * r + 3 * p)
    
    real.drop(['label'], axis=1, inplace=True)
    print('<---------------我是分隔线--------------->')
    print('f12准确率--->: {0}'.format(p))
    print('f12召回率--->: {0}'.format(r))
    print('f12分数-->: {0}'.format(f12))

    return f12

def eval_socre(pred, real):
    '''
    param:
        pred --> dataframe
        real --> dataframe
    '''
    f11 = eval_f11(pred.user_id, real.user_id)
    f12 = eval_f12(pred[['user_id', 'sku_id']], real[['user_id', 'sku_id']])

    socre = 0.4 * f11 + 0.6 * f12
    print('最终分数-->: {0}'.format(socre))

    return socre


def eval_f11_withAB(pred, real):
    '''
    param: 
        pred --> series
        real --> dataframe  ['user_id', 'AorB', ...]
    '''
    A_board = real[real.AorB=='A'].user_id
    B_board = real[real.AorB=='B'].user_id

    p_A = np.sum(pred.isin(A_board)) / pred.shape[0]
    r_A = np.sum(pred.isin(A_board)) / A_board.shape[0]
    p_B = np.sum(pred.isin(B_board)) / pred.shape[0]
    r_B = np.sum(pred.isin(B_board)) / B_board.shape[0]

    f11_A = (6 * p_A * r_A) / (5 * r_A + p_A)
    f11_B = (6 * p_B * r_B) / (5 * r_B + p_B)
    
    print('<---------------我是分隔线--------------->')
    print('A: f11准确率--->: {0}'.format(p_A))
    print('A: f11召回率--->: {0}'.format(r_A))
    print('A: f11分数-->: {0}'.format(f11_A))
    print('<---------------我是分隔线--------------->')
    print('B: f11准确率--->: {0}'.format(p_B))
    print('B: f11召回率--->: {0}'.format(r_B))
    print('B: f11分数-->: {0}'.format(f11_B))

    return f11_A, f11_B

def eval_f12_withAB(pred, real):
    '''
    param:
        pred -->dataframe ['user_id', 'sku_id', ...]
        real -->dataframe ['user_id', 'sku_id', 'AorB', ...]
    '''
    real['label'] = 1
    A_board = real[real.AorB=='A'][['user_id', 'sku_id', 'label']]
    B_board = real[real.AorB=='B'][['user_id', 'sku_id', 'label']]
    A_pred = pd.merge(pred, A_board, on=['user_id', 'sku_id'], how='left').fillna(0)
    B_pred = pd.merge(pred, B_board, on=['user_id', 'sku_id'], how='left').fillna(0)

    p_A = np.mean(A_pred.label)
    r_A = np.sum(A_pred.label) / A_board.shape[0]
    p_B = np.mean(B_pred.label)
    r_B = np.sum(B_pred.label) / B_board.shape[0]

    real.drop(['label'], axis=1, inplace=True)
    f12_A = (5 * r_A * p_A) / (2 * r_A + 3 * p_A)
    f12_B = (5 * r_B * p_B) / (2 * r_B + 3 * p_B)
    print('<---------------我是分隔线--------------->')
    print('A: f12准确率--->: {0}'.format(p_A))
    print('A: f12召回率--->: {0}'.format(r_A))
    print('A: f12分数-->: {0}'.format(f12_A))
    print('<---------------我是分隔线--------------->')
    print('B: f12准确率--->: {0}'.format(p_B))
    print('B: f12召回率--->: {0}'.format(r_B))
    print('B: f12分数-->: {0}'.format(f12_B))

    return f12_A, f12_B

def eval_socre_withAB(pred, real):
    '''
    param:
        pred: dataframe ['user_id', 'sku_id']
        real: dataframe ['user_id', 'sku_id', 'AorB']
    '''
    f11_A, f11_B = eval_f11_withAB(pred.user_id, real[['user_id', 'AorB']])
    f12_A, f12_B = eval_f12_withAB(pred[['user_id', 'sku_id']], real[['user_id', 'sku_id', 'AorB']])
    A_socre = 0.4 * f11_A + 0.6 * f12_A
    B_socre = 0.4 * f11_B + 0.6 * f12_B

    print('A: 最终分数-->: {0}'.format(A_socre))
    print('B: 最终分数-->: {0}'.format(B_socre))
    return A_socre, B_socre


def load_eval_data(start_date = '2016-04-09 00:00:00',end_date = '2016-04-16 00:00:00'):
    dump_path = './cache/eval_date_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'type'])
        df = df[df.type==4][['user_id', 'sku_id']].drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        N = df.shape[0]
        sampleIndex = random.sample(list(range(N)), int(N*0.5))
        df['AorB'] = 'A'
        df.loc[sampleIndex, 'AorB'] = 'B'
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df


def load_sub_eval_data(start_date = '2016-04-09 00:00:00',end_date = '2016-04-16 00:00:00'):
    dump_path = './cache/sub_eval_date_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_eval_data(start_date = start_date, end_date = end_date)
        JProduct = pd.read_csv(PRODUCT_FILE, encoding='gbk')
        JUser = pd.read_csv(USER_FILE, encoding='gbk')
        df = df[df.user_id.isin(JUser.user_id)]
        df = df[df.sku_id.isin(JProduct.sku_id)]
        #对于购买多个商品的用户，只选取一个(对成绩影响应该不大)
        df = df[df.user_id.duplicated()==False]
        df.reset_index(drop=True, inplace=True)
        N = df.shape[0]
        sampleIndex = random.sample(list(range(N)), int(N*0.5))
        df['AorB'] = 'A'
        df.loc[sampleIndex, 'AorB'] = 'B'
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df





    





