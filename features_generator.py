#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import pickle
import os
import re
import warnings
warnings.filterwarnings("ignore")

'''文件路径定义'''
ACTION_FILE = './datasets/JData_Action.csv'
COMMENT_FILE = './datasets/JData_Comment.csv'
PRODUCT_FILE = './datasets/JData_Product.csv'
USER_FILE = './datasets/JData_User.csv'
TIMEDECAY = './datasets/TimeDecay.csv'

def get_action_data(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', chunk_size=100000, field=['user_id', 'sku_id', 'time', 'type', 'cate', 'brand']):
    '''
    分块读取数据
    '''
    field.sort()
    dump_path = './cache/Action_{0}_{1}_with_{2}'.format(start_date[:10], end_date[:10], '-'.join(field))
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        reader = pd.read_csv(ACTION_FILE, iterator=True)
        chunks = []
        loop = True
        while loop:
            try:
                chunk = reader.get_chunk(chunk_size)
                chunk = chunk[(chunk.time >= start_date) & (chunk.time < end_date)][field]
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print('Iteration is stopped.')
        df = pd.concat(chunks, ignore_index=True)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df

'''特征提取函数'''
def load_user_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    提取用户行为计数,不带时间衰减.
    '''
    dump_path = './cache/user_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'time', 'type'])
        prefix = 'Action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        drop_cols = ['time', 'type']
        df.drop(drop_cols, axis=1, inplace=True)
        df = df.groupby(['user_id'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df

def load_user_action_cnt_with_timeDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    提取用户行为计数,带有时间衰减.
    '''
    dump_path = './cache/user_action_cnt_with_timeDecay_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'time', 'type'])
        prefix = 'Action_cnt_{0}_{1}_with_timeDecay'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        ###加入时间衰减
        df['time'] = pd.to_datetime(df['time'])
        end_date = pd.to_datetime(end_date)
        df['time_dist'] = df.time.apply(lambda t: int((end_date-t).total_seconds()/60/60)+1)
        w = pd.read_csv(TIMEDECAY)
        df = pd.merge(df, w, on=['time_dist'], how='left')
        df[prefix + '_1'] *= df['weight']
        df[prefix + '_2'] *= df['weight']
        df[prefix + '_3'] *= df['weight']
        df[prefix + '_4'] *= df['weight']
        df[prefix + '_5'] *= df['weight']
        df[prefix + '_6'] *= df['weight']
        drop_cols = ['time', 'type', 'weight', 'time_dist']
        df.drop(drop_cols, axis=1, inplace=True)
        df = df.groupby(['user_id'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df

def load_buy_to_others_ratio(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    计算购买转化率
    '''
    dump_path = './cache/buy_to_others_ratio_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_user_action_cnt(start_date = start_date, end_date = end_date)
        cols = df.columns.tolist()
        cols = ''.join(cols)
        colname_4 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_4', cols)[0]
        colname_1 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_1', cols)[0]
        colname_2 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_2', cols)[0]
        colname_3 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_3', cols)[0]
        colname_5 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_5', cols)[0]
        colname_6 = re.findall('Action_cnt_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_6', cols)[0]

        df['Action_ratio_4_1'] = df[colname_4] / df[colname_1]
        df['Action_ratio_4_2'] = df[colname_4] / df[colname_2]
        df['Action_ratio_4_3'] = df[colname_4] / df[colname_3]
        df['Action_ratio_4_5'] = df[colname_4] / df[colname_5]
        df['Action_ratio_4_6'] = df[colname_4] / df[colname_6]

        save_cols = ['user_id', 'Action_ratio_4_1', 'Action_ratio_4_2', 'Action_ratio_4_3', 'Action_ratio_4_5', 'Action_ratio_4_6']
        df = df[save_cols]
        df.fillna(1, inplace=True)
        df.replace(np.inf, 1, inplace=True)
        
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df

def load_base_user_feat(end_date='2016-04-16'):
    '''
    提取用户基础信息
    '''
    dump_path = './cache/base_user_feat_{0}.pkl'.format(end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(USER_FILE, encoding='gbk')
        # sex_dummies = pd.get_dummies(df.sex, prefix='sex')
        df.user_reg_tm.fillna('2016-02-01', inplace=True)
        df.user_reg_tm = pd.to_datetime(df.user_reg_tm).apply(lambda t: pd.to_datetime('2016-02-01') if t > pd.to_datetime('2016-04-15') else t)
        df['reg_tm_dist'] = df.user_reg_tm.apply(lambda t: (pd.to_datetime(end_date) - t).days)
        df = df[['user_id', 'user_lv_cd', 'reg_tm_dist']]
        # df = pd.concat([df, sex_dummies], axis=1)
        # age_dummies = pd.get_dummies(df.age, prefix='age')
        # N = age_dummies.shape[1]
        # age_dummies.columns = ['age_{0}'.format(i) for i in range(N)]
        # df = pd.concat([df, age_dummies], axis=1)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df

def load_user_login_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    计算用户第一次登陆到现在的时间距离(单位：秒)
    '''
    dump_path = './cache/user_login_tm_dist_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time'])
        df.time = pd.to_datetime(df.time)
        df = df.groupby(['user_id'], as_index=False).time.min()
        df['login_tm_dist'] = df.time.apply(lambda t: (pd.to_datetime(end_date) - t).total_seconds())
        df = df[['user_id', 'login_tm_dist']]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_user_last_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    计算最后一次操作的时间距离
    '''
    dump_path = './cache/user_last_tm_dist_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time'])
        df.time = pd.to_datetime(df.time)
        df = df.groupby(['user_id'], as_index=False).time.max()
        df['last_tm_dist'] = df.time.apply(lambda t: (pd.to_datetime(end_date) - t).total_seconds())
        df = df[['user_id', 'last_tm_dist']]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_user_cnt_Nitem_withCate(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate=[8]):
    '''
    计算用户一段时间内看过某类商品多少种
    '''
    dump_path = './cache/user_cnt_Nitem_withCate_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'cate'])
        df = df.drop_duplicates()
        df = df.groupby(['user_id','cate']).size().reset_index(name='Nitem_{0}_{1}'.format(start_date[:10], end_date[:10]))

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    df = df[df.cate.isin(cate)]

    return df

def load_UCPair_onlyact(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate=[8]):
    '''
    计算用户在某段时间内是否只看过某类商品
    '''
    df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'cate'])
    df = df.drop_duplicates()
    temp = df.groupby(['user_id']).size().reset_index(name='ncate')
    df = pd.merge(df, temp, on=['user_id'], how='left')
    df = df[df.cate==8]
    df['ncate'] = (df['ncate'] == 1).astype(int)

    return df[['user_id', 'cate', 'ncate']]

# def get_uid_label(start_date = '2016-02-01 00:00:00', end_date = '2016-04-15 00:00:00'):
#     dump_path = './cache/uid_label_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
#     if os.path.exists(dump_path):
#         with open(dump_path, 'rb') as f:
#             df = pickle.load(f)
#     else:
#         df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'type'])
#         df = df[df.type==4].user_id.drop_duplicates().to_frame()
#         with open(dump_path, 'wb') as f:
#             pickle.dump(df, f)
#     return df

def get_uid_with_sid_label(start_date = '2016-02-01 00:00:00', end_date = '2016-04-15 00:00:00'):
    '''
    获取全集Label
    '''
    dump_path = './cache/uid_with_sid_label_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'sku_id', 'type'])
        df = df[df.type==4][['user_id', 'sku_id']].drop_duplicates()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    uid = df.user_id.drop_duplicates().to_frame()
    return uid, df

def get_sub_uid_with_sid_label(start_date = '2016-02-01 00:00:00', end_date = '2016-04-15 00:00:00'):
    '''
    获取子集Label
    '''
    dump_path = './cache/sub_uid_with_sid_label_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        _, df = get_uid_with_sid_label(start_date = start_date, end_date = end_date)
        JProduct = pd.read_csv(PRODUCT_FILE, encoding='gbk')
        JUser = pd.read_csv(USER_FILE, encoding='gbk')
        df = df[df.sku_id.isin(JProduct.sku_id)]
        df = df[df.user_id.isin(JUser.user_id)]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    uid = df.user_id.drop_duplicates().to_frame()
    return uid, df

def load_action_ratio_with_timeWindow(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                                        sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00'):
    '''
    统计时间窗口内用户的行为比例
    '''
    dump_path ='./cache/action_ratio_with_tm_{0}_{1}_{2}_{3}.pkl'.format(start_date[:10], end_date[:10], sub_start_date[:10], sub_end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_user_action_cnt(start_date = start_date, end_date = end_date)
        sub_df = load_user_action_cnt(start_date = sub_start_date, end_date = sub_end_date)
        prefix_1 = 'Action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        prefix_2 = 'Action_cnt_{0}_{1}'.format(sub_start_date[:10], sub_end_date[:10])
        df = pd.merge(sub_df, df, on='user_id', how='left')
        df['Action_cnt_ratio_browse']  = df[prefix_2 +'_1'] / df[prefix_1 + '_1']
        df['Action_cnt_ratio_cart']  = df[prefix_2 +'_2'] / df[prefix_1 + '_2']
        df['Action_cnt_ratio_delcart']  = df[prefix_2 +'_3'] / df[prefix_1 + '_3']
        df['Action_cnt_ratio_buy']  = df[prefix_2 +'_4'] / df[prefix_1 + '_4']
        df['Action_cnt_ratio_favor']  = df[prefix_2 +'_5'] / df[prefix_1 + '_5']
        df['Action_cnt_ratio_click']  = df[prefix_2 +'_6'] / df[prefix_1 + '_6']

        save_cols = ['user_id', 'Action_cnt_ratio_browse', 'Action_cnt_ratio_cart', 'Action_cnt_ratio_delcart', \
                    'Action_cnt_ratio_buy', 'Action_cnt_ratio_favor', 'Action_cnt_ratio_click']
        df = df[save_cols]
        df.replace(np.inf, 1, inplace=True)
        df.fillna(1, inplace=True)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df
def load_user_action_date_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户在一段时间内多少天有过操作(可理解为登陆京东天数)
    '''
    dump_path = './cache/user_action_date_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field = ['user_id', 'time'])
        df.time = df.time.apply(lambda t: t[:10])
        df = df.drop_duplicates()
        colname = 'user_action_date_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = df.groupby(['user_id'], as_index=False).size().reset_index(name=colname)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_user_action_totalCnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,6]):
    '''
    统计用户在一段时间的总操作数
    '''
    df = load_user_action_cnt(start_date = start_date, end_date = end_date)
    cols = ['user_id']
    cols.extend(['Action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[cols]
    colname = 'Action_totalCnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], '_'.join([str(i) for i in actions]))
    df[colname] = df.drop(['user_id'], axis=1).sum(axis=1)

    return df[['user_id', colname]]

def load_user_action_totalCnt_withDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,6]):
    '''
    统计用户在一段时间的总操作数(带时间衰减)
    '''
    df = load_user_action_cnt_with_timeDecay(start_date = start_date, end_date = end_date)
    cols = ['user_id']
    cols.extend(['Action_cnt_{0}_{1}_with_timeDecay_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[cols]
    colname = 'Action_totalCnt_{0}_{1}_with_timeDecay_{2}'.format(start_date[:10], end_date[:10], '_'.join([str(i) for i in actions]))
    df[colname] = df.drop(['user_id'], axis=1).sum(axis=1)

    return df[['user_id', colname]]


'''用户-商品特征统计'''
def load_UIPair_action_date_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户在一段时间内对某商品多少天具有操作(不可衡量用户对该商品的关注度，转化为最大值比值可衡量)
    '''
    dump_path = './cache/UIPair_action_date_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'time'])
        df.time = df.time.apply(lambda t: t[:10])
        df = df.drop_duplicates()
        colname = 'UIPair_action_date_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = df.groupby(['user_id', 'sku_id']).size().reset_index(name=colname)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df

def load_UIPair_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions= [1,2,3,4,5,6]):
    '''
    UI pair行为计数
    '''
    dump_path = './cache/UIPair_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'cate', 'type'])
        prefix = 'UIPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        df.drop(['type'], axis=1, inplace=True)
        df = df.groupby(['user_id', 'sku_id', 'cate'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    actions.sort()
    rt_cols = ['user_id', 'sku_id', 'cate']
    rt_cols.extend(['UIPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_UIPair_action_cnt_withDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计UIPair操作数（带时间衰减）
    '''
    actions.sort()
    rt_cols = ['user_id', 'sku_id','cate']
    rt_cols.extend(['UIPair_action_cnt_withDecay_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    dump_path = './cache/UIPair_action_cnt_withDecay_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'type', 'cate', 'time', 'sku_id'])
        prefix = 'UIPair_action_cnt_withDecay_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        end_date = pd.to_datetime(end_date)
        df = pd.concat([df, type_dummies], axis=1)
        df['time'] = pd.to_datetime(df.time)
        df['time_dist'] = df.time.apply(lambda t: int((end_date-t).total_seconds()/60/60)+1)
        decay = pd.read_csv(TIMEDECAY)
        df = pd.merge(df, decay, on=['time_dist'], how='left')
        for i in [1,2,3,4,5,6]:
            df[prefix+'_{0}'.format(i)] = df['weight'] * df[prefix+'_{0}'.format(i)]
        
        drop_cols = ['type', 'time', 'weight', 'time_dist']
        df.drop(drop_cols, axis=1, inplace=True)

        df = df.groupby(['user_id','sku_id', 'cate'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    df = df[rt_cols]

    return df



def load_UIPair_action_ratio_with_timeWindow(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                                        sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00', cate = [8]):
    '''
    统计UIPair在某个时间窗口的行为比例(可体现用户的购买欲望?)
    '''
    dump_path = './cache/UIPair_action_ratio_with_timeWindow_{0}_{1}.pkl'.format(sub_start_date[:10], sub_end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UIPair_action_cnt(start_date = start_date, end_date = end_date)
        sub_df = load_UIPair_action_cnt(start_date = sub_start_date, end_date = sub_end_date)
        df = pd.merge(sub_df, df, on=['user_id', 'sku_id', 'cate'], how='left')
        # df.fillna(0, inplace=True)
        prefixs_1 = ['UIPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UIPair_action_cnt_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        prefixs = ['UIPair_action_ratio_with_timeWindow_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        for i in range(6):
            df[prefixs[i]] = df[prefixs_2[i]] / df[prefixs_1[i]]
        save_cols = ['user_id', 'sku_id']
        save_cols.extend(prefixs)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    df.fillna(1, inplace=True)
    df.replace(np.inf, 1, inplace=True)

    return df


def load_UIPair_action_totalCnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户对某个商品的总操作数
    '''
    dump_path = './cache/UIPair_action_totalCnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UIPair_action_cnt(start_date = start_date, end_date = end_date)
        col = 'UIPair_action_totalCnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        temp = df.drop(['user_id', 'sku_id', 'cate'], axis=1).sum(axis=1).to_frame(name=col)
        df = pd.concat([df[['user_id', 'sku_id', 'cate']], temp], axis=1)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df


def load_UIPair_action_ratio(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions = [1,2,3,4,5,6]):
    '''
    UI pair行为比例 
    '''
    dump_path = './cache/UIPair_action_ratio_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df_u_cnt = load_UCPair_action_cnt(start_date = start_date, end_date = end_date)
        df = load_UIPair_action_cnt(start_date = start_date, end_date = end_date)
        prefix_1 = 'UCPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        prefix_2 = 'UIPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = pd.merge(df, df_u_cnt, on=['user_id', 'cate'], how='left')
        prefix_3 = 'UIPair_action_ratio_{0}_{1}'.format(start_date[:10], end_date[:10])

        df[prefix_3 + '_1'] = df[prefix_2 + '_1'] / df[prefix_1 + '_1']
        df[prefix_3 + '_2'] = df[prefix_2 + '_2'] / df[prefix_1 + '_2']
        df[prefix_3 + '_3'] = df[prefix_2 + '_3'] / df[prefix_1 + '_3']
        df[prefix_3 + '_4'] = df[prefix_2 + '_4'] / df[prefix_1 + '_4']
        df[prefix_3 + '_5'] = df[prefix_2 + '_5'] / df[prefix_1 + '_5']
        df[prefix_3 + '_6'] = df[prefix_2 + '_6'] / df[prefix_1 + '_6']

        save_cols = ['user_id', 'sku_id']
        save_cols.extend([prefix_3 + '_' + str(i) for i in range(1,7)])

        df = df[save_cols]
        df.fillna(1, inplace=True)
        df.fillna(np.inf, inplace=True)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    actions.sort()
    rt_cols = ['user_id', 'sku_id']
    rt_cols.extend(['UIPair_action_ratio_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_UIPair_modelClick_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UIPair模块点击数
    '''
    dump_path = './cache/UIPair_modelClick_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'model_id'])
        df = df[~df.model_id.isnull()]
        df = df.drop_duplicates()
        df = df.groupby(['user_id', 'sku_id']).size().reset_index(name='num_model_cnt')
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UIPair_action_bool(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions = [1,2,3,4,5,6]):
    '''
    统计UIPair是否具有某种action交互行为
    '''
    dump_path = './cache/UIPair_action_bool_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UIPair_action_cnt(start_date = start_date, end_date = end_date)
        prefixs_1 = ['UIPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UIPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]

        for i in range(0,6):
            df[prefixs_2[i]] = (df[prefixs_1[i]] > 0).astype(int)
        
        save_cols = ['user_id', 'sku_id']
        save_cols.extend(prefixs_2)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    actions.sort()
    rt_cols = ['user_id', 'sku_id']
    rt_cols.extend(['UIPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])

    return df[rt_cols]

def load_UIPair_fisrt_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UIPair最早出现交互时间到现在的距离 (单位是秒)
    '''
    dump_path = './cache/UIPair_first_tm_dist_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'time'])
        df = df.groupby(['user_id', 'sku_id'], as_index=False).time.min()
        df.time = pd.to_datetime(df.time)
        colname = 'UIPair_first_tm_dist_{0}_{1}'.format(start_date[:10], end_date[:10])
        end_date = pd.to_datetime(end_date)
        df[colname] = df.time.apply(lambda t: (end_date - t).total_seconds())
        df = df[['user_id', 'sku_id', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UIPair_last_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UIPair最晚出现交互时间到现在的距离 (单位是秒)
    '''
    dump_path = './cache/UIPair_last_tm_dist_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'time'])
        df = df.groupby(['user_id', 'sku_id'], as_index=False).time.max()
        df.time = pd.to_datetime(df.time)
        colname = 'UIPair_last_tm_dist_{0}_{1}'.format(start_date[:10], end_date[:10])
        end_date = pd.to_datetime(end_date)
        df[colname] = df.time.apply(lambda t: (end_date - t).total_seconds())
        df = df[['user_id', 'sku_id', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UIPair_action_last_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计UIPair各行为的最晚时间距离
    '''
    dump_path = './cache/UIPair_action_last_tm_dist_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'type', 'time'])
        df.time = pd.to_datetime(df.time)
        prefix = 'UIPair_action_last_tm_dist_{0}_{1}_'.format(start_date[:10], end_date[:10])
        end_date = pd.to_datetime(end_date)
        merge_obj = []
        UIPair = df[['user_id', 'sku_id']].drop_duplicates()
        for i in range(1,7):
            temp = df[df.type==i].drop(['type'], axis=1).groupby(['user_id', 'sku_id'], as_index=False).time.max()
            temp.time = temp.time.apply(lambda t: (end_date - t).total_seconds())
            temp.columns = ['user_id', 'sku_id', prefix+str(i)]
            merge_obj.append(temp.copy())
        for obj in merge_obj:
            UIPair = pd.merge(UIPair, obj, on=['user_id', 'sku_id'], how='left')
        df = UIPair
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
        end_date = str(end_date)
    rt_cols = ['user_id', 'sku_id']
    rt_cols.extend(['UIPair_action_last_tm_dist_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df 
    

def load_UCPair_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计UCPair操作数
    '''
    dump_path = './cache/UCPair_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'type', 'cate'])
        prefix = 'UCPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        df = df.groupby(['user_id', 'cate'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    actions.sort()
    rt_cols = ['user_id', 'cate']
    rt_cols.extend(['UCPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_UCPair_action_cnt_withDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计UCPair操作数（带时间衰减）
    '''
    actions.sort()
    rt_cols = ['user_id', 'cate']
    rt_cols.extend(['UCPair_action_cnt_withDecay_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    dump_path = './cache/UCPair_action_cnt_withDecay_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'type', 'cate', 'time'])
        prefix = 'UCPair_action_cnt_withDecay_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        end_date = pd.to_datetime(end_date)
        df = pd.concat([df, type_dummies], axis=1)
        df['time'] = pd.to_datetime(df.time)
        df['time_dist'] = df.time.apply(lambda t: int((end_date-t).total_seconds()/60/60)+1)
        decay = pd.read_csv(TIMEDECAY)
        df = pd.merge(df, decay, on=['time_dist'], how='left')
        for i in [1,2,3,4,5,6]:
            df[prefix+'_{0}'.format(i)] = df['weight'] * df[prefix+'_{0}'.format(i)]
        
        drop_cols = ['type', 'time', 'weight', 'time_dist']
        df.drop(drop_cols, axis=1, inplace=True)

        df = df.groupby(['user_id', 'cate'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    df = df[rt_cols]

    return df

def load_UCPair_action_totalCnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UCPair在某段时间的总操作数
    '''
    dump_path = './cache/UCPair_action_totalCnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UCPair_action_cnt(start_date = start_date, end_date = end_date)
        colname = 'UCPair_action_totalCnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df[colname] = df.drop(['user_id', 'cate'], axis=1).sum(axis=1)
        df = df[['user_id', 'cate', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UCPair_action_totalCnt_withDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UCPair在某段时间的总操作数(带时间衰减)
    '''
    dump_path = './cache/UCPair_action_totalCnt_withDecay_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UCPair_action_cnt_withDecay(start_date = start_date, end_date = end_date)
        colname = 'UCPair_action_totalcnt_withDecay_{0}_{1}'.format(start_date[:10], end_date[:10])
        df[colname] = df.drop(['user_id', 'cate'], axis=1).sum(axis=1)
        df = df[['user_id', 'cate', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UIPair_action_totalCnt_withDecay(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计UIPair在某段时间的总操作数(带时间衰减)
    '''
    dump_path = './cache/UIPair_action_totalCnt_withDecay_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UIPair_action_cnt_withDecay(start_date = start_date, end_date = end_date)
        colname = 'UIPair_action_totalcnt_withDecay_{0}_{1}'.format(start_date[:10], end_date[:10])
        df[colname] = df.drop(['user_id', 'cate', 'sku_id'], axis=1).sum(axis=1)
        df = df[['user_id', 'sku_id', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df
    

def load_UCPair_action_ratio_with_timeWindow(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                                        sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00', cate = [8]):
    '''
    统计UCPair在某个时间窗口的行为比例(可体现用户的购买欲望)
    '''
    dump_path = './cache/UCPair_action_ratio_with_timeWindow_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UCPair_action_cnt(start_date = start_date, end_date = end_date)
        sub_df = load_UCPair_action_cnt(start_date = sub_start_date, end_date = sub_end_date)
        df = pd.merge(sub_df, df, on=['user_id', 'cate'], how='left')
        # df.fillna(0, inplace=True)
        prefixs_1 = ['UCPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UCPair_action_cnt_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        prefixs = ['UCPair_action_ratio_with_timeWindow_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        for i in range(6):
            df[prefixs[i]] = df[prefixs_2[i]] / df[prefixs_1[i]]
        save_cols = ['user_id', 'cate']
        save_cols.extend(prefixs)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    df.fillna(1, inplace=True)
    df.replace(np.inf, 1, inplace=True)
    df = df[df.cate.isin(cate)]

    return df

def load_UCPair_allAction_ratio_with_timeWindow(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                                        sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00', cate = [8]):
    '''
    统计UCPair在某个时间窗口的行为占所有行为比例(可体现用户的购买欲望)
    '''
    dump_path = './cache/UCPair_allAction_ratio_with_timeWindow_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_user_action_cnt(start_date = start_date, end_date = end_date)
        sub_df = load_UCPair_action_cnt(start_date = sub_start_date, end_date = sub_end_date)
        df = pd.merge(sub_df, df, on=['user_id'], how='left')
        # df.fillna(0, inplace=True)
        prefixs_1 = ['Action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UCPair_action_cnt_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        prefixs = ['UCPair_action_ratio_with_timeWindow_{0}_{1}_{2}'.format(sub_start_date[:10], sub_end_date[:10], i) for i in range(1,7)]
        for i in range(6):
            df[prefixs[i]] = df[prefixs_2[i]] / df[prefixs_1[i]]
        save_cols = ['user_id', 'cate']
        save_cols.extend(prefixs)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    # df.fillna(1, inplace=True)
    # df.replace(np.inf, 1, inplace=True)
    df = df[df.cate.isin(cate)]

    return df



def load_UCPair_action_Ncnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计某个时段UCPair的数目(例如浏览A类商品多少个)
    '''
    dump_path = './cache/UCPair_action_Ncnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id', 'cate', 'type'])
        df = df.drop_duplicates()
        cols = ['UCPair_action_Ncnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        rtdf = df[['user_id', 'cate']].drop_duplicates()
        dfs = [df[df.type==i].groupby(['user_id', 'cate'], as_index=False).size().reset_index(name=cols[i-1]) for i in range(1,7)]
        for df in dfs:
            rtdf = pd.merge(rtdf, df, on=['user_id', 'cate'], how='left')
        rtdf.fillna(0, inplace=True)
        df = rtdf
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    actions.sort()
    rt_cols = ['user_id', 'cate']
    rt_cols.extend(['UCPair_action_Ncnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]
    
    return df

def load_base_item_feat(end_date = '2016/4/16'):
    '''
    获取评论数据
    '''
    JComment = pd.read_csv(COMMENT_FILE, encoding='gbk')
    end_date = pd.to_datetime(end_date)
    JComment.dt = pd.to_datetime(JComment.dt)
    dts = JComment.dt.drop_duplicates()
    dts.sort_index(inplace=True, ascending=False)
    for dt in dts.iteritems():
        if dt[-1] < end_date:
            break
    JComment = JComment[JComment.dt == dt[-1]].drop(['dt'], axis=1)
    Comment_num_dummies = pd.get_dummies(JComment.comment_num, prefix='Comment_num')
    JComment = pd.concat([JComment, Comment_num_dummies], axis=1)

    return JComment.drop(['comment_num'], axis=1)


        
def load_UCPair_action_bool(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions = [1,2,3,4,5,6]):
    '''
    统计UCPair是否具有某种action交互行为
    '''
    dump_path = './cache/UCPair_action_bool_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UCPair_action_cnt(start_date = start_date, end_date = end_date)
        prefixs_1 = ['UCPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UCPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]

        for i in range(0,6):
            df[prefixs_2[i]] = (df[prefixs_1[i]] > 0).astype(int)
        
        save_cols = ['user_id', 'cate']
        save_cols.extend(prefixs_2)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    actions.sort()
    rt_cols = ['user_id', 'cate']
    rt_cols.extend(['UCPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])

    return df[rt_cols]

def load_UCPair_action_date_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate = [8]):
    '''
    统计用户在一段时间内对某类商品多少天具有操作(可衡量用户对类该商品的关注度，转化为最大值比值可衡量)
    '''
    dump_path = './cache/UCPair_action_date_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'cate', 'time'])
        df.time = df.time.apply(lambda t: t[:10])
        df = df.drop_duplicates()
        colname = 'UCPair_action_date_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = df.groupby(['user_id', 'cate']).size().reset_index(name=colname)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    df = df[df.cate.isin(cate)]
            
    return df


'''商品基础特征统计'''
def load_item_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    商品行为计数
    '''
    dump_path = './cache/item_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['sku_id', 'type'])
        prefix = 'item_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df['type'], prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        df.drop(['type'], axis=1, inplace=True)
        df = df.groupby(['sku_id'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    rt_cols = ['sku_id']
    rt_cols.extend(['item_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_item_people_flow_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    商品净流量
    '''
    dump_path = './cache/item_people_flow_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id'])
        df = df.drop_duplicates()
        colname = 'item_people_flow_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = df.groupby(['sku_id']).size().reset_index(name=colname)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df


'''用户品牌特征提取'''
def load_UBPair_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', actions = [1,2,3,4,5,6]):
    '''
    用户品牌行为计数
    '''
    dump_path = './cache/UBPair_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'brand', 'type'])
        prefix = 'UBPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df.type, prefix=prefix)
        df = pd.concat([df, type_dummies], axis=1)
        df.drop(['type'], axis=1, inplace=True)
        df = df.groupby(['user_id', 'brand'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    rt_cols = ['user_id', 'brand']
    rt_cols.extend(['UBPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_UBPair_action_bool(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    统计用户对某个品牌是否有过某种行为
    '''
    dump_path = './cache/UBPair_action_bool_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UBPair_action_cnt(start_date = start_date, end_date = end_date)
        prefixs_1 = ['UBPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        prefixs_2 = ['UBPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in range(1,7)]
        for i in range(0,6):
            df[prefixs_2] = (df[prefixs_1] > 0).astype(int)
        save_cols = ['user_id', 'brand']
        save_cols.extend(prefixs_2)
        df = df[save_cols]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    rt_cols = ['user_id', 'brand']
    rt_cols.extend(['UBPair_action_bool_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df


def load_UBPair_action_totalCnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00'):
    '''
    统计一段时间用户对该品牌的所有行为计数
    '''
    dump_path = './cache/UBPair_action_totalCnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_UBPair_action_cnt(start_date = start_date, end_date = end_date)
        colname = 'UBPair_action_totalCnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df[colname] = df.drop(['user_id', 'brand'], axis=1).sum(axis=1)
        df = df[['user_id', 'brand', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

'''品牌特征提取'''
def load_brand_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    品牌行为计数
    '''
    dump_path = './cache/brand_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['brand', 'type'])
        prefix = 'brand_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df.type, prefix=prefix)
        df.drop(['type'], axis=1, inplace=True)
        df = pd.concat([df, type_dummies], axis=1)
        df = df.groupby(['brand'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    rt_cols = ['brand']
    rt_cols.extend(['brand_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_brand_people_flow_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00'):
    '''
    品牌净流量
    '''
    dump_path = './cache/brand_people_flow_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field = ['brand', 'user_id'])
        df = df.drop_duplicates()
        colname = 'brand_people_flow_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        df = df.groupby(['brand']).size().reset_index(name=colname)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_brand_comment_ratio(end_date = '2016-04-01 00:00:00'):
    '''
    品牌差评率
    '''
    dump_path = './cache/brand_comment_ratio_{0}.pkl'.format(end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        item_feat = load_base_item_feat(end_date = end_date)
        item_feat = item_feat[['sku_id', 'bad_comment_rate']]
        brands = get_action_data(start_date = '2016-02-01 00:00:00', end_date = end_date, field=['sku_id', 'brand'])
        brands = brands.drop_duplicates()
        df = pd.merge(item_feat, brands, on=['sku_id'], how='left')
        df = df[['brand', 'bad_comment_rate']]
        df = df.groupby(['brand'], as_index=False).mean()
        df.columns = ['brand', 'brand_bad_comment_rate']
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df


def load_ICB_tb(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    获取item cate brand
    '''
    dump_path = './cache/ICB_tb_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['sku_id', 'cate', 'brand'])
        df = df.drop_duplicates()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df

def load_BCPair_action_cnt(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', actions=[1,2,3,4,5,6]):
    '''
    获取品牌-类别行为计数
    '''
    dump_path = './cache/BCPair_action_cnt_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['brand', 'cate', 'type'])
        prefix = 'BCPair_action_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
        type_dummies = pd.get_dummies(df.type, prefix=prefix)
        df = pd.concat([df.drop(['type'], axis=1), type_dummies], axis=1)
        df = df.groupby(['brand', 'cate'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    rt_cols = ['brand', 'cate']
    rt_cols.extend(['BCPair_action_cnt_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], i) for i in actions])
    df = df[rt_cols]

    return df

def load_user_login_matrix(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    获取用户登录矩阵
    '''
    dump_path = './cache/user_login_matrix_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field = ['user_id', 'time'])
        df.time = df.time.apply(lambda t: t[:10])
        df = df.drop_duplicates()
        time_dummies = pd.get_dummies(df.time, prefix='login')
        df = pd.concat([df, time_dummies], axis=1)
        df.drop(['time'], axis=1, inplace=True)
        df = df.groupby(['user_id'], as_index=False).sum()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df

# def load_filter_uid(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
#     '''
#     用户过滤 (只登陆一天并且只在登陆那天购买)
#     '''
#     dump_path = 'filter_uid_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
#     if os.path.exists(dump_path):
#         with open(dump_path, 'rb') as f:
#             df = pickle.load(f)
#     else:
#         df_date_cnt = load_user_action_date_cnt(start_date = start_date, end_date = end_date)
#         df_date_cnt.columns = ['user_id', 'login_cnt']
#         df_date_cnt = df_date_cnt[df_date_cnt.login_cnt==1]
#         df_user_buy = load_user_action_cnt(start_date = start_date, end_date = end_date)
#         prefix = 'Action_cnt_{0}_{1}_4'.format(start_date[:10], end_date[:10])
#         df_user_buy = df_user_buy[['user_id', prefix]]
#         df_user_buy = df_user_buy[df_user_buy[prefix]>0]
#         drop_uid = df_date_cnt[df_date_cnt.user_id.isin(df_user_buy.user_id)].user_id
#         df = drop_uid
#         with open(dump_path, 'wb') as f:
#             pickle.dump(df, f)
    
#     return df

def load_UCPair_last_tm_dist(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate = [8]):
    '''
    获取用户与某类商品最后一次交互时间距离
    '''
    dump_path = "./cache/UCPair_last_tm_dist_{0}_{1}.pkl".format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date ,field=['user_id', 'time',  'cate',])
        df = df.groupby(['user_id', 'cate'], as_index=False).time.max()
        df.time = pd.to_datetime(df.time)
        colname = 'last_tm_dist_{0}_{1}'.format(start_date[:10], end_date[:10])
        end_date = pd.to_datetime(end_date)
        df[colname] = df.time.apply(lambda t: (end_date - t).total_seconds())
        df = df[['user_id', 'cate', colname]]
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    df = df[df.cate.isin(cate)]

    return df

def load_filter_uid(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户单次访问最大值
    '''
    dump_path = './cache/user_click_cnt_{0}_{1}'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'time'])
        df = df.groupby(['user_id']).time.value_counts()
        df = df.reset_index(name='cnt')
        df = df.groupby('user_id', as_index=False).cnt.max()

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_user_click_freq(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户单次访问均值
    '''
    dump_path = './cache/user_click_freq_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date=start_date, end_date=end_date, field=['user_id', 'time'])
        df = df.groupby(['user_id']).time.value_counts()
        df = df.reset_index(name='cnt')
        df = df.groupby('user_id', as_index=False).cnt.mean()

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_user_act_cnt_with_timeZone(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    获取用户各时间窗行为统计
    '''
    dump_path = './cache/user_act_cnt_with_timeZone_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time_zone'])
        timeZone_dummies = pd.get_dummies(df.time_zone, prefix='time_zone_cnt')
        df = pd.concat([df, timeZone_dummies], axis=1)
        df.drop(['time_zone'], axis=1, inplace=True)
        df = df.groupby(['user_id'], as_index=False).sum()

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UCPair_act_cnt_with_timeZone(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate=[8]):
    '''
    获取用户与某类商品各时间窗行为统计
    '''
    dump_path = './cache/UCPair_act_cnt_with_timeZone_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time_zone', 'cate'])
        timeZone_dummies = pd.get_dummies(df.time_zone, prefix='uc_time_zone_cnt')
        df = pd.concat([df, timeZone_dummies], axis=1)
        df.drop(['time_zone'], axis=1, inplace=True)
        df = df.groupby(['user_id', 'cate'], as_index=False).sum()

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    df = df[df.cate.isin(cate)]
    return df

def load_UIPair_act_cnt_with_timeZone(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', cate=[8]):
    '''
    获取用户与某商品各时间窗行为统计
    '''
    dump_path = './cache/UIPair_act_cnt_with_timeZone_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time_zone', 'sku_id'])
        timeZone_dummies = pd.get_dummies(df.time_zone, prefix='time_zone_cnt')
        df = pd.concat([df, timeZone_dummies], axis=1)
        df.drop(['time_zone'], axis=1, inplace=True)
        df = df.groupby(['user_id', 'sku_id'], as_index=False).sum()

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
            
    return df

'''
统计用户有效交互时间
'''
VALIDTIME_THRESHOLD = 180

def gen_up_validtime(actions):
    up_validtime = actions.groupby(['user_id'], sort=False)['time'].agg(compute_validtime).reset_index()
    up_validtime.columns = ['user_id', 'valid_time']
    return up_validtime

def gen_up_validtime_with_UI(actions):
    up_validtime = actions.groupby(['user_id', 'sku_id'], sort=False)['time'].agg(compute_validtime).reset_index()
    up_validtime.columns = ['user_id', 'sku_id','valid_time_ui']
    return up_validtime

def gen_up_validtime_with_UC(actions):
    up_validtime = actions.groupby(['user_id', 'cate'], sort=False)['time'].agg(compute_validtime).reset_index()
    up_validtime.columns = ['user_id', 'cate', 'valid_time_uc']
    return up_validtime

def compute_validtime(time):
    time = list(time)
    time.sort()
    temp_time = time[1:]
    temp_time.append(max(time))
    f = lambda x, y: second_diff(x, y)
    valid_time_list = map(f, time, temp_time)
    valid_time = sum([time for time in valid_time_list if time <= VALIDTIME_THRESHOLD]) + 1
    return valid_time

def second_diff(start, end):
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    days = (end - start).days
    seconds = (end - start).seconds
    diff = days * 24 * 60 * 60 + seconds
    return diff

def load_user_act_time(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计用户的有效行为时间
    '''
    dump_path = './cache/user_act_time_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'time'])
        df = df.drop_duplicates()
        df = gen_up_validtime(df)

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UI_act_time(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计ui的有效行为时间
    '''
    dump_path = './cache/ui_act_time_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'sku_id','time'])
        df = df.drop_duplicates()
        df = gen_up_validtime_with_UI(df)

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def load_UC_act_time(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00'):
    '''
    统计uc的有效行为时间
    '''
    dump_path = './cache/uc_act_time_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'cate','time'])
        df = df.drop_duplicates()
        df = gen_up_validtime_with_UC(df)

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df


'''
计算度 
'''
def gen_indegree(actions):
    actions = actions.sort(['user_id', 'time']).reset_index(drop=True)
    actions = actions[['user_id', 'sku_id']]
    user = list(actions.user_id)[1:]
    user.append(999999999)
    sku = list(actions.sku_id)[1:]
    sku.append(999999999)
    actions['user'] = user
    actions['sku'] = sku
    maps = actions[(actions.sku_id != actions.sku) | (actions.user_id != actions.user)]
    indegree = maps.groupby(['user_id', 'sku_id'], sort=False).size().reset_index()
    indegree.columns = ['user_id', 'sku_id', 'indegree']
    return indegree


# ###LDA
# import lda

# from sklearn.feature_extraction.text import CountVectorizer
# # df        action表
# # prefix    特征名称前缀
# # entity1   实体1名称，看作文档
# # entity2   实体2名称，看作文档中的单词
# # n_topics  提取主题个数
# # n_iter    LDA模型迭代次数

# def feature_lda( df, prefix, entity1, entity2, n_topics, n_iter ):

#     df = df[ [entity1, entity2] ]


#     df[entity2] = df[entity2].astype('str')
#     df[entity2] = df[entity2] + ' '
#     temp = df.groupby( entity1 ).sum().reset_index()


#     vc = CountVectorizer()
#     vc.fit( temp[entity2] )
#     X = vc.transform( temp[entity2] ) 


#     model = lda.LDA( n_topics=n_topics, n_iter=n_iter, random_state=0 )
#     model.fit(X)


#     result = pd.DataFrame( model.doc_topic_ ).add_prefix( prefix )
#     result[entity1] = temp[entity1]

#     return result

def load_UCPair_last_act_tm(start_date = '2016-02-01 00:00:00', end_date = '2016-04-11 00:00:00', actions=2, cate=[8]):
    '''
    获取uc某行为最晚时间距离
    '''
    dump_path = './cache/UCPair_last_act_tm_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], actions)
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'cate', 'type', 'time'])
        df = df[df.type==actions]
        df = df.drop_duplicates()
        df = df.groupby(['user_id', 'cate'], as_index=False).time.max()
        df.time = pd.to_datetime(df.time)
        end_date = pd.to_datetime(end_date)
        df.time = df.time.apply(lambda t: (end_date - t).total_seconds())

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    df = df[df.cate.isin(cate)]

    return df

def load_UCPair_second_act_tm(start_date = '2016-02-01 00:00:00', end_date = '2016-04-11 00:00:00', actions=2, cate=[8]):
    '''
    获取uc某行为倒数第二晚时间距离
    '''
    dump_path = './cache/UCPair_second_act_tm_{0}_{1}_{2}'.format(start_date[:10], end_date[:10], actions)
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = start_date, end_date = end_date, field=['user_id', 'cate', 'type', 'time'])
        df = df[df.type==actions]
        df = df.drop_duplicates()
        temp = df.groupby(['user_id', 'cate'], as_index=False).time.max()
        temp['flag'] = 1
        df = pd.merge(df, temp, on=['user_id', 'cate', 'time'], how='left')
        df = df[df.flag.isnull()]
        df = df[['user_id', 'cate', 'time']]
        df = df.groupby(['user_id', 'cate'], as_index=False).time.max()
        df.time = pd.to_datetime(df.time)
        end_date = pd.to_datetime(end_date)
        df.time = df.time.apply(lambda t: (end_date - t).total_seconds())

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    df = df[df.cate.isin(cate)]

    return df


