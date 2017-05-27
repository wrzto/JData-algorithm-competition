#-*-coding:utf-8-*-

from features_generator import *
import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta

# UNTRAIN_COLUMNS = ['user_id', 'sku_id', 'cate', 'brand', 'label']


def compute_str_time(str_t, delta, add = False):
    t = pd.to_datetime(str_t)
    if add:
        return str(t + timedelta(delta))
    else:
        return str(t - timedelta(delta))

def load_UIPair_to_model(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', \
                            sub_start_date = '2016-04-01 00:00:00', sub_end_date = '2016-04-06 00:00:00', offline=True, update = False, day=7):
    '''
    提取UI pair
    '''
    if offline:
        dump_path = './cache/UIPair_to_model_offline_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    else:
        dump_path = './cache/UIPair_to_model_online_{0}_{1}.pkl'.format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path) and update == False:
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = get_action_data(start_date = compute_str_time(end_date, day), end_date = end_date, field=['user_id', 'sku_id', 'cate', 'brand','type'])
        df = df[df.cate==8]
        drop_uid = df[df.type==4].user_id
        df = df[~df.user_id.isin(drop_uid)]
        df.drop(['type'], axis=1, inplace=True)
        df = df.drop_duplicates()
        if offline:
            _, label_df =  get_sub_uid_with_sid_label(start_date = sub_start_date, end_date = sub_end_date)
            label_df['label'] = 1
            df = pd.merge(df, label_df, on=['user_id', 'sku_id'], how='left')
            df.fillna(0, inplace=True)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df

def load_all_UIPair_feat(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', df=None):

    merge_obj = []

    #时间特征
    #用户与该商品最后交互时间
    ui_last_tm_dist = load_UIPair_last_tm_dist(start_date = start_date, end_date = end_date)
    ui_last_tm_dist.columns = ['user_id', 'sku_id', 'ui_last_tm_dist']

    ui_first_tm_dist = load_UIPair_fisrt_tm_dist(start_date = start_date, end_date = end_date)
    ui_first_tm_dist.columns = ['user_id', 'sku_id', 'ui_first_tm_dist']

    temp = pd.merge(ui_first_tm_dist, ui_last_tm_dist, on=['user_id', 'sku_id'], how='left')
    temp['ui_diff_tm_dist'] = temp['ui_first_tm_dist'] - temp['ui_last_tm_dist']
    merge_obj.append(temp)

    #用户前7/15/28/60对该商品的交互天数
    ui_date_cnt_7day = load_UIPair_action_date_cnt(start_date = compute_str_time(end_date, 7), end_date = end_date)
    ui_date_cnt_7day.columns = ['user_id', 'sku_id', 'ui_date_cnt_7day']

    ui_date_cnt_15day = load_UIPair_action_date_cnt(start_date = compute_str_time(end_date, 15), end_date = end_date)
    ui_date_cnt_15day.columns = ['user_id', 'sku_id', 'ui_date_cnt_15day']

    ui_date_cnt_28day = load_UIPair_action_date_cnt(start_date = compute_str_time(end_date, 28), end_date = end_date)
    ui_date_cnt_28day.columns = ['user_id', 'sku_id', 'ui_date_cnt_28day']

    ui_date_cnt = load_UIPair_action_date_cnt(start_date = start_date, end_date = end_date)
    ui_date_cnt.columns = ['user_id', 'sku_id', 'ui_date_cnt']

    #用户登录天数
    user_date_cnt_7day = load_user_action_date_cnt(start_date=compute_str_time(end_date, 7), end_date=end_date)
    user_date_cnt_7day.columns =['user_id', 'user_date_cnt_7day']

    user_date_cnt_15day = load_user_action_date_cnt(start_date=compute_str_time(end_date, 15), end_date=end_date)
    user_date_cnt_15day.columns =['user_id', 'user_date_cnt_15day']

    user_date_cnt_28day = load_user_action_date_cnt(start_date=compute_str_time(end_date, 28), end_date=end_date)
    user_date_cnt_28day.columns =['user_id', 'user_date_cnt_28day']
    
    user_date_cnt = load_user_action_date_cnt(start_date=start_date, end_date=end_date)
    user_date_cnt.columns =['user_id', 'user_date_cnt']

    temp = pd.merge(ui_date_cnt, ui_date_cnt_15day, on=['user_id', 'sku_id'], how='left')
    temp = pd.merge(temp, ui_date_cnt_7day, on=['user_id', 'sku_id'], how='left')
    temp = pd.merge(temp, ui_date_cnt_28day, on=['user_id', 'sku_id'], how='left')

    temp = pd.merge(temp, user_date_cnt, on=['user_id'], how='left')
    temp = pd.merge(temp, user_date_cnt_7day, on=['user_id'], how='left')
    temp = pd.merge(temp, user_date_cnt_15day, on=['user_id'], how='left')
    temp = pd.merge(temp, user_date_cnt_28day, on=['user_id'], how='left')

    temp.fillna(0, inplace=True)
    temp['ui_user_ratio_7day'] = temp['ui_date_cnt_7day'] / temp['user_date_cnt_7day'].replace(0,1)
    temp['ui_user_ratio_15day'] = temp['ui_date_cnt_15day'] / temp['user_date_cnt_15day'].replace(0,1)
    temp['ui_user_ratio_28day'] = temp['ui_date_cnt_28day'] / temp['user_date_cnt_28day'].replace(0,1)
    temp['ui_user_ratio_60day'] = temp['ui_date_cnt'] / temp['user_date_cnt'].replace(0,1)

    drop_cols = ['user_date_cnt_7day', 'user_date_cnt_15day', 'user_date_cnt', 'user_date_cnt_28day']
    merge_obj.append(temp.drop(drop_cols, axis=1))

    #用户行为特征
    #用户对该商品的行为统计(时间衰减)
    ui_act_cnt_decay = load_UIPair_action_totalCnt_withDecay(start_date = compute_str_time(end_date, 28), end_date=end_date)
    ui_act_cnt_decay.columns = ['user_id', 'sku_id', 'ui_act_cnt_decay']
    merge_obj.append(ui_act_cnt_decay)

    #用户是否(2,3,5)
    ui_act_bool = load_UIPair_action_bool(start_date = start_date, end_date=end_date, actions = [2,3,5])
    ui_act_bool.columns = ['user_id', 'sku_id', 'ui_act_bool2', 'ui_act_bool3', 'ui_act_bool5']
    merge_obj.append(ui_act_bool)

    #前(4h,8h,16h,24h,2,3,5,7,15,28)
    for delta in [4,8,16,24,2*24,3*24,5*24,7*24,15*24,28*24]:
        temp = load_UIPair_action_cnt(start_date=compute_str_time(end_date, delta/24.0), end_date=end_date, actions= [1,2,3,5,6])       
        cols = ['user_id', 'sku_id', 'cate']
        cols.extend(['ui_act_cnt_{0}_{1}hour'.format(i, delta) for i in [1,2,3,5,6]])
        temp.columns = cols
        temp = temp[temp.cate==8]

        merge_obj.append(temp.drop(['cate'], axis=1))

    #用户前1/2/3/5/7/15/28的行为数/总行为数目
    for delta in [1,2,3,5,7,15,28]:
        temp1 = load_UCPair_action_totalCnt(start_date=compute_str_time(end_date, delta), end_date=end_date)
        temp1.columns = ['user_id', 'cate', 'uc_total_cnt_{0}day'.format(delta)]
        temp1 = temp1[temp1.cate==8]

        temp2 = load_UIPair_action_totalCnt(start_date=compute_str_time(end_date, delta), end_date=end_date)
        temp2.columns = ['user_id', 'sku_id', 'cate', 'ui_total_cnt_{0}day'.format(delta)]
        temp2 = temp2[temp2.cate==8]

        temp3 = load_UIPair_action_date_cnt(start_date=compute_str_time(end_date, delta), end_date=end_date)
        temp3.columns = ['user_id', 'sku_id', 'ui_act_tm_{0}day'.format(delta)]

        temp = pd.merge(temp1, temp2, on=['user_id', 'cate'], how='left')
        temp = pd.merge(temp, temp3, on=['user_id', 'sku_id'], how='left')
        temp.fillna(0, inplace=True)
        temp['ui_uc_ratio_{0}day'.format(delta)] = temp['ui_total_cnt_{0}day'.format(delta)] / temp['uc_total_cnt_{0}day'.format(delta)].replace(0, 1)
        temp['mean_ui_act_{0}day'.format(delta)] = temp['ui_total_cnt_{0}day'.format(delta)] / temp['ui_act_tm_{0}day'.format(delta)].replace(0, 1)

        merge_obj.append(temp.drop(['cate', 'ui_act_tm_{0}day'.format(delta)], axis=1))
    
    #用户有效行为时间
    for delta in [1,2,3,5,7]:
        temp1 = load_UI_act_time(start_date=compute_str_time(end_date, delta), end_date = end_date)
        temp1.columns = ['user_id', 'sku_id', 'ui_act_tm_{0}day'.format(delta)]
        temp2 = load_UC_act_time(start_date=compute_str_time(end_date, delta), end_date=end_date)
        temp2.columns = ['user_id', 'cate', 'uc_act_tm_{0}day'.format(delta)]
        temp2 = temp2[temp2.cate==8]
        temp = pd.merge(temp1, temp2, on=['user_id'], how='left')
        temp.fillna(0, inplace=True)

        temp['ui_uc_tm_{0}day'.format(delta)] = temp['ui_act_tm_{0}day'.format(delta)] / temp['uc_act_tm_{0}day'.format(delta)].replace(0,1)
        
        merge_obj.append(temp.drop(['cate'], axis=1))
    
    #点击/浏览
    temp = load_UIPair_action_cnt(start_date = compute_str_time(end_date, 7), end_date = end_date, actions= [1,6])
    temp.columns = ['user_id', 'sku_id', 'cate', 'act_1_cnt', 'act_6_cnt']
    temp['ui_ratio_1_6'] = temp['act_1_cnt'] / temp['act_6_cnt'].replace(0,1)

    merge_obj.append(temp[['user_id', 'sku_id', 'ui_ratio_1_6']])

    #度
    for delta in [4,8,16,24,2*24,3*24,5*24,7*24]:
        data = get_action_data(start_date=compute_str_time(end_date, delta/24.0), end_date = end_date, field=['user_id', 'sku_id', 'cate', 'time'])
        data = data[data.cate==8]
        temp = gen_indegree(data[['user_id', 'sku_id', 'time']])
        temp.columns = ['user_id', 'sku_id', 'indegree_{0}hour'.format(delta)]

        merge_obj.append(temp)
    
    #用户最后一次加购物车/关注/删除时间
    ui_act_last_tm = load_UIPair_action_last_tm_dist(start_date = start_date, end_date = end_date , actions=[2,3,5])
    ui_act_last_tm.columns = ['user_id', 'sku_id', 'act_last_tm_2', 'act_last_tm_3', 'act_last_tm_5']
    ui_act_last_tm.fillna(0, inplace=True)
    ui_act_last_tm['delcart_later'] = (ui_act_last_tm['act_last_tm_2'] > ui_act_last_tm['act_last_tm_3']).astype(int)
    merge_obj.append(ui_act_last_tm)

    ##用户与该商品的行为计数(带时间衰减)
    UIPair_action_cnt_withdecay = load_UIPair_action_cnt_withDecay(start_date = compute_str_time(end_date, 28), end_date = end_date, actions=[1,2,3,5,6])
    UIPair_action_cnt_withdecay.columns = ['user_id', 'sku_id', 'cate', 'act1_withdecay', 'act2_withdecay', 'act3_withdecay', 'act5_withdecay', 'act6_withdecay']
    UIPair_action_cnt_withdecay.drop(['cate'], axis=1, inplace=True)
    merge_obj.append(UIPair_action_cnt_withdecay)

    ###3.用户加购该商品数目/总购物车数目 (nan填0)
    UIPair_cart_cnt = load_UIPair_action_cnt(start_date = start_date, end_date = end_date, actions= [2])
    UIPair_cart_cnt.columns = ['user_id', 'sku_id', 'cate', 'UIPair_cart_cnt']
    user_cart_cnt = UIPair_cart_cnt[['user_id', 'UIPair_cart_cnt']].groupby(['user_id'], as_index=False).sum()
    user_cart_cnt.columns = ['user_id', 'user_cart_cnt']
    UIPair_cart_cnt = pd.merge(UIPair_cart_cnt, user_cart_cnt, on=['user_id'], how='left')
    UIPair_cart_cnt['cart_ratio_UIPair_user'] = UIPair_cart_cnt['UIPair_cart_cnt'] / UIPair_cart_cnt['user_cart_cnt'].replace(0, 1)
    UIPair_cart_ratio = UIPair_cart_cnt[['user_id', 'sku_id', 'cart_ratio_UIPair_user']]
    UIPair_cart_ratio.fillna(0, inplace=True)

    merge_obj.append(UIPair_cart_ratio)

    print('开始拼接商品表 {0}'.format(df.shape))
    N_b = df.shape[0]
    for obj in merge_obj:
        df = pd.merge(df, obj, on=['user_id', 'sku_id'], how='left')
    N_e = df.shape[0]
    
    assert N_b == N_e

    return df

def load_all_item_feat(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', df = None):
    merge_obj = []

    #基础特征
    sku_info = pd.read_csv(PRODUCT_FILE, encoding='gbk')
    sku_info = sku_info[['sku_id', 'a1', 'a2', 'a3']]
    sku_info = sku_info.replace(-1,0)
    a1_dummies = pd.get_dummies(sku_info.a1, prefix='a1')
    a2_dummies = pd.get_dummies(sku_info.a2, prefix='a2')
    a3_dummies = pd.get_dummies(sku_info.a3, prefix='a3')

    sku_info = pd.concat([sku_info[['sku_id']], a1_dummies, a2_dummies, a3_dummies], axis=1)
    merge_obj.append(sku_info)

    #评论特征
    item_base_feat = load_base_item_feat(end_date = end_date)
    if 'Comment_num_0' not in item_base_feat.columns.tolist():
        item_base_feat['Comment_num_0'] = 0
    item_base_feat = item_base_feat[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'Comment_num_0', \
       'Comment_num_1', 'Comment_num_2', 'Comment_num_3', 'Comment_num_4']]
    merge_obj.append(item_base_feat)

    #top N?sale
    temp = get_action_data(start_date = compute_str_time(end_date, 7), end_date = end_date, field=['sku_id', 'type', 'cate'])
    temp = temp[(temp.cate==8)&(temp.type==4)]
    temp = temp.sku_id.value_counts()
    Top5 = (temp.iloc[:5].index).astype(int)
    df['top5'] = (df.sku_id.isin(Top5)).astype(int)
    

    #热度特征
    # ###11.前(1/2/3/5/7/10)的净流量/总净流量 (nan填0)
    total_flow_cnt = load_item_people_flow_cnt(start_date = start_date, end_date = end_date)
    total_flow_cnt.columns = ['sku_id', 'total_item_flow_cnt']
    for delta in [1,3,5,7,10,15,28]:
        temp = load_item_people_flow_cnt(start_date = compute_str_time(end_date, delta), end_date = end_date)
        temp.columns = ['sku_id', 'item_flow']
        total_flow_cnt = pd.merge(total_flow_cnt, temp, on=['sku_id'], how='left')
        total_flow_cnt.fillna(0, inplace=True)
        total_flow_cnt['item_flow_ratio_day_{0}'.format(delta)] = total_flow_cnt['item_flow'] / total_flow_cnt['total_item_flow_cnt'].replace(0, 1)
        total_flow_cnt.drop(['item_flow'], axis=1, inplace=True)
    total_flow_cnt.drop(['total_item_flow_cnt'], axis=1, inplace=True) 
    merge_obj.append(total_flow_cnt)


    print('开始拼接商品 {0}'.format(df.shape))
    N_b = df.shape[0]
    for obj in merge_obj:
        df = pd.merge(df, obj, on=['sku_id'], how='left')
    N_e = df.shape[0]

    assert N_b == N_e

    return df

def load_all_brand_feat(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', df = None):
    merge_obj = []
    #品牌差评率
    comment_ratio = load_brand_comment_ratio(end_date = end_date)
    merge_obj.append(comment_ratio)
    #topn
    temp = get_action_data(start_date=compute_str_time(end_date, 7), end_date = end_date, field=['type', 'cate', 'brand'])
    temp = temp[(temp.cate==8)&(temp.type==4)]
    temp = temp.brand.value_counts()
    Top5 = (temp.iloc[:5].index).astype(int)
    df['top5_brand'] = (df.brand.isin(Top5)).astype(int)

    print("开始拼接商品表 {0}".format(df.shape))
    N_b = df.shape[0]
    for obj in merge_obj:
        df = pd.merge(df, obj, on=['brand'], how='left')
    N_e = df.shape[0]

    assert N_b == N_e

    return df

def load_all_UBPair_feat(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', df = None):
    '''
    ---->用户品牌特征(挖掘用户对品牌的信赖程度)
    3.用户前(1/3/5/7/10/14/28/60)对该品牌的操作数/用户前(1/3/5/7/10/14/28/60)的总操作数
    '''
    merge_obj = []
    # ###3.用户前(1/3/5/7/10/14/28/60)对该品牌的操作数/用户前(1/3/5/7/10/14/28/60)的总操作数
    for delta in [1,3,5,7,10,14,28]:
        temp_1 = load_UBPair_action_totalCnt(start_date = compute_str_time(end_date, delta), end_date = end_date)
        temp_1.columns = ['user_id', 'brand', 'UBPair_act_totalCnt']
        temp_2 = load_user_action_totalCnt(start_date = compute_str_time(end_date, delta), end_date = end_date)
        temp_2.columns = ['user_id', 'user_act_totalCnt']
        temp = pd.merge(temp_1, temp_2, on=['user_id'], how='left')
        colname = 'UBC_act_ratio_{0}'.format(delta)
        temp[colname] = temp['UBPair_act_totalCnt'] / temp['user_act_totalCnt'].replace(0,1)
        temp = temp[['user_id', 'brand', colname]]
        merge_obj.append(temp.copy())
    
    print('开始拼接用户品类表. shape: {0}'.format(df.shape))
    N_b = df.shape[0]
    for obj in merge_obj:
        df = pd.merge(df, obj, on=['user_id', 'brand'], how='left')
    N_e = df.shape[0]
    print('用户品类表拼接完成. shape: {0}'.format(df.shape))
    assert N_b == N_e

    return df

def load_other_feat(start_date = '2016-02-01 00:00:00', end_date = '2016-04-01 00:00:00', df = None):
    '''
    1.该商品前(1/2/3/5/7/10/14/28/60)销量/该商品品牌同类商品(1/2/3/5/7/10/14/28/60)的销量 (nan填0)
    '''
    merge_obj = []
    ICB_tb = load_ICB_tb()
    for delta in [1,2,3,5,7,10,14,28]:
        item_buy_cnt = load_item_action_cnt(start_date = compute_str_time(end_date, delta), end_date = end_date, actions=[4])
        item_buy_cnt.columns = ['sku_id', 'item_buy_cnt']
        bc_buy_cnt = load_BCPair_action_cnt(start_date = compute_str_time(end_date, delta), end_date = end_date, actions=[4])
        bc_buy_cnt.columns = ['brand', 'cate', 'bc_buy_cnt']
        item_buy_cnt = pd.merge(item_buy_cnt, ICB_tb, on=['sku_id'], how='left')
        item_buy_cnt = pd.merge(item_buy_cnt, bc_buy_cnt, on=['brand', 'cate'], how='left')
        item_buy_cnt.fillna(0, inplace=True)
        item_buy_cnt['ICB_sale_ratio_day_{0}'.format(delta)] = item_buy_cnt['item_buy_cnt'] / item_buy_cnt['bc_buy_cnt'].replace(0,1)
        item_buy_cnt = item_buy_cnt[['sku_id', 'ICB_sale_ratio_day_{0}'.format(delta)]]
        merge_obj.append(item_buy_cnt)
    
    print('开始拼接交叉表. shape: {0}'.format(df.shape))
    N_b = df.shape[0]
    for obj in merge_obj:
        df = pd.merge(df, obj, on=['sku_id'], how='left')
    N_e = df.shape[0]
    print('交叉表拼接完成. shape: {0}'.format(df.shape))
    assert N_b == N_e

    return df.fillna(0)


def extract_feat_to_model(feat_extract_start_date = '2016-02-01 00:00:00', feat_extract_end_date = '2016-04-01 00:00:00', \
                            label_extract_start_date = '2016-04-01 00:00:00', label_extract_end_date = '2016-04-06 00:00:00', \
                            offline=True, update=False, day=7):
    dump_path = './model_feat/cate8_feature_{0}_{1}.pkl'.format(feat_extract_start_date[:10], feat_extract_end_date[:10])
    if os.path.exists(dump_path) and update == False:
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        #获取UI pair ['user_id', 'sku_id', 'cate', 'brand', 'label]
        df = load_UIPair_to_model(start_date = compute_str_time(feat_extract_end_date, day), end_date = feat_extract_end_date, \
                                    sub_start_date = label_extract_start_date, sub_end_date = label_extract_end_date, offline=offline, update=update, day=day)

        #获取所有用户-商品特征
        df = load_all_UIPair_feat(start_date = feat_extract_start_date, end_date = feat_extract_end_date, df = df)

        # #获取所有用户-品牌特征
        df = load_all_UBPair_feat(start_date = feat_extract_start_date, end_date = feat_extract_end_date, df = df)

        #获取所有商品特征
        df = load_all_item_feat(start_date = feat_extract_start_date, end_date = feat_extract_end_date, df = df)

        #获取所有品牌特征
        df = load_all_brand_feat(start_date = feat_extract_start_date, end_date = feat_extract_end_date, df = df)


        df = load_other_feat(start_date = feat_extract_start_date, end_date = feat_extract_end_date, df = df)

        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    # df.replace(np.inf, 1, inplace=True)
    return df



    
