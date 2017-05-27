import pandas as pd
import numpy as np
from features_generator import *
import pickle

'''文件路径定义'''
ACTION_FILE = './datasets/JData_Action.csv'
COMMENT_FILE = './datasets/JData_Comment.csv'
PRODUCT_FILE = './datasets/JData_Product.csv'
USER_FILE = './datasets/JData_User.csv'


'''
def load_rule_prdict_uid(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                            sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00'):
    
    # 根据规则生成预测uid   
    # rule： Action_cnt_ratio_buy==1    Action_cnt_ratio_browse==1   user_lv_cd>2

    user_act = load_action_ratio_with_timeWindow(start_date = start_date, end_date = end_date, \
                                                    sub_end_date = sub_end_date, sub_start_date = sub_start_date)
    JUser = pd.read_csv(USER_FILE, encoding='gbk')
    user_act = pd.merge(user_act, JUser, on='user_id', how='left')

    predict_uid = get_action_data(start_date = sub_start_date, end_date = sub_end_date)
    addCart_uid = predict_uid[(predict_uid.cate==8)&(predict_uid.type==2)].user_id.drop_duplicates()
    buy_uid = predict_uid[(predict_uid.cate==8)&(predict_uid.type==4)].user_id.drop_duplicates()
    explore_user = predict_uid[predict_uid.cate==8].user_id.drop_duplicates()
    
    addCart_uid = addCart_uid[~addCart_uid.isin(buy_uid)]
    user_act = user_act[user_act.user_id.isin(explore_user)]
    user_act = user_act[user_act.user_id.isin(addCart_uid)]
    ##不含sid
    uid = user_act[(user_act.Action_cnt_ratio_buy==1)&(user_act.Action_cnt_ratio_browse==1)&(user_act.user_lv_cd>2)].user_id.to_frame()

    return uid
'''
def load_rule_prdict_uid(start_date = '2016-02-01 00:00:00', end_date = '2016-04-16 00:00:00', \
                            sub_start_date = '2016-04-09 00:00:00', sub_end_date = '2016-04-16 00:00:00'):
    '''
    根据规则生成预测uid   
    rule： UCPair_action_ratio_browse==1 UCPair_action_ratio_buy==1 lv>2  date_cnt<=2(cate=8)  对商品操作数，加购？购买数
    '''
    user_act = load_UCPair_action_ratio_with_timeWindow(start_date = start_date, end_date = end_date, \
                                                    sub_end_date = sub_end_date, sub_start_date = sub_start_date)
    user_act.columns = ['user_id', 'cate', 'uc_act_ratio_browse', 'uc_act_ratio_cart', 'uc_act_ratio_delcart', \
                        'uc_act_ratio_buy', 'uc_act_ratio_favor', 'uc_act_ratio_click']
    JUser = pd.read_csv(USER_FILE, encoding='gbk')
    user_act = pd.merge(user_act, JUser, on='user_id', how='left')

    predict_uid = get_action_data(start_date = sub_start_date, end_date = sub_end_date)
    addCart_uid = predict_uid[(predict_uid.cate==8)&(predict_uid.type==2)].user_id.drop_duplicates()
    buy_uid = predict_uid[(predict_uid.cate==8)&(predict_uid.type==4)].user_id.drop_duplicates()
    explore_user = predict_uid[predict_uid.cate==8].user_id.drop_duplicates()
    
    addCart_uid = addCart_uid[~addCart_uid.isin(buy_uid)]
    user_act = user_act[user_act.user_id.isin(explore_user)]
    user_act = user_act[user_act.user_id.isin(addCart_uid)]
    ##不含sid
    user_date_cnt = load_UCPair_action_date_cnt(start_date = sub_start_date, end_date = sub_end_date)
    user_date_cnt.columns = ['user_id', 'cate', 'date_cnt']
    user_act = pd.merge(user_act, user_date_cnt, on=['user_id'], how='left')

    uid = user_act[(user_act.uc_act_ratio_buy==1)&(user_act.uc_act_ratio_browse==1)&(user_act.user_lv_cd>2)&(user_act.date_cnt<=2)].user_id.to_frame()

    return uid


    
    

