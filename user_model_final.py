#-*-coding:utf-8-*-
from features_generator import *
from tools import *
from rule import *

def load_uid_to_train(start_date="2016-02-01 00:00:00", end_date="2016-04-11 00:00:00", offline = True, update = False):
    dump_path = "./cache/uid_to_train_{0}_{1}.pkl".format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path) and update == False:
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        data = get_action_data(start_date = compute_str_time(end_date, 7), end_date = end_date, field=['user_id', 'cate', 'type', 'time'])
        data = data[data.cate==8]
        drop_uid = data[(data.cate==8)&(data.type==4)].user_id.drop_duplicates()
        uid = data.user_id.to_frame().drop_duplicates()
        uid = uid[~uid.user_id.isin(drop_uid)]

        if offline:
            label = load_sub_eval_data(start_date=end_date, end_date=compute_str_time(end_date, 5, add=True))
            uid['label'] = uid.user_id.isin(label.user_id).astype(int)
        
        df = uid
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    
    return df

def extract_user_model_feat(start_date="2016-02-01 00:00:00", end_date="2016-04-11 00:00:00", offline = True, update = False):
    dump_path = "./cache/user_model_feat_{0}_{1}.pkl".format(start_date[:10], end_date[:10])
    if os.path.exists(dump_path) and update == False:
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        #用户基础特征
        merge_obj = []
        df = load_uid_to_train(start_date=start_date, end_date=end_date, offline = offline, update = update)
        base_user_feat = load_base_user_feat(end_date=end_date)
        merge_obj.append(base_user_feat)

        #规则特征
        rule_uid = load_rule_prdict_uid(start_date = start_date, end_date = end_date, sub_start_date=compute_str_time(end_date, 7), sub_end_date=end_date)
        df['rule_uid'] = (df.user_id.isin(rule_uid.user_id)).astype(int)

        #新老客户
        #用户最后一次交互时间
        user_last_act_tm = load_user_last_tm_dist(start_date = start_date, end_date = end_date)
        merge_obj.append(user_last_act_tm)
        #用户第一次交互时间
        user_first_login_tm = load_user_login_tm_dist(start_date = start_date, end_date = end_date)
        merge_obj.append(user_first_login_tm)
        #用户与第8类最后交互时间
        uc_last_tm_dist = load_UCPair_last_tm_dist(start_date = start_date, end_date = end_date)
        uc_last_tm_dist.columns = ['user_id', 'cate', 'uc_last_tm_dist']
        merge_obj.append(uc_last_tm_dist.drop(['cate'], axis=1))

        #活跃
        #用户登陆多少天
        user_total_login_cnt = load_user_action_date_cnt(start_date = start_date, end_date = end_date)
        user_total_login_cnt.columns = ['user_id', 'user_total_login_cnt']
        # merge_obj.append(user_total_login_cnt)

        #用户前7/15天登陆天数
        user_date_cnt_b7day = load_user_action_date_cnt(start_date=compute_str_time(end_date, 7), end_date=end_date)
        user_date_cnt_b7day.columns = ['user_id', 'user_date_cnt_b7day']
        user_date_cnt_b15day = load_user_action_date_cnt(start_date=compute_str_time(end_date, 15), end_date=end_date)
        user_date_cnt_b15day.columns = ['user_id', 'user_date_cnt_b15day']

        temp = pd.merge(user_total_login_cnt, user_date_cnt_b7day, on=['user_id'], how='left')
        temp = pd.merge(temp, user_date_cnt_b15day, on=['user_id'], how='left')

        #用户前7/15对第8类商品的交互天数
        uc_date_cnt_b7day = load_UCPair_action_date_cnt(start_date=compute_str_time(end_date, 7), end_date=end_date, cate = [8])
        uc_date_cnt_b7day.columns = ['user_id', 'cate', 'uc_date_cnt_b7day']
        uc_date_cnt_b7day = uc_date_cnt_b7day[['user_id', 'uc_date_cnt_b7day']]

        uc_date_cnt_b15day = load_UCPair_action_date_cnt(start_date=compute_str_time(end_date, 15), end_date=end_date, cate = [8])
        uc_date_cnt_b15day.columns = ['user_id', 'cate', 'uc_date_cnt_b15day']
        uc_date_cnt_b15day = uc_date_cnt_b15day[['user_id', 'uc_date_cnt_b15day']]

        temp = pd.merge(temp, uc_date_cnt_b7day, on=['user_id'], how='left')
        temp = pd.merge(temp, uc_date_cnt_b15day, on=['user_id'], how='left')

        temp.fillna(0, inplace=True)
        temp['date_ratio_7'] = temp['user_date_cnt_b7day'] / temp['user_total_login_cnt'].replace(0,1)
        temp['date_ratio_15'] = temp['user_date_cnt_b15day'] / temp['user_total_login_cnt'].replace(0,1)

        temp['uc_date_ratio_7'] = temp['uc_date_cnt_b7day'] / temp['user_date_cnt_b7day'].replace(0,1)
        temp['uc_date_ratio_15'] = temp['uc_date_cnt_b15day'] / temp['user_date_cnt_b15day'].replace(0,1)

        merge_obj.append(temp)

        #用户对第8类的关注程度
        #用户前7/15/60天的行为总数
        user_act_totalCnt_7day = load_user_action_totalCnt(start_date=compute_str_time(end_date, 7), end_date=end_date)
        user_act_totalCnt_7day.columns = ['user_id', 'user_act_totalCnt_7day']

        user_act_totalCnt_15day = load_user_action_totalCnt(start_date=compute_str_time(end_date, 15), end_date=end_date)
        user_act_totalCnt_15day.columns = ['user_id', 'user_act_totalCnt_15day']

        user_act_totalCnt = load_user_action_totalCnt(start_date=start_date, end_date=end_date)
        user_act_totalCnt.columns = ['user_id', 'user_act_totalCnt']

        uc_act_totalCnt_7day = load_UCPair_action_totalCnt(start_date=compute_str_time(end_date, 7), end_date=end_date)
        uc_act_totalCnt_7day = uc_act_totalCnt_7day[uc_act_totalCnt_7day.cate==8]
        uc_act_totalCnt_7day.columns = ['user_id', 'cate', 'uc_act_totalCnt_7day']
        uc_act_totalCnt_7day = uc_act_totalCnt_7day[['user_id', 'uc_act_totalCnt_7day']]

        uc_act_totalCnt_15day = load_UCPair_action_totalCnt(start_date=compute_str_time(end_date, 15), end_date=end_date)
        uc_act_totalCnt_15day = uc_act_totalCnt_15day[uc_act_totalCnt_15day.cate==8]
        uc_act_totalCnt_15day.columns = ['user_id', 'cate', 'uc_act_totalCnt_15day']
        uc_act_totalCnt_15day = uc_act_totalCnt_15day[['user_id', 'uc_act_totalCnt_15day']]

        uc_act_totalCnt = load_UCPair_action_totalCnt(start_date=start_date, end_date=end_date)
        uc_act_totalCnt.columns = ['user_id', 'cate', 'uc_act_totalCnt']
        uc_act_totalCnt = uc_act_totalCnt[uc_act_totalCnt.cate==8]
        uc_act_totalCnt = uc_act_totalCnt[['user_id', 'uc_act_totalCnt']]

        temp = pd.merge(user_act_totalCnt, user_act_totalCnt_7day, on=['user_id'], how='left')
        temp = pd.merge(temp, user_act_totalCnt_15day, on=['user_id'], how='left')
        temp = pd.merge(temp, uc_act_totalCnt_7day, on=['user_id'], how='left')
        temp = pd.merge(temp, uc_act_totalCnt_15day, on=['user_id'], how='left')
        temp = pd.merge(temp, uc_act_totalCnt, on=['user_id'], how='left')

        temp.fillna(0, inplace=True)
        temp['uc_act_ratio_7day'] = temp['uc_act_totalCnt_7day'] / (temp['user_act_totalCnt_7day'].replace(0,1))
        temp['uc_act_ratio_15day'] = temp['uc_act_totalCnt_15day'] / (temp['user_act_totalCnt_15day'].replace(0,1))
        temp['uc_act_ratio_60day'] = temp['uc_act_totalCnt'] / (temp['user_act_totalCnt'].replace(0,1))

        merge_obj.append(temp)

        #用户最大点击次数
        max_click = load_filter_uid(start_date = compute_str_time(end_date, 7), end_date = end_date)
        max_click.columns = ['user_id', 'max_click']
        merge_obj.append(max_click)

        freq_click = load_user_click_freq(start_date = compute_str_time(end_date, 7), end_date = end_date)
        freq_click.columns = ['user_id', 'freq_click']
        merge_obj.append(freq_click)

        #用户有效行为时间(1/2/3/5/7)
        for delta in [1,2,3,5,7]:
            user_act_time = load_user_act_time(start_date = compute_str_time(end_date, delta), end_date = end_date)
            user_act_time.columns = ['user_id', 'user_act_time_{0}day'.format(delta)]
            uc_act_time = load_UC_act_time(start_date = compute_str_time(end_date, delta), end_date = end_date)
            uc_act_time.columns = ['user_id', 'cate', 'uc_act_time_{0}day'.format(delta)]
            uc_act_time = uc_act_time[uc_act_time.cate==8]
            uc_act_time = uc_act_time[['user_id', 'uc_act_time_{0}day'.format(delta)]]

            temp = pd.merge(uc_act_time, user_act_time, on=['user_id'], how='left')
            temp['ratio_act_time_{0}day'.format(delta)] = temp['uc_act_time_{0}day'.format(delta)] / temp['user_act_time_{0}day'.format(delta)].replace(0,1)
            merge_obj.append(temp)
        
        #用户衰减特征
        uc_act_cnt_decay = load_UCPair_action_cnt_withDecay(start_date = compute_str_time(end_date, 15), end_date = end_date, actions=[1,2,3,5,6])
        uc_act_cnt_decay.columns = ['user_id', 'cate', 'uc_act_decay_1', 'uc_act_decay_2', 'uc_act_decay_3', 'uc_act_decay_5', 'uc_act_decay_6']
        uc_act_cnt_decay = uc_act_cnt_decay[uc_act_cnt_decay.cate==8]

        merge_obj.append(uc_act_cnt_decay.drop(['cate'], axis=1))

        #用户行为统计特征
        uc_act_cnt = load_UCPair_action_cnt(start_date = compute_str_time(end_date, 7), end_date = end_date, actions=[2,4,5])
        uc_act_cnt = uc_act_cnt[uc_act_cnt.cate==8]
        uc_act_cnt.columns = ['user_id', 'cate', 'uc_act_2', 'uc_act_4', 'uc_act_5']

        merge_obj.append(uc_act_cnt.drop(['cate'], axis=1))

        #用户前7天的均值特征

        temp = pd.merge(uc_date_cnt_b7day, uc_act_totalCnt_7day, on=['user_id'], how='left')
        temp['mean_uc_act'] = temp['uc_act_totalCnt_7day'] / temp['uc_date_cnt_b7day'].replace(0, 1)
        merge_obj.append(temp[['user_id', 'mean_uc_act']])

        uc_act_time_zone = load_UCPair_act_cnt_with_timeZone(start_date = compute_str_time(end_date, 7), end_date = end_date, cate=[8])
        uc_act_time_zone.columns = ['user_id','cate','uc_act_time_zone_0','uc_act_time_zone_1','uc_act_time_zone_2','uc_act_time_zone_3']
        uc_act_time_zone = uc_act_time_zone[['user_id','uc_act_time_zone_0','uc_act_time_zone_1','uc_act_time_zone_2','uc_act_time_zone_3']]
        temp = pd.merge(uc_act_time_zone, uc_date_cnt_b7day, on=['user_id'], how='left')
        temp.fillna(0, inplace=True)
        temp['uc_act_time_zone_0'] = temp['uc_act_time_zone_0'] / temp['uc_date_cnt_b7day'].replace(0,1)
        temp['uc_act_time_zone_1'] = temp['uc_act_time_zone_1'] / temp['uc_date_cnt_b7day'].replace(0,1)
        temp['uc_act_time_zone_2'] = temp['uc_act_time_zone_2'] / temp['uc_date_cnt_b7day'].replace(0,1)
        temp['uc_act_time_zone_3'] = temp['uc_act_time_zone_3'] / temp['uc_date_cnt_b7day'].replace(0,1)

        merge_obj.append(temp.drop(['uc_date_cnt_b7day'], axis=1))

        #用户是否购买其他品类商品
        uc_buy_bool = load_UCPair_action_bool(start_date=compute_str_time(end_date, 7), end_date=end_date, actions = [4])
        uc_buy_bool.columns = ['user_id', 'cate', 'uc_buy_bool_day7']
        uc_buy_bool = uc_buy_bool[uc_buy_bool.cate!=8]
        uc_buy_bool = uc_buy_bool.groupby(['user_id'], as_index=False)['uc_buy_bool_day7'].sum()
        uc_buy_bool['uc_buy_bool'] = (uc_buy_bool['uc_buy_bool_day7'] > 0).astype(int)

        merge_obj.append(uc_buy_bool[['user_id', 'uc_buy_bool_day7']])

        ##用户点击转化
        temp = load_UCPair_action_cnt(start_date = compute_str_time(end_date, 15), end_date = end_date, actions=[1,6])
        temp = temp[temp.cate==8]
        temp.columns = ['user_id', 'cate', 'act_15day_1', 'act_15day_6']
        temp.fillna(0, inplace=True)
        temp['ratio_1_6'] = temp['act_15day_1'] / temp['act_15day_6'].replace(0,1)

        merge_obj.append(temp[['user_id', 'ratio_1_6']])

        ##user 度
        for delta in [1,2,3,5,7]:
            data = get_action_data(start_date=compute_str_time(end_date, delta), end_date=end_date, field=['user_id', 'sku_id', 'time', 'cate'])
            total_indegree = gen_indegree(data[['user_id', 'sku_id', 'time']])
            total_indegree = total_indegree.groupby(['user_id'], as_index=False)['indegree'].sum()
            total_indegree.columns = ['user_id', 'indegree_total_{0}day'.format(delta)]

            uc_indegree = gen_indegree(data[data.cate==8][['user_id', 'sku_id', 'time']])
            uc_indegree = uc_indegree.groupby(['user_id'], as_index=False)['indegree'].sum()
            uc_indegree.columns = ['user_id', 'uc_indegree_{0}day'.format(delta)]

            temp = pd.merge(uc_indegree, total_indegree, on=['user_id'], how='left')
            temp.fillna(0, inplace=True)
            temp['ratio_indegree_{0}day'.format(delta)] = temp['uc_indegree_{0}day'.format(delta)] / temp['indegree_total_{0}day'.format(delta)].replace(0, 1)

            merge_obj.append(temp)

        print("开始拼接 {0}".format(df.shape))
        N_b = df.shape[0]
        for obj in merge_obj:
            df = pd.merge(df, obj, on=['user_id'], how='left')
        N_e = df.shape[0]
        
        assert N_b == N_e


        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)

    return df
