#-*-coding:utf-8-*-

##生成时间衰减权重文件
from features_generator import *
from tools import *
import warnings
warnings.filterwarnings("ignore")

TIMEDECAY = './datasets/TimeDecay.csv'

if __name__ == '__main__':
    print("开始生成时间衰减权重文件.")
    JData = get_action_data(start_date='2016-02-01 00:00:00', end_date='2016-04-11 00:00:00', field=['user_id', 'cate', 'time'])
    target = load_sub_eval_data(start_date='2016-04-11 00:00:00', end_date='2016-04-16 00:00:00')
    JData = JData.drop_duplicates()
    JData = JData[JData.cate==8]
    weights = []
    for hour in range(1,721):
        temp = JData[JData.time>=compute_str_time("2016-04-11 00:00:00", hour/24.0)].user_id.drop_duplicates()
        weight = temp.isin(target.user_id.drop_duplicates()).mean()
        weights.append(weight)
    weights = np.array(weights)
    #归一化 压缩至[0,1]
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    TimeDecay = pd.DataFrame({'weight':weights, 'time_dist':range(1,721)})
    print("保存时间衰减权重文件.")
    TimeDecay[['time_dist', 'weight']].to_csv(TIMEDECAY, index=False)