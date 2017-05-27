#-*-coding:utf-8-*-

import pandas as pd
from tools import *
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print("开始训练商品模型.")
    print("训练时间约20分钟,无打印信息.")
    Train = pd.read_csv("./feat/train_sku_feat.csv")
    # Valid = pd.read_csv("./feat/valid_sku_feat.csv")
    Online = pd.read_csv("./feat/online_sku_feat.csv")

    Drop_cols = ['user_id', 'sku_id', 'cate', 'brand', 'label']
    Online_drop_cols = ['user_id', 'sku_id', 'cate', 'brand']

    dtrain = xgb.DMatrix(Train.drop(Drop_cols, axis=1), label=Train.label)
    # dvalid = xgb.DMatrix(Valid.drop(Drop_cols, axis=1))
    donline = xgb.DMatrix(Online.drop(Online_drop_cols, axis=1))

    ##网格搜索调整的参数 并非最优最优参数 
    param = {'learning_rate' : 0.01, 'n_estimators': 1400, 'max_depth':3, 
        'min_child_weight': 3, 'gamma': 0.4, 'subsample': 0.88, 'colsample_bytree': 0.76,
        'scale_pos_weight': 1.2, 'silent': 1, 'objective': 'binary:logistic','eta':0.01,
        'nthread':-1, 'eval_metric':["auc", "logloss"], 'alpha':0.6, 'lambda':0.6}
    
    # ##cv 获取最佳迭代次数
    # cv_output = xgb.cv(dtrain=dtrain, params=param, num_boost_round=5000, early_stopping_rounds=50, verbose_eval=50, show_stdv=False, nfold=5, seed=6)
    # ##1378
    # num_boost_rounds = len(cv_output)
    # xgb_model = xgb.train(dict(param, silent=1), dtrain, num_boost_round= num_boost_rounds)

    #可以根据上方注释获取最佳迭代次数
    xgb_model = xgb.train(dict(param, silent=1), dtrain, num_boost_round= 1378)

    ##预测
    xgb_proba = xgb_model.predict(donline)
    sku_proba = Online[['user_id', 'sku_id']]
    sku_proba['sku_proba'] = xgb_proba

    ##提取每个用户购买概率最大的商品
    sku_proba = sku_proba.groupby(['user_id'], as_index=False).apply(lambda t: t[t.sku_proba == t.sku_proba.max()]).reset_index()[['user_id', 'sku_id', 'sku_proba']]
    ##读取用户模型训练结果
    user_proba = pd.read_csv("./online_user_proba.csv")

    ##按照概率值从大到小排序
    sku_proba.sort_values(by="sku_proba", ascending=False, inplace=True)
    user_proba.sort_values(by="proba", ascending=False, inplace=True)

    ##用户模型 与 商品模型 各取前500并集
    Top_user = user_proba.iloc[:500]
    Top_sku = sku_proba.iloc[:500][['user_id', 'sku_id']]
    Top_user = sku_proba[sku_proba.user_id.isin(Top_user.user_id)]
    Top_user = Top_user.groupby(['user_id'], as_index=False).apply(lambda t: t[t.sku_proba == t.sku_proba.max()]).reset_index()[['user_id', 'sku_id']]

    pred = pd.concat([Top_sku, Top_user])
    pred = pred.drop_duplicates()
    pred = pred[pred.user_id.duplicated()==False]

    pred.astype(int).to_csv("online_submit.csv", index=False)
    xgb_model.save_model('./model/xgb_sku.model')
    print("商品模型训练完成,模型已保存至model文件夹,预测结果保存至当前根目录,名为online_submit.csv。")


