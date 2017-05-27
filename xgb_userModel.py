#-*-coding:utf-8-*-

import xgboost as xgb
import pandas as pd
import numpy as np
from tools import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print("开始训练用户模型.")
    Train = pd.read_csv('./feat/train_user_model_feat.csv')
    # Valid = pd.read_csv('./feat/valid_user_model_feat.csv')
    Online = pd.read_csv('./feat/online_user_model_feat.csv')

    # target = load_sub_eval_data(start_date='2016-04-06 00:00:00', end_date='2016-04-11 00:00:00')
    
    ## 网格搜索调整的参数
    param = {'learning_rate' : 0.01, 'n_estimators': 1200, 'max_depth':3, 
        'min_child_weight': 1, 'gamma': 0.0, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logistic',
        'nthread':-1, 'eval_metric':["auc", "logloss"], 'alpha':0.4, 'lambda':0.6}
    
    dtrain = xgb.DMatrix(Train.drop(['user_id', 'label'], axis=1), label=Train.label)
    # dvalid = xgb.DMatrix(Valid.drop(['user_id', 'label'], axis=1), label=Valid.label)
    donline = xgb.DMatrix(Online.drop(['user_id'], axis=1))

    ## cv 调整最佳迭代次数
    cv_output = xgb.cv(dtrain=dtrain, params=param, num_boost_round=3000, early_stopping_rounds=20, verbose_eval=5, show_stdv=False, nfold=5, seed=6)
    num_boost_rounds = len(cv_output)
    xgb_model = xgb.train(dict(param, silent=1), dtrain, num_boost_round= num_boost_rounds)

    ##预测用户购买第8类商品的概率并保存到文件
    xgb_proba = xgb_model.predict(donline)
    online_proba = Online[['user_id']]
    online_proba['proba'] = xgb_proba
    online_proba.to_csv("./online_user_proba.csv", index=False)
    xgb_model.save_model('./model/xgb_user.model')
    print("用户模型训练完成，已保存至model文件夹.")

