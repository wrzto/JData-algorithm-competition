#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def predict_user():
    print('预测用户.')
    xgb_model = xgb.Booster({'nthread':-1})
    xgb_model.load_model('./model/xgb_user.model')
    Online = pd.read_csv('./feat/online_user_model_feat.csv')
    donline = xgb.DMatrix(Online.drop(['user_id'], axis=1))
    xgb_proba = xgb_model.predict(donline)
    online_proba = Online[['user_id']]
    online_proba.loc[:,'proba'] = xgb_proba
    online_proba.to_csv("./online_user_proba.csv", index=False)

def predict_sku():
    print('预测商品.')
    xgb_model = xgb.Booster({'nthread':-1})
    xgb_model.load_model('./model/xgb_sku.model')
    Online = pd.read_csv("./feat/online_sku_feat.csv")
    Online_drop_cols = ['user_id', 'sku_id', 'cate', 'brand']
    donline = xgb.DMatrix(Online.drop(Online_drop_cols, axis=1))

    ##预测
    xgb_proba = xgb_model.predict(donline)
    sku_proba = Online[['user_id', 'sku_id']]
    sku_proba.loc[:,'sku_proba'] = xgb_proba

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
    print('运行结束.')
  

if __name__ == '__main__':
    predict_user()
    predict_sku()
