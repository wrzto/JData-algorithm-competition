#-*-coding:utf-8-*-

from sku_model import *

if __name__ == '__main__':
    Valid = extract_feat_to_model(feat_extract_start_date='2016-02-01 00:00:00', feat_extract_end_date='2016-04-06 00:00:00',\
                        label_extract_start_date = '2016-04-06 00:00:00', label_extract_end_date='2016-04-11 00:00:00', \
                        offline=True, update=True)

    Train = extract_feat_to_model(feat_extract_start_date='2016-02-06 00:00:00', feat_extract_end_date='2016-04-11 00:00:00',\
                        label_extract_start_date = '2016-04-11 00:00:00', label_extract_end_date='2016-04-16 00:00:00', \
                        offline=True, update=True)

    Online = extract_feat_to_model(feat_extract_start_date='2016-02-11 00:00:00', feat_extract_end_date='2016-04-16 00:00:00',\
                        label_extract_start_date = '2016-04-16 00:00:00', label_extract_end_date='2016-04-21 00:00:00', \
                        offline=False, update=True)

    train_user_feat = pd.read_csv('./feat/train_user_model_feat.csv')
    valid_user_feat = pd.read_csv('./feat/valid_user_model_feat.csv')
    online_user_feat = pd.read_csv('./feat/online_user_model_feat.csv')

    train = pd.merge(Train, train_user_feat.drop(['label'], axis=1), on=['user_id'], how='left')
    valid = pd.merge(Valid, valid_user_feat.drop(['label'], axis=1), on=['user_id'], how='left')
    online = pd.merge(Online, online_user_feat, on=['user_id'], how='left')

    train.to_csv("./feat/train_sku_feat.csv", index=False)
    valid.to_csv("./feat/valid_sku_feat.csv", index=False)
    online.to_csv("./feat/online_sku_feat.csv", index=False)