#-*-coding:utf-8-*-

from user_model_final import *

if __name__ == '__main__':
    Valid = extract_user_model_feat(start_date='2016-02-01 00:00:00', end_date='2016-04-06 00:00:00', offline=True, update=True)
    Train = extract_user_model_feat(start_date='2016-02-06 00:00:00', end_date='2016-04-11 00:00:00', offline=True, update=True)
    Online = extract_user_model_feat(start_date='2016-02-11 00:00:00', end_date='2016-04-16 00:00:00', offline=False, update=True)

    Valid.to_csv("./feat/valid_user_model_feat.csv", index=False)
    Train.to_csv("./feat/train_user_model_feat.csv", index=False)
    Online.to_csv("./feat/online_user_model_feat.csv", index=False)