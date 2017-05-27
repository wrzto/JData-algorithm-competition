#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ACTION201602_FILE ='./datasets/JData_Action_201602.csv'
ACTION201603_FILE ='./datasets/JData_Action_201603.csv'
ACTION201604_FILE ='./datasets/JData_Action_201604.csv'
ACTION_FILE='./datasets/JData_Action.csv'


if __name__ == '__main__':
    print("开始拼接文件.")
    JData_201602 = pd.read_csv(ACTION201602_FILE)
    JData_201603 = pd.read_csv(ACTION201603_FILE)
    JData_201604 = pd.read_csv(ACTION201604_FILE)
    JData = pd.concat([JData_201602, JData_201603, JData_201604])
    del JData_201602
    del JData_201603
    del JData_201604
    JData.user_id = JData.user_id.astype(int)
    tm = pd.to_datetime(JData.time)
    JData['time_zone'] = tm.apply(lambda t: int(t.hour / 6))
    print("保存拼接文件结果.")
    JData.to_csv(ACTION_FILE, index=False)


