#coding=utf-8
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
def xgbpa(trainX, trainY):
    # init
    xgb1 = XGBClassifier(
        learning_rate=0.3,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=6
    )

    # max_depth 和 min_weight 参数调优
    param1 = {'max_depth': list(range(3, 7)), 'min_child_weight': list(range(1, 5, 2))}

    from sklearn import svm, datasets
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.3,
            n_estimators=150,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=6
        ),
        param_grid=param1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(trainX, trainY)
    print(gsearch1.scorer_)
    print(gsearch1.best_params_, gsearch1.best_score_)
    best_max_depth = gsearch1.best_params_['max_depth']
    best_min_child_weight = gsearch1.best_params_['min_child_weight']

    # gamma参数调优
    param2 = {'gamma': [i / 10.0 for i in range(0, 5, 2)]}
    gsearch2 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.3,  # 如同学习率
            n_estimators=150,  # 树的个数
            max_depth=best_max_depth,
            min_child_weight=best_min_child_weight,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=6
        ),
        param_grid=param2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch2.fit(trainX, trainY)
    print(gsearch2.scorer_)
    print(gsearch2.best_params_, gsearch2.best_score_)
    best_gamma = gsearch2.best_params_['gamma']

    # 调整subsample 和 colsample_bytree参数
    param3 = {'subsample': [i / 10.0 for i in range(6, 9)], 'colsample_bytree': [i / 10.0 for i in range(6, 9)]}
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.3,
            n_estimators=150,
            max_depth=best_max_depth,
            min_child_weight=best_min_child_weight,
            gamma=best_gamma,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=6
        ),
        param_grid=param3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch3.fit(trainX, trainY)
    print(gsearch3.scorer_)
    print(gsearch3.best_params_, gsearch3.best_score_)
    best_subsample = gsearch3.best_params_['subsample']
    best_colsample_bytree = gsearch3.best_params_['colsample_bytree']

    # 正则化参数调优
    param4 = {'reg_alpha': [i / 10.0 for i in range(2, 10, 2)], 'reg_lambda': [i / 10.0 for i in range(2, 10, 2)]}
    gsearch4 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.3,
            n_estimators=150,
            max_depth=best_max_depth,
            min_child_weight=best_min_child_weight,
            gamma=best_gamma,
            subsample=best_subsample,
            colsample_bytree=best_colsample_bytree,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=6
        ),
        param_grid=param4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(trainX, trainY)
    print(gsearch4.scorer_)
    print(gsearch4.best_params_, gsearch4.best_score_)
    best_reg_alpha = gsearch4.best_params_['reg_alpha']
    best_reg_lambda = gsearch4.best_params_['reg_lambda']


    param5= {'scale_pos_weight': [i for i in [0.5, 1, 2]]}

    gsearch5 = GridSearchCV(
        estimator = XGBClassifier(
            learning_rate = 0.3,
            n_estimators = 150,
            max_depth = best_max_depth,
            min_child_weight = best_min_child_weight,
            gamma = best_gamma,
            subsample = best_subsample,
            colsample_bytree = best_colsample_bytree,
            reg_alpha = best_reg_alpha,
            reg_lambda = best_reg_lambda,
            objective = 'binary:logistic',
            nthread = 4,
            scale_pos_weight = 1,
            seed = 6
            ),
        param_grid = param5, scoring = 'roc_auc', n_jobs = 4, iid = False, cv = 5)
    gsearch5.fit(trainX, trainY)
    print(gsearch5.best_params_, gsearch5.best_score_)
    best_scale_pos_weight = gsearch5.best_params_['scale_pos_weight']

    # 降低学习速率，数的数量
    param6 = [{'learning_rate': [0.01, 0.05, 0.1, 0.2], 'n_estimators': [800, 1000, 1200]}]

    gsearch6 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.3,
            n_estimators=150,
            max_depth=best_max_depth,
            min_child_weight=best_min_child_weight,
            gamma=best_gamma,
            subsample=best_subsample,
            colsample_bytree=best_colsample_bytree,
            reg_alpha=best_reg_alpha,
            reg_lambda = best_reg_lambda,
            objective = 'binary:logistic',
            nthread = 4,
            scale_pos_weight = best_scale_pos_weight,
            seed = 6
    ),
    param_grid = param6, scoring = 'roc_auc', n_jobs = 4, iid = False, cv = 5)
    gsearch6.fit(trainX, trainY)
    print(gsearch6.scorer_)
    print(gsearch6.best_params_, gsearch6.best_score_)
    best_learning_rate = gsearch6.best_params_['learning_rate']
    best_n_estimators = gsearch6.best_params_['n_estimators']
	
    print(gsearch1.best_params_, gsearch1.best_score_)
    print(gsearch2.best_params_, gsearch2.best_score_)
    print(gsearch3.best_params_, gsearch3.best_score_)
    print(gsearch4.best_params_, gsearch4.best_score_)
    print(gsearch5.best_params_, gsearch5.best_score_)
    print(gsearch6.best_params_, gsearch6.best_score_)


if __name__ == '__main__':
    # user_model cv
    print('--------------user model---------------')
    Train = pd.read_csv('./train_user_model_feat.csv')
    xgbpa(Train.drop(['user_id', 'label'], axis=1), Train.label)
    del Train
    
    #sku_model cv
    print('--------------sku model---------------')
    Drop_cols = ['user_id', 'sku_id', 'cate', 'brand', 'label']
    Train = pd.read_csv("./train_sku_feat.csv")
    xgbpa(Train.drop(Drop_cols, axis=1), Train.label)
    del Train
