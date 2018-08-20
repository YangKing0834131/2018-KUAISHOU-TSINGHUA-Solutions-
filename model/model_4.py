# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from functools import reduce
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


def load_data():
    # A：source_data
    app_launch_a = pd.read_table('data/a3d6_chusai_a_train/app_launch_log.txt', names=['user_id', 'day'],encoding='utf-8', sep='\t', )
    user_activity_a = pd.read_table('data/a3d6_chusai_a_train/user_activity_log.txt',names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'],encoding='utf-8', sep='\t')
    user_register_a = pd.read_table('data/a3d6_chusai_a_train/user_register_log.txt',names=['user_id', 'register_day', 'register_type', 'device_type'], encoding='utf-8',sep='\t')
    video_create_a = pd.read_table('data/a3d6_chusai_a_train/video_create_log.txt', names=['user_id', 'day'],encoding='utf-8', sep='\t')
    # B：source_data
    app_launch_b = pd.read_table('data/chusai_b_train/app_launch_log.txt', names=['user_id', 'day'],encoding='utf-8', sep='\t', )
    user_activity_b = pd.read_table('data/chusai_b_train/user_activity_log.txt',names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'],encoding='utf-8', sep='\t')
    user_register_b = pd.read_table('data/chusai_b_train/user_register_log.txt',names=['user_id', 'register_day', 'register_type', 'device_type'], encoding='utf-8',sep='\t')
    video_create_b = pd.read_table('data/chusai_b_train/video_create_log.txt', names=['user_id', 'day'],encoding='utf-8', sep='\t')
    # A榜 数据重编码的user_id
    app_launch_a['user_id'] = app_launch_a['user_id'].map(lambda x: 'a_' + str(x))
    user_activity_a['user_id'] = user_activity_a['user_id'].map(lambda x: 'a_' + str(x))
    user_register_a['user_id'] = user_register_a['user_id'].map(lambda x: 'a_' + str(x))
    video_create_a['user_id'] = video_create_a['user_id'].map(lambda x: 'a_' + str(x))
    # 重编码a   author_id
    user_activity_a['author_id'] = user_activity_a['author_id'].map(lambda x: 'a_' + str(x))
    # 重编码a   video_id
    user_activity_a['video_id'] = user_activity_a['video_id'].map(lambda x: 'a_' + str(x))

    # B榜  数据重编码的user_id
    app_launch_b['user_id'] = app_launch_b['user_id'].map(lambda x: 'b_' + str(x))
    user_activity_b['user_id'] = user_activity_b['user_id'].map(lambda x: 'b_' + str(x))
    user_register_b['user_id'] = user_register_b['user_id'].map(lambda x: 'b_' + str(x))
    video_create_b['user_id'] = video_create_b['user_id'].map(lambda x: 'b_' + str(x))
    # 重编码B的author_id
    user_activity_b['author_id'] = user_activity_b['author_id'].map(lambda x: 'b_' + str(x))
    # 重编码B的video_id
    user_activity_b['video_id'] = user_activity_b['video_id'].map(lambda x: 'b_' + str(x))

    # 合并A+B数据
    app_launch = pd.concat([app_launch_a, app_launch_b], axis=0)
    user_activity = pd.concat([user_activity_a, user_activity_b], axis=0)
    user_register = pd.concat([user_register_a, user_register_b], axis=0)
    video_create = pd.concat([video_create_a, video_create_b], axis=0)

    # 重置index
    app_launch.index = range(len(app_launch))
    user_activity.index = range(len(user_activity))
    user_register.index = range(len(user_register))
    video_create.index = range(len(video_create))

    # user_register表：离散register_type
    data = user_register[['user_id', 'register_type']].copy()
    label_data = pd.get_dummies(data['register_type'], prefix='register_type')
    data = pd.concat([data, label_data], axis=1)
    del data['register_type']
    user_register = pd.merge(user_register, data, on='user_id', how='left')

    # # user_register表：register_type和device_type标准化标签
    data = user_register[['register_type', 'device_type']].copy()
    data['register_type_device_type'] = list(map(lambda x, y: str(x) + '_' + str(y), data['register_type'], data['device_type']))
    le = preprocessing.LabelEncoder()
    le.fit(data['register_type_device_type'])
    data['register_type_device_type_LabelEncoder'] = le.transform(data['register_type_device_type'])
    user_register['register_type_device_type_LabelEncoder'] = data['register_type_device_type_LabelEncoder']
    return app_launch,user_activity,user_register,video_create

def get_dataset(app_launch, user_activity, user_register, video_create):
    # 打标和特征区间划分
    train_data_label_1_to_9, train_data_label_1_to_16, train_data_label_1_to_23, test_data_label_1_to_30, \
    train_app_launch_feature1_to_9, train_user_activity1_to_9, train_user_register_feature1_to_9, train_video_create1_to_9, \
    train_app_launch_feature8_to_16, train_user_activity8_to_16, train_user_register_feature8_to_16, train_video_create8_to_16, \
    train_app_launch_feature15_to_23, train_user_activity15_to_23, train_user_register_feature15_to_23, train_video_create15_to_23, \
    train_app_launch_feature22_to_30, train_user_activity22_to_30, train_user_register_feature22_to_30, train_video_create22_to_30 =Marking_label(app_launch, user_activity, user_register, video_create)
    # 显示进度
    i = 0
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1
    # 测试集 特征提取
    test_app_launch_feature_data22_to_30 = App_launch_log_Fun(test_data_label_1_to_30, train_app_launch_feature22_to_30)
    test_video_create_feature_data22_to_30 = Video_create_log_Fun(test_data_label_1_to_30, train_video_create22_to_30)
    test_user_activity_feature_data22_to_30 = User_activity_log_Fun(test_data_label_1_to_30,train_user_activity22_to_30)
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1

    # # 训练集 特征提取
    train_app_launch_feature_data15_to_23 = App_launch_log_Fun(train_data_label_1_to_23,train_app_launch_feature15_to_23)
    train_video_create_feature_data15_to_23 = Video_create_log_Fun(train_data_label_1_to_23, train_video_create15_to_23)
    train_user_activity_feature_data15_to_23 = User_activity_log_Fun(train_data_label_1_to_23,train_user_activity15_to_23)
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1

    train_app_launch_feature_data8_to_16 = App_launch_log_Fun(train_data_label_1_to_16, train_app_launch_feature8_to_16)
    train_video_create_feature_data8_to_16 = Video_create_log_Fun(train_data_label_1_to_16, train_video_create8_to_16)
    train_user_activity_feature_data8_to_16 = User_activity_log_Fun(train_data_label_1_to_16,train_user_activity8_to_16)
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1

    train_app_launch_feature_data1_to_9 = App_launch_log_Fun(train_data_label_1_to_9, train_app_launch_feature1_to_9)
    train_video_create_feature_data1_to_9 = Video_create_log_Fun(train_data_label_1_to_9, train_video_create1_to_9)
    train_user_activity_feature_data1_to_9 = User_activity_log_Fun(train_data_label_1_to_9, train_user_activity1_to_9)
    print(str(round(i/4*100)) + '%')
    i += 1

    # 测试集 连接特征
    test_label_1_to_30 = pd.merge(test_data_label_1_to_30, test_app_launch_feature_data22_to_30, on='user_id',how='left')
    test_label_1_to_30 = pd.merge(test_label_1_to_30, test_video_create_feature_data22_to_30, on='user_id', how='left')
    test_label_1_to_30 = pd.merge(test_label_1_to_30, test_user_activity_feature_data22_to_30, on='user_id', how='left')
    test_label_1_to_30 = pd.merge(test_label_1_to_30, user_register, on='user_id', how='left')

    # 训练集 连接特征
    train_label_1_to_23 = pd.merge(train_data_label_1_to_23, train_app_launch_feature_data15_to_23, on='user_id',how='left')
    train_label_1_to_23 = pd.merge(train_label_1_to_23, train_video_create_feature_data15_to_23, on='user_id',how='left')
    train_label_1_to_23 = pd.merge(train_label_1_to_23, train_user_activity_feature_data15_to_23, on='user_id',how='left')
    train_label_1_to_23 = pd.merge(train_label_1_to_23, user_register, on='user_id', how='left')

    train_label_1_to_16 = pd.merge(train_data_label_1_to_16, train_app_launch_feature_data8_to_16, on='user_id',how='left')
    train_label_1_to_16 = pd.merge(train_label_1_to_16, train_video_create_feature_data8_to_16, on='user_id',how='left')
    train_label_1_to_16 = pd.merge(train_label_1_to_16, train_user_activity_feature_data8_to_16, on='user_id',how='left')
    train_label_1_to_16 = pd.merge(train_label_1_to_16, user_register, on='user_id', how='left')

    train_label_1_to_9 = pd.merge(train_data_label_1_to_9, train_app_launch_feature_data1_to_9, on='user_id',how='left')
    train_label_1_to_9 = pd.merge(train_label_1_to_9, train_video_create_feature_data1_to_9, on='user_id', how='left')
    train_label_1_to_9 = pd.merge(train_label_1_to_9, train_user_activity_feature_data1_to_9, on='user_id', how='left')
    train_label_1_to_9 = pd.merge(train_label_1_to_9, user_register, on='user_id', how='left')

    # # 连接训练集
    train_label_1_to_23 = train_label_1_to_23.sort_values(by='user_id', ascending=False)
    train_label_1_to_16 = train_label_1_to_16.sort_values(by='user_id', ascending=False)
    train_label_1_to_9 = train_label_1_to_9.sort_values(by='user_id', ascending=False)
    test_label_1_to_30 = test_label_1_to_30.sort_values(by='user_id', ascending=False)
    train_label_data = pd.concat([train_label_1_to_23, train_label_1_to_16, train_label_1_to_9])

    # 交互特征提取
    train_label_data = Interactive_feature(train_label_data)
    test_label_1_to_30 = Interactive_feature(test_label_1_to_30)

    # 测试集挑选出B榜数据
    test_label_1_to_30 = test_label_1_to_30[test_label_1_to_30['user_id'].map(lambda x: 'b_' in x)]
    test_label_1_to_30['user_id'] = test_label_1_to_30['user_id'].map(lambda x: int(x.split('_')[1]))
    
    # 重置index
    train_label_data.index = range(len(train_label_data))
    test_label_1_to_30.index = range(len(test_label_1_to_30))

    # 返回
    return train_label_data,test_label_1_to_30

def Marking_label(app_launch, user_activity, user_register, video_create):
    '''
    划分数据、打标
    :return: 
    '''
    # 提供打标数据源
    app_launch1_to_9 = app_launch[app_launch['day'] <= 9]
    app_launch1_to_16 = app_launch[app_launch['day'] <=16]
    app_launch1_to_23 = app_launch[(app_launch['day'] <= 23)]
    app_launch1_to_30 = app_launch[app_launch['day'] <= 30]
    app_launch8_to_16 = app_launch[(app_launch['day'] >= 8) & (app_launch['day'] <= 16)]
    app_launch15_to_23 = app_launch[(app_launch['day'] >= 15) & (app_launch['day'] <= 23)]
    app_launch22_to_30 = app_launch[(app_launch['day'] >= 22) & (app_launch['day'] <= 30)]
    app_launch10_to_16 = app_launch[(app_launch['day'] >= 10) & (app_launch['day'] <= 16)]
    app_launch17_to_23 = app_launch[(app_launch['day'] >= 17) & (app_launch['day'] <= 23)]
    app_launch24_to_30 = app_launch[(app_launch['day'] >= 24) & (app_launch['day'] <= 30)]

    user_activity1_to_9 = user_activity[user_activity['day'] <= 9]
    user_activity1_to_16 = user_activity[user_activity['day'] <= 16]
    user_activity1_to_23 = user_activity[(user_activity['day'] <= 23)]
    user_activity1_to_30 = user_activity[user_activity['day'] <= 30]
    user_activity8_to_16 = user_activity[(user_activity['day'] >= 8) & (user_activity['day'] <= 16)]
    user_activity15_to_23 = user_activity[(user_activity['day'] >= 15) & (user_activity['day'] <= 23)]
    user_activity22_to_30 = user_activity[(user_activity['day'] >= 22) & (user_activity['day'] <= 30)]
    user_activity10_to_16 = user_activity[(user_activity['day'] >= 10) & (user_activity['day'] <= 16)]
    user_activity17_to_23 = user_activity[(user_activity['day'] >= 17) & (user_activity['day'] <= 23)]
    user_activity24_to_30 = user_activity[(user_activity['day'] >= 24) & (user_activity['day'] <= 30)]

    user_register1_to_9 = user_register[user_register['register_day'] <= 9]
    user_register1_to_16 = user_register[user_register['register_day'] <= 16]
    user_register1_to_23 = user_register[(user_register['register_day'] <= 23)]
    user_register1_to_30 = user_register[user_register['register_day'] <= 30]
    user_register8_to_16 = user_register[(user_register['register_day'] >= 8) & (user_register['register_day'] <= 16)]
    user_register15_to_23 = user_register[(user_register['register_day'] >= 15) & (user_register['register_day'] <= 23)]
    user_register22_to_30 = user_register[(user_register['register_day'] >= 22) & (user_register['register_day'] <= 30)]
    user_register10_to_16 = user_register[(user_register['register_day'] >= 10) & (user_register['register_day'] <= 16)]
    user_register17_to_23 = user_register[(user_register['register_day'] >= 17) & (user_register['register_day'] <= 23)]
    user_register24_to_30 = user_register[(user_register['register_day'] >= 24) & (user_register['register_day'] <= 30)]

    video_create1_to_9 = video_create[video_create['day'] <= 9]
    video_create1_to_16 = video_create[video_create['day'] <= 16]
    video_create1_to_23 = video_create[(video_create['day'] <= 23)]
    video_create1_to_30 = video_create[video_create['day'] <= 30]
    video_create8_to_16 = video_create[(video_create['day'] >= 8) & (video_create['day'] <= 16)]
    video_create15_to_23 = video_create[(video_create['day'] >= 15) & (video_create['day'] <= 23)]
    video_create22_to_30 = video_create[(video_create['day'] >= 22) & (video_create['day'] <= 30)]
    video_create10_to_16 = video_create[(video_create['day'] >= 10) & (video_create['day'] <= 16)]
    video_create17_to_23 = video_create[(video_create['day'] >= 17) & (video_create['day'] <= 23)]
    video_create24_to_30 = video_create[(video_create['day'] >= 24) & (video_create['day'] <= 30)]

    data1_to_9 = list(set(app_launch1_to_9['user_id']) | set(user_activity1_to_9['user_id']) | set(user_register1_to_9['user_id']) | set(video_create1_to_9['user_id']))
    data1_to_9 = pd.DataFrame(data1_to_9, columns=['user_id'])

    data1_to_16 = list(set(app_launch1_to_16['user_id']) | set(user_activity1_to_16['user_id']) | set(user_register1_to_16['user_id']) | set(video_create1_to_16['user_id']))
    data1_to_16 = pd.DataFrame(data1_to_16, columns=['user_id'])

    data1_to_23 = list(set(app_launch1_to_23['user_id']) | set(user_activity1_to_23['user_id']) | set(user_register1_to_23['user_id']) | set(video_create1_to_23['user_id']))
    data1_to_23 = pd.DataFrame(data1_to_23, columns=['user_id'])

    data1_to_30 = list(set(app_launch1_to_30['user_id']) | set(user_activity1_to_30['user_id']) | set(user_register1_to_30['user_id']) | set(video_create1_to_30['user_id']))
    data1_to_30 = pd.DataFrame(data1_to_30, columns=['user_id'])

    data10_to_16 = list(set(app_launch10_to_16['user_id']) | set(user_activity10_to_16['user_id']) | set(user_register10_to_16['user_id']) | set(video_create10_to_16['user_id']))
    data10_to_16 = pd.DataFrame(data10_to_16, columns=['user_id'])

    data17_to_23 = list(set(app_launch17_to_23['user_id']) | set(user_activity17_to_23['user_id']) | set(user_register17_to_23['user_id']) | set(video_create17_to_23['user_id']))
    data17_to_23 = pd.DataFrame(data17_to_23, columns=['user_id'])

    data24_to_30 = list(set(app_launch24_to_30['user_id']) | set(user_activity24_to_30['user_id']) | set(user_register24_to_30['user_id']) | set(video_create24_to_30['user_id']))
    data24_to_30 = pd.DataFrame(data24_to_30, columns=['user_id'])

    # 打标
    # 测试集 标签区间1-30号
    test_data_label_1_to_30 = data1_to_30

    # 训练集打标 7个划窗
    set1_to_23 = list(set(data1_to_23['user_id']) & set(data24_to_30['user_id']))
    set1_to_23 = pd.DataFrame(set1_to_23, columns=['user_id'])
    set1_to_23['label'] = 1
    data1_to_23 = pd.merge(data1_to_23, set1_to_23, on='user_id', how='left')
    train_data_label_1_to_23 = data1_to_23.fillna(0)

    set1_to_16 = list(set(data1_to_16['user_id']) & set(data17_to_23['user_id']))
    set1_to_16 = pd.DataFrame(set1_to_16, columns=['user_id'])
    set1_to_16['label'] = 1
    data1_to_16 = pd.merge(data1_to_16, set1_to_16, on='user_id', how='left')
    train_data_label_1_to_16 = data1_to_16.fillna(0)

    set1_to_9 = list(set(data1_to_9['user_id']) & set(data10_to_16['user_id']))
    set1_to_9 = pd.DataFrame(set1_to_9, columns=['user_id'])
    set1_to_9['label'] = 1
    data1_to_9 = pd.merge(data1_to_9, set1_to_9, on='user_id', how='left')
    train_data_label_1_to_9 = data1_to_9.fillna(0)


    # 训练集 特征区间17-23号
    train_app_launch_feature1_to_9 = app_launch1_to_9
    train_user_activity1_to_9 = user_activity1_to_9
    train_user_register_feature1_to_9 = user_register1_to_9
    train_video_create1_to_9 = video_create1_to_9

    train_app_launch_feature8_to_16 = app_launch8_to_16
    train_user_activity8_to_16 = user_activity8_to_16
    train_user_register_feature8_to_16 = user_register8_to_16
    train_video_create8_to_16 = video_create8_to_16

    train_app_launch_feature15_to_23 = app_launch15_to_23
    train_user_activity15_to_23 = user_activity15_to_23
    train_user_register_feature15_to_23 = user_register15_to_23
    train_video_create15_to_23 = video_create15_to_23

    train_app_launch_feature22_to_30 = app_launch22_to_30
    train_user_activity22_to_30 = user_activity22_to_30
    train_user_register_feature22_to_30 = user_register22_to_30
    train_video_create22_to_30 = video_create22_to_30

    return train_data_label_1_to_9,train_data_label_1_to_16,train_data_label_1_to_23,test_data_label_1_to_30, \
           train_app_launch_feature1_to_9,train_user_activity1_to_9,train_user_register_feature1_to_9,train_video_create1_to_9, \
           train_app_launch_feature8_to_16,train_user_activity8_to_16,train_user_register_feature8_to_16 ,train_video_create8_to_16,\
           train_app_launch_feature15_to_23,train_user_activity15_to_23,train_user_register_feature15_to_23,train_video_create15_to_23,\
           train_app_launch_feature22_to_30,train_user_activity22_to_30,train_user_register_feature22_to_30,train_video_create22_to_30

def App_launch_log_Fun(label_data,feature_data):
    '''
    提取APP 启动日志相关特征
    :return: 
    '''
    App_user_id=list(set(label_data['user_id']))
    App_launch_log_data = pd.DataFrame(App_user_id, columns=['user_id'])

    day_max = max(feature_data['day'])
    day_min = min(feature_data['day'])
    # #用户每天出现次数###################################
    for i in range(day_min, day_max + 1):
        data = feature_data[['user_id', 'day']].copy()
        data = data[data['day'] == i]
        data['app_lau_user_last_cnt' + str(i - day_min) + ''] = 1
        del data['day']
        data = data.groupby(['user_id']).agg('sum').reset_index()
        App_launch_log_data = pd.merge(App_launch_log_data, data, on=['user_id'], how='left')

    # 用户出现方差、总和、平均数、最大值、最小值
    needs = []
    for col in App_launch_log_data.columns.tolist():
        if 'app_lau_user_last_cnt' in col:
            needs.append(col)
    App_launch_log_data.fillna(0, inplace=True)
    App_launch_log_data['app_lau_user_cnt_var_cnt'] = App_launch_log_data[needs].var(1)
    App_launch_log_data['app_lau_user_cnt_sum_cnt'] = App_launch_log_data[needs].sum(1)
    App_launch_log_data['app_lau_user_cnt_avg_cnt'] = App_launch_log_data[needs].mean(1)
    App_launch_log_data['app_lau_user_cnt_max_cnt'] = App_launch_log_data[needs].max(1)
    App_launch_log_data['app_lau_user_cnt_min_cnt'] = App_launch_log_data[needs].min(1)

    # #用户登陆天数
    needs = []
    for col in App_launch_log_data.columns.tolist():
        if 'app_lau_user_last_cnt' in col:
            needs.append(col)
    df = App_launch_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    needs = []
    for col in df.columns.tolist():
        if 'app_lau_user_last_cnt' in col:
            needs.append(col)
    df['launch_continue_sum_cnt'] = df[needs].sum(1)
    App_launch_log_data['launch_continue_sum_cnt'] = df['launch_continue_sum_cnt']

    # #用户连续登陆最大值
    needs = []
    for col in App_launch_log_data.columns.tolist():
        if 'app_lau_user_last_cnt' in col:
            needs.append(col)
    df = App_launch_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    df['launch_list'] = df.apply(lambda x: reduce(lambda y, z: str(y) + str(z), x), axis=1)
    df['launch_continue_max_cnt'] = df['launch_list'].map(lambda x: max([len(y) for y in str(x).split('0')]) if '0' in str(x) else 16)
    App_launch_log_data['launch_continue_max_cnt'] = df['launch_continue_max_cnt']

    # 用户最大启动时间
    data = feature_data[['user_id', 'day']].copy()
    data=data.groupby(['user_id'])['day'].agg({'app_lau_user_id_max_day': np.max}).reset_index()
    App_launch_log_data=pd.merge(App_launch_log_data,data,on=['user_id'],how='left')

    # APP 用户最小启动时间
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'app_lau_user_id_min_day': np.min}).reset_index()
    App_launch_log_data = pd.merge(App_launch_log_data, data, on=['user_id'], how='left')

    # 用户距离标签最近出现次数
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'day': np.max}).reset_index()
    data_temp = feature_data[['user_id', 'day']].copy()
    data_temp = pd.merge(data_temp, data, on=['user_id', 'day'], how='inner')
    data_temp['app_lau_user_id_max_day_cnt'] = 1
    data_temp = data_temp.groupby(['user_id', 'day']).agg('sum').reset_index()
    del data_temp['day']
    App_launch_log_data = pd.merge(App_launch_log_data, data_temp, on=['user_id'], how='left')

    #  APP 用户最大启动时间-APP 用户最小启动时间
    App_launch_log_data['app_lau_user_id_max-min_day']=App_launch_log_data['app_lau_user_id_max_day']-App_launch_log_data['app_lau_user_id_min_day']
    return App_launch_log_data

def Video_create_log_Fun(label_data,feature_data):
    '''
    提取Video 拍摄日志相关特征
    :return: 
    '''
    Video_create_id = list(set(label_data['user_id']))
    Video_create_log_data = pd.DataFrame(Video_create_id, columns=['user_id'])

    # 用户的全部数据
    # 用户每天出现次数################################################33
    day_max = max(feature_data['day'])
    day_min = min(feature_data['day'])
    for i in range(day_min, day_max + 1):
        data = feature_data[['user_id', 'day']].copy()
        data = data[data['day'] == i]
        data['video_create_user_last_cnt' + str(i - day_min) + ''] = 1
        del data['day']
        data = data.groupby(['user_id']).agg('sum').reset_index()
        Video_create_log_data = pd.merge(Video_create_log_data, data, on=['user_id'], how='left')

    # 用户出现方差、总和、平均数、最大值、最小值
    needs = []
    for col in Video_create_log_data.columns.tolist():
        if 'video_create_user_last_cnt' in col:
            needs.append(col)
    Video_create_log_data.fillna(0, inplace=True)
    Video_create_log_data['video_create_user_cnt_var_cnt'] = Video_create_log_data[needs].var(1)
    Video_create_log_data['video_create_user_cnt_sum_cnt'] = Video_create_log_data[needs].sum(1)
    Video_create_log_data['video_create_user_cnt_avg_cnt'] = Video_create_log_data[needs].mean(1)
    Video_create_log_data['video_create_user_cnt_max_cnt'] = Video_create_log_data[needs].max(1)
    Video_create_log_data['video_create_user_cnt_min_cnt'] = Video_create_log_data[needs].min(1)

    # #用户登陆天数
    needs = []
    for col in Video_create_log_data.columns.tolist():
        if 'video_create_user_last_cnt' in col:
            needs.append(col)
    df = Video_create_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    needs = []
    for col in df.columns.tolist():
        if 'video_create_user_last_cnt' in col:
            needs.append(col)
    df['video_create_continue_sum_cnt'] = df[needs].sum(1)
    Video_create_log_data['video_create_continue_sum_cnt'] = df['video_create_continue_sum_cnt']

    # #用户连续登陆最大值
    needs = []
    for col in Video_create_log_data.columns.tolist():
        if 'video_create_user_last_cnt' in col:
            needs.append(col)
    df = Video_create_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    df['video_create_list'] = df.apply(lambda x: reduce(lambda y, z: str(y) + str(z), x), axis=1)
    df['video_create_continue_max_cnt'] = df['video_create_list'].map(lambda x: max([len(y) for y in str(x).split('0')]) if '0' in str(x) else 16)
    Video_create_log_data['video_create_continue_max_cnt'] = df['video_create_continue_max_cnt']

    # 用户最大启动时间
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'video_create_user_id_max_day': np.max}).reset_index()
    Video_create_log_data = pd.merge(Video_create_log_data, data, on=['user_id'], how='left')

    # APP 用户最小启动时间
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'video_create_user_id_min_day': np.min}).reset_index()
    Video_create_log_data = pd.merge(Video_create_log_data, data, on=['user_id'], how='left')

    # 用户距离标签最近出现次数
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'day': np.max}).reset_index()
    data_temp = feature_data[['user_id', 'day']].copy()
    data_temp = pd.merge(data_temp, data, on=['user_id', 'day'], how='inner')
    data_temp['video_create_user_id_max_day_cnt'] = 1
    data_temp = data_temp.groupby(['user_id', 'day']).agg('sum').reset_index()
    del data_temp['day']
    Video_create_log_data = pd.merge(Video_create_log_data, data_temp, on=['user_id'], how='left')

    #  APP 用户最大启动时间-APP 用户最小启动时间
    Video_create_log_data['video_create_user_id_max-min_day'] = Video_create_log_data['video_create_user_id_max_day'] - Video_create_log_data['video_create_user_id_min_day']

    return Video_create_log_data

def User_activity_log_Fun(label_data,feature_data):
    '''
    提取行为日志相关特征
    :return: 
    '''
    user_activity_id = list(set(label_data['user_id']))
    User_activity_log_data = pd.DataFrame(user_activity_id, columns=['user_id'])
    User_activity_page_temp = pd.DataFrame(user_activity_id, columns=['user_id'])
    User_activity_action_temp = pd.DataFrame(user_activity_id, columns=['user_id'])

    

    day_max = max(feature_data['day'])
    day_min = min(feature_data['day'])
    # 判断user_id ==author_id
    data = feature_data[['user_id', 'author_id']].copy()
    data['is_author_user'] = list(map(lambda x, y: 1 if x == y else 0, data['user_id'], data['author_id']))
    del data['author_id']
    data = data.sort_values(by='is_author_user', ascending=False)
    data = data.drop_duplicates(['user_id'])
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # activity中user_id一共看了不同的author_id多少次
    data = feature_data[['user_id', 'author_id']].copy()
    data['user_activity_user_id_author_id_dif_cnt'] = 1
    data = data.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    del data['author_id']
    data['user_activity_user_id_author_id_dif_cnt'] = 1
    data = data.groupby(['user_id']).agg('sum').reset_index()
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # activity中user_id一共看了不同的video_id多少次
    data = feature_data[['user_id', 'video_id']].copy()
    data['user_activity_user_id_video_id_dif_cnt'] = 1
    data = data.groupby(['user_id', 'video_id']).agg('sum').reset_index()
    del data['video_id']
    data['user_activity_user_id_video_id_dif_cnt'] = 1
    data = data.groupby(['user_id']).agg('sum').reset_index()
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # activity中user_id page 出现次数的 众数
    data = feature_data[['user_id', 'page']].copy()
    data['user_activity_user_id_page_cnt'] = 1
    data = data.groupby(['user_id', 'page']).agg('sum').reset_index()
    data = data.sort_values(by='user_activity_user_id_page_cnt', ascending=False)
    data = data.drop_duplicates(['user_id'])
    data.rename(columns={'page': 'user_activity_user_id_page_zhongshu'}, inplace=True)
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    ###page出现方差、总和、平均数、最大值、最小值#######################################################################
    for ii in range(day_min, day_max + 1):
        feature_datas = feature_data[['user_id', 'page', 'day']]
        feature_datas = feature_datas[feature_datas['day'] == ii]
        del feature_datas['day']
        for i in range(5):
            data = feature_datas[['user_id', 'page']]
            data = data[data['page'] == i]
            data = data.groupby(['user_id']).agg('count').reset_index()
            data.rename(columns={'page': 'user_activity_user_id_per_page_' + str(i) + '_cnt' + str(ii - day_min) + ''},inplace=True)
            User_activity_page_temp = pd.merge(User_activity_page_temp, data, on='user_id', how='left')


    User_activity_page_temp.fillna(0, inplace=True)
    for i in range(5):
        needs = []
        for col in User_activity_page_temp.columns.tolist():
            if 'user_activity_user_id_per_page_' + str(i) + '_cnt' in col:
                needs.append(col)
        User_activity_log_data.fillna(0, inplace=True)
        User_activity_log_data['user_activity_user_id_page_' + str(i) + '_cnt_var'] = User_activity_page_temp[needs].var(1)
        User_activity_log_data['user_activity_user_id_page_' + str(i) + '_cnt_sum'] = User_activity_page_temp[needs].sum(1)
        User_activity_log_data['user_activity_user_id_page_' + str(i) + '_cnt_mean'] = User_activity_page_temp[needs].mean(1)
        User_activity_log_data['user_activity_user_id_page_' + str(i) + '_cnt_max'] = User_activity_page_temp[needs].max(1)
        User_activity_log_data['user_activity_user_id_page_' + str(i) + '_cnt_min'] = User_activity_page_temp[needs].min(1)

    # activity中user_id action_type 出现次数的 众数
    data = feature_data[['user_id', 'action_type']].copy()
    data['user_activity_user_id_action_type_cnt'] = 1
    data = data.groupby(['user_id', 'action_type']).agg('sum').reset_index()
    data = data.sort_values(by='user_activity_user_id_action_type_cnt', ascending=False)
    data = data.drop_duplicates(['user_id'])
    data.rename(columns={'action_type': 'user_activity_user_id_action_type_zhongshu'}, inplace=True)
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # activity中user_id action_type每个类别出现的现次数#################################################3###############
    for ii in range(day_min, day_max + 1):
        feature_datas = feature_data[['user_id', 'action_type', 'day']]
        feature_datas = feature_datas[feature_datas['day'] == ii]
        del feature_datas['day']
        for i in range(6):
            data = feature_datas[['user_id', 'action_type']].copy()
            data = data[data['action_type'] == i]
            data = data.groupby(['user_id']).agg('count').reset_index()
            data.rename(columns={'action_type': 'user_activity_user_id_action_type_' + str(i) + '_cnt' + str(ii - day_min) + ''},inplace=True)
            User_activity_action_temp = pd.merge(User_activity_action_temp, data, on='user_id', how='left')

    # # action_type出现方差、总和、平均数、最大值、最小值
    User_activity_action_temp.fillna(0, inplace=True)
    for i in range(6):
        needs = []
        for col in User_activity_action_temp.columns.tolist():
            if 'user_activity_user_id_action_type_' + str(i) + '_cnt' in col:
                needs.append(col)
        User_activity_log_data.fillna(0, inplace=True)
        User_activity_log_data['user_activity_user_id_action_type_' + str(i) + '_cnt_var'] = User_activity_action_temp[needs].var(1)
        User_activity_log_data['user_activity_user_id_action_type_' + str(i) + '_cnt_sum'] = User_activity_action_temp[needs].sum(1)
        User_activity_log_data['user_activity_user_id_action_type_' + str(i) + '_cnt_mean'] = User_activity_action_temp[needs].mean(1)
        User_activity_log_data['user_activity_user_id_action_type_' + str(i) + '_cnt_max'] = User_activity_action_temp[needs].max(1)
        User_activity_log_data['user_activity_user_id_action_type_' + str(i) + '_cnt_min'] = User_activity_action_temp[needs].min(1)

    # #  #  用户的全部数据##############################################################################################
    # 用户每天出现次数
    for i in range(day_min, day_max + 1):
        data = feature_data[['user_id', 'day']].copy()
        data = data[data['day'] == i]
        data['user_activity_user_last_cnt' + str(i - day_min) + ''] = 1
        del data['day']
        data = data.groupby(['user_id']).agg('sum').reset_index()
        User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # 用户出现方差、总和、平均数、最大值、最小值
    needs = []
    for col in User_activity_log_data.columns.tolist():
        if 'user_activity_user_last_cnt' in col:
            needs.append(col)
    User_activity_log_data.fillna(0, inplace=True)
    User_activity_log_data['user_activity_user_cnt_var_cnt'] = User_activity_log_data[needs].var(1)
    User_activity_log_data['user_activity_user_cnt_sum_cnt'] = User_activity_log_data[needs].sum(1)
    User_activity_log_data['user_activity_user_cnt_avg_cnt'] = User_activity_log_data[needs].mean(1)
    User_activity_log_data['user_activity_user_cnt_max_cnt'] = User_activity_log_data[needs].max(1)
    User_activity_log_data['user_activity_user_cnt_min_cnt'] = User_activity_log_data[needs].min(1)

    # #用户登陆天数
    needs = []
    for col in User_activity_log_data.columns.tolist():
        if 'user_activity_user_last_cnt' in col:
            needs.append(col)
    df = User_activity_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    needs = []
    for col in df.columns.tolist():
        if 'user_activity_user_last_cnt' in col:
            needs.append(col)
    df['user_activity_continue_sum_cnt'] = df[needs].sum(1)
    User_activity_log_data['user_activity_continue_sum_cnt'] = df['user_activity_continue_sum_cnt']

    # #用户连续登陆最大值
    needs = []
    for col in User_activity_log_data.columns.tolist():
        if 'user_activity_user_last_cnt' in col:
            needs.append(col)
    df = User_activity_log_data[needs]
    df.fillna(0, inplace=True)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    df['user_activity_list'] = df.apply(lambda x: reduce(lambda y, z: str(y) + str(z), x), axis=1)
    df['user_activity_continue_max_cnt'] = df['user_activity_list'].map(
        lambda x: max([len(y) for y in str(x).split('0')]) if '0' in str(x) else 16)
    User_activity_log_data['user_activity_continue_max_cnt'] = df['user_activity_continue_max_cnt']

    # 用户最大启动时间
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'user_activity_user_id_max_day': np.max}).reset_index()
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # APP 用户最小启动时间
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'user_activity_user_id_min_day': np.min}).reset_index()
    User_activity_log_data = pd.merge(User_activity_log_data, data, on=['user_id'], how='left')

    # 用户距离标签最近出现次数
    data = feature_data[['user_id', 'day']].copy()
    data = data.groupby(['user_id'])['day'].agg({'day': np.max}).reset_index()
    data_temp = feature_data[['user_id', 'day']].copy()
    data_temp = pd.merge(data_temp, data, on=['user_id', 'day'], how='inner')
    data_temp['user_activity_user_id_max_day_cnt'] = 1
    data_temp = data_temp.groupby(['user_id', 'day']).agg('sum').reset_index()
    del data_temp['day']
    User_activity_log_data = pd.merge(User_activity_log_data, data_temp, on=['user_id'], how='left')

    #  APP 用户最大启动时间-APP 用户最小启动时间
    User_activity_log_data['user_activity_user_id_max-min_day'] = User_activity_log_data['user_activity_user_id_max_day'] - User_activity_log_data['user_activity_user_id_min_day']
    return User_activity_log_data

def Interactive_feature(feature_data):
    '''
    交互特征
    :param train_feature_data: 
    :param test_feature_data: 
    :return: 
    '''
    ###########1、注册日期和其他的日期的交互特征##################
    # 注册当天是否有启动
    feature_data['register_day_isTrue=app_lau_user_id_min_day'] = list(map(lambda x, y: 1 if x == y else 0, feature_data['app_lau_user_id_min_day'], feature_data['register_day']))
    # #注册当天是否有拍摄
    feature_data['register_day_isTrue=video_create_user_id_min_day'] = list(map(lambda x, y: 1 if x == y else 0, feature_data['video_create_user_id_min_day'], feature_data['register_day']))
    # 注册当天是否有活动
    feature_data['register_day_isTrue=user_activity_user_id_min_day'] = list(map(lambda x, y: 1 if x == y else 0, feature_data['user_activity_user_id_min_day'], feature_data['register_day']))

    # user_reg注册时间注册日期-app最大启动时间
    feature_data['app_lau_user_reg_min-regday']=feature_data['app_lau_user_id_min_day']-feature_data['register_day']
    # user_reg注册时间注册日期-app最小启动时间
    feature_data['app_lau_user_reg_max-regday'] = feature_data['app_lau_user_id_max_day'] - feature_data['register_day']
    # # user_reg注册时间注册日期-app中位数时间
    # feature_data['app_lau_user_reg_dedian-regday'] = feature_data['app_lau_user_id_day_dedian'] - feature_data['register_day']
    # # user_reg注册时间注册日期-app众数时间
    # feature_data['app_lau_user_reg_zhongshu-regday'] = feature_data['app_lau_user_id_day_zhongshu'] - feature_data['register_day']

    # user_reg注册时间注册日期-video_creat最大时间
    feature_data['video_create_user_reg_min-regday']=feature_data['video_create_user_id_min_day']-feature_data['register_day']
    # user_reg注册时间注册日期-video_create最小启动时间
    feature_data['video_create_user_reg_max-regday'] = feature_data['video_create_user_id_max_day'] - feature_data['register_day']
    # # user_reg注册时间注册日期-video_create中位数时间
    # feature_data['video_create_user_reg_dedian-regday'] = feature_data['video_create_user_id_day_dedian'] - feature_data['register_day']
    # # user_reg注册时间注册日期-video_create众数时间
    # feature_data['video_create_user_reg_zhongshu-regday'] = feature_data['video_create_user_id_day_zhongshu'] - feature_data['register_day']

    # user_reg注册时间注册日期-user_activity最大时间
    feature_data['user_activity_user_reg_min-regday']=feature_data['user_activity_user_id_min_day']-feature_data['register_day']
    # user_reg注册时间注册日期-user_activity最小启动时间
    feature_data['user_activity_user_reg_max-regday'] = feature_data['user_activity_user_id_max_day'] - feature_data['register_day']
    return feature_data

def xgb_for_te(train,test):
    result = test[['user_id']]

    drop_feature = ['user_id', 'register_day', 'app_lau_user_id_max_day', 'app_lau_user_id_min_day',
                    'video_create_user_id_max_day', 'video_create_user_id_min_day', 'user_activity_user_id_min_day',
                    'user_activity_user_id_max_day']
    train = train.drop(drop_feature, axis=1)
    test = test.drop(drop_feature, axis=1)
    test.fillna(-999, inplace=True)
    train.fillna(-999, inplace=True)

    train_y = train.label
    train_X = train.drop(['label'], axis=1)

    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_test = xgb.DMatrix(test)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'scale_pos_weight': 1,
        'min_child_weight': 18,
    }
    num_rounds = 240  # 迭代次数

    # training model
    model = xgb.train(params, xgb_train, num_rounds)

    # 测试集
    preds = model.predict(xgb_test)
    test_pre_y = pd.DataFrame(preds)
    result['predicted_score'] = test_pre_y
    result = result.sort_values(by='predicted_score', ascending=False)
    # 返回
    return result

if __name__ == '__main__':
    # 线上训练集、线上测试集
    print('模型4:')
    print('    预处理...')
    app_launch, user_activity, user_register, video_create = load_data()
    print('    构造训练集、测试集...', end='')
    train,test= get_dataset(app_launch, user_activity, user_register, video_create)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = xgb_for_te(train,test)
    print('完毕!')
    # 训练结果
    predict.to_csv(r'tmp/model_4.csv', index=False, header=None)