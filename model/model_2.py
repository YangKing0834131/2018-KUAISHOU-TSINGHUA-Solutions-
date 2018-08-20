# -*- coding: utf-8 -*-
"""
Created on Tue May 29 00:02:34 2018

@author: gxr
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn import preprocessing
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore") 

def load_data():
    dump_path = r'data/source_b.pkl'
    if os.path.exists(dump_path):
        source = pickle.load(open(dump_path,'rb'))
    else:
        # 源数据
        app_launch_log = pd.read_table('data/chusai_b_train/app_launch_log.txt',\
                                       delimiter = '	',\
                                       names = ['user_id','day'])
        user_activity_log = pd.read_table('data/chusai_b_train/user_activity_log.txt',\
                                          delimiter = '	',\
                                          names = ['user_id','day','page','video_id','author_id','action_type'])
        user_register_log = pd.read_table('data/chusai_b_train/user_register_log.txt',\
                                          delimiter = '	',\
                                          names = ['user_id','register_day','register_type','device_type'])
        video_create_log = pd.read_table('data/chusai_b_train/video_create_log.txt',\
                                         delimiter = '	',\
                                         names = ['user_id','day'])
        # 去除异常(貌似没有)
        app_launch_log = pd.merge(app_launch_log,user_register_log[['user_id','register_day']],on = 'user_id',how = 'left')
        user_activity_log = pd.merge(user_activity_log,user_register_log[['user_id','register_day']],on = 'user_id',how = 'left')
        video_create_log = pd.merge(video_create_log,user_register_log[['user_id','register_day']],on = 'user_id',how = 'left')
        app_launch_log['wrong'] = list(map(lambda x,y : 1 if x < y else 0,app_launch_log['day'],app_launch_log['register_day']))
        user_activity_log['wrong'] = list(map(lambda x,y : 1 if x < y else 0,user_activity_log['day'],user_activity_log['register_day']))
        video_create_log['wrong'] = list(map(lambda x,y : 1 if x < y else 0,video_create_log['day'],video_create_log['register_day']))
        app_launch_log = app_launch_log[app_launch_log['wrong'] == 0]
        user_activity_log = user_activity_log[user_activity_log['wrong'] == 0]
        video_create_log = video_create_log[video_create_log['wrong'] == 0]
        app_launch_log.drop(['register_day','wrong'],axis = 1,inplace = True)
        user_activity_log.drop(['register_day','wrong'],axis = 1,inplace = True)
        video_create_log.drop(['register_day','wrong'],axis = 1,inplace = True)
        ### 合一个大表
        # app_launch_log的['user_id','day']无重复
        app_launch_log['launch_index'] = range(len(app_launch_log))
        # user_activity_log的['user_id','day']有重复
        user_activity_log['activity_index'] = range(len(user_activity_log))
        # user_register_log的['user_id']无重复
        user_register_log['register_index'] = range(len(user_register_log))
        # video_create_log的['user_id','day']有重复
        video_create_log['video_index'] = range(len(video_create_log))
        ## 合表
        # 连接app_launch_log,user_activity_log,连接后activity_index有值的均无重复，因为app_launch_log的['user_id','day']无重复
        source = pd.merge(app_launch_log,user_activity_log,on = ['user_id','day'],how = 'outer')
        # 连接app_launch_log,user_activity_log,video_create_log,连接后activity_index和video_index均有重复，因为user_activity_log的['user_id','day']和video_create_log的['user_id','day']均有重复
        source = pd.merge(source,video_create_log,on = ['user_id','day'],how = 'outer')
        # 连接app_launch_log,user_activity_log,video_create_log,user_register_log,左连接
        source = pd.merge(source,user_register_log,on = ['user_id'],how = 'outer')
        # 标记是否为注册当天
        source['register_behavior'] = list(map(lambda x,y : 1 if x == y else 0,source['day'],source['register_day']))
        # 空值填-1
        source.fillna(-1,downcast = 'infer',inplace = True)
        # 保存
#        pickle.dump(source,open(dump_path,'wb'))
    # 返回
    return source

def get_label(source,label_dates):
    label = source[source['day'].map(lambda x : x in label_dates)][['user_id']].drop_duplicates('user_id',keep = 'first')
    label['label'] = 1
    return label

def get_launch_feat(source,feat_dates):
    history = source[source['launch_index'] != -1][source['day'].map(lambda x : x in feat_dates)].drop_duplicates(['user_id','day','launch_index'],keep = 'first')
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    # 启动数avg
    pivot = pd.pivot_table(history,index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
    pivot.rename(columns = {'cnt' : 'user_launch_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 最后一天启动数
    pivot = pd.pivot_table(history[history['day'] == feat_dates[-1]],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_launch_cnt_on_last_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 注册当天启动数
    pivot = pd.pivot_table(history[history['register_behavior'] == 1],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_launch_cnt_on_register_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 只启动无活动数avg
    pivot = pd.pivot_table(history[history['activity_index'] == -1],index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
    pivot.rename(columns = {'cnt' : 'user_launch_cnt_of_not_activity_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 最近/远一次启动距离最近考察日的时间间隔
    near = 'nearest_day_launch'
    fur = 'furest_day_launch'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    # 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    # 返回
    return feature

def get_activity_feat(source,feat_dates):
    history = source[source['activity_index'] != -1][source['day'].map(lambda x : x in feat_dates)].drop_duplicates(['user_id','day','activity_index'],keep = 'first')
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    
    ### feat_dates总行为
    ## 总行为数avg
    pivot = pd.pivot_table(history,index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
    pivot.rename(columns = {'cnt' : 'user_activity_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    ## 在各种不同页面的行为和不同类型的行为及其交互avg
    ## (关注页page = 0,个人主页page = 1,发现页page = 2,同城页page = 3,其他页page = 4)
    ## (播放action_type = 0,关注action_type = 1,点赞action_type = 2,转发action_type = 3,举报action_type = 4,减少此类作品action_type = 5)
    for i in [0,1,2,3,4]:
        pivot = pd.pivot_table(history[history['page'] == i],index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_in_page_' + str(i)},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    for i in [0,1,2,3,4,5]:
        pivot = pd.pivot_table(history[history['action_type'] == i],index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_of_action_type_' + str(i)},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    
            
    ### 最后一天行为
    ## 总行为数
    pivot = pd.pivot_table(history[history['day'] == feat_dates[-1]],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_activity_cnt_on_last_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    ## 在各种不同页面的行为和不同类型的行为及其交互
    for i in [0,1,2,3,4]:
        pivot = pd.pivot_table(history[(history['day'] == feat_dates[-1]) & (history['page'] == i)],index = ['user_id'],values = 'cnt',aggfunc = len)
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_in_page_' + str(i) + '_on_last_date'},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    for i in [0,1,2,3,4,5]:
        pivot = pd.pivot_table(history[(history['day'] == feat_dates[-1]) & (history['action_type'] == i)],index = ['user_id'],values = 'cnt',aggfunc = len)
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_of_action_type_' + str(i) + '_on_last_date'},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
            
    ### 注册当天行为
    ## 总行为数
    pivot = pd.pivot_table(history[history['register_behavior'] == 1],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_activity_cnt_on_register_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    ## 在各种不同页面的行为和不同类型的行为及其交互
    for i in [0,1,2,3,4]:
        pivot = pd.pivot_table(history[(history['register_behavior'] == 1) & (history['page'] == i)],index = ['user_id'],values = 'cnt',aggfunc = len)
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_in_page_' + str(i) + '_on_register_date'},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    for i in [0,1,2,3,4,5]:
        pivot = pd.pivot_table(history[(history['register_behavior'] == 1) & (history['action_type'] == i)],index = ['user_id'],values = 'cnt',aggfunc = len)
        if len(pivot) == 0:
            pivot['cnt'] = 0
        pivot.rename(columns = {'cnt' : 'user_activity_cnt_of_action_type_' + str(i) + '_on_register_date'},inplace = True)
        pivot.reset_index(inplace = True)
        feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    
    ### 只活动无启动avg
    pivot = pd.pivot_table(history[history['launch_index'] == -1],index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
    pivot.rename(columns = {'cnt' : 'user_activity_cnt_of_not_launch_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    
    # 最近/远一次活动距离最近考察日的时间间隔
    near = 'nearest_day_activity'
    fur = 'furest_day_activity'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    
    # 最近/远一次page活动距离最近考察日的时间间隔
    for i in [0,1,2,3,4]:
        near = 'nearest_day_page_' + str(i)
        fur = 'furest_day_page_' + str(i)
        pivot_n = pd.pivot_table(history[history['page'] == i],index = ['user_id'],values = 'day',aggfunc = max)
        pivot_n.rename(columns = {'day' : near},inplace = True)
        pivot_n.reset_index(inplace = True)
        pivot_f = pd.pivot_table(history[history['page'] == i],index = ['user_id'],values = 'day',aggfunc = min)
        pivot_f.rename(columns = {'day' : fur},inplace = True)
        pivot_f.reset_index(inplace = True)
        feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
        feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
        feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
        feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
        feature.drop([near,fur],axis = 1,inplace = True)
    # 最近/远一次action_type活动距离最近考察日的时间间隔
    for i in [0,1,2,3,4,5]:
        near = 'nearest_day_action_type_' + str(i)
        fur = 'furest_day_action_type_' + str(i)
        pivot_n = pd.pivot_table(history[history['action_type'] == i],index = ['user_id'],values = 'day',aggfunc = max)
        pivot_n.rename(columns = {'day' : near},inplace = True)
        pivot_n.reset_index(inplace = True)
        pivot_f = pd.pivot_table(history[history['action_type'] == i],index = ['user_id'],values = 'day',aggfunc = min)
        pivot_f.rename(columns = {'day' : fur},inplace = True)
        pivot_f.reset_index(inplace = True)
        feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
        feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
        feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
        feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
        feature.drop([near,fur],axis = 1,inplace = True)
    
            
    # 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    # 返回
    return feature

def get_video_feat(source,feat_dates):
    history = source[source['video_index'] != -1][source['day'].map(lambda x : x in feat_dates)].drop_duplicates(['user_id','day','video_index'],keep = 'first')
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    # 拍摄数avg
    pivot = pd.pivot_table(history,index = ['user_id'],values = 'cnt',aggfunc = lambda x : len(x) / len(feat_dates))
    pivot.rename(columns = {'cnt' : 'user_video_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 最后一天拍摄数
    pivot = pd.pivot_table(history[history['day'] == feat_dates[-1]],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_video_cnt_on_last_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 注册当天拍摄数
    pivot = pd.pivot_table(history[history['register_behavior'] == 1],index = ['user_id'],values = 'cnt',aggfunc = len)
    pivot.rename(columns = {'cnt' : 'user_video_cnt_on_register_date'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = ['user_id'],how = 'outer')
    # 最近/远一次拍摄距离最近考察日的时间间隔
    near = 'nearest_day_video'
    fur = 'furest_day_video'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    # 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    # 返回
    return feature

def get_register_feat(source,feat_dates):
    history = source[source['register_index'] != -1][source['day'].map(lambda x : x in feat_dates)].drop_duplicates(['user_id','register_day','register_index'],keep = 'first')
    # 返回的特征
    feature = history[['user_id','register_day']]
    # 注册日据最近考察日间隔
    feature['label_sub_register'] = feature['register_day'].map(lambda x : feat_dates[-1] + 1 - x)
    # 删不需要的
    feature.drop(['register_day'],axis = 1,inplace = True)
    # 返回
    return feature

def get_base_feat(source):
    # 返回的特征
    feature = source[source['register_index'] != -1].drop_duplicates(['user_id','register_day','register_index'],keep = 'first')[['user_id','register_type','device_type']]
    # 离散register_type
    df = pd.get_dummies(feature['register_type'],prefix = 'register_type')
    feature = pd.concat([feature,df],axis = 1)
    # 联合register_type和device_type编码
    feature['register_type_device_type'] = list(map(lambda x,y : str(x) + '_' + str(y),feature['register_type'],feature['device_type']))
    le = preprocessing.LabelEncoder()
    le.fit(feature['register_type_device_type'])
    feature['register_type_device_type'] = le.transform(feature['register_type_device_type'])
#    # 离散device_type
#    df = pd.get_dummies(feature['device_type'],prefix = 'device_type')
#    feature = pd.concat([feature,df],axis = 1)
    # 返回
    return feature


def create_train(launch_feat,activity_feat,video_feat,register_feat,base_feat,label):
    train = pd.merge(launch_feat,activity_feat,on = ['user_id'],how = 'outer')
    train = pd.merge(train,video_feat,on = ['user_id'],how = 'outer')
    train = pd.merge(train,register_feat,on = ['user_id'],how = 'outer')
    train = pd.merge(train,base_feat,on = ['user_id'],how = 'left')
    train = pd.merge(train,label,on = ['user_id'],how = 'left')
    train.fillna(0,downcast = 'infer',inplace = True)
    return train

def create_test(launch_feat,activity_feat,video_feat,register_feat,base_feat):
    test = pd.merge(launch_feat,activity_feat,on = ['user_id'],how = 'outer')
    test = pd.merge(test,video_feat,on = ['user_id'],how = 'outer')
    test = pd.merge(test,register_feat,on = ['user_id'],how = 'outer')
    test = pd.merge(test,base_feat,on = ['user_id'],how = 'left')
    test.fillna(0,downcast = 'infer',inplace = True)
    return test
    
def get_dataset(source):
    base_feat = get_base_feat(source)
    # off_tr
    off_tr = pd.DataFrame()
    i = 0
    print(str(round(i/18*100)) + '%...',end = '')
    i += 1
    for start in [17,16,15,14,13,12,11,10,9,8]:
        label_dates = list(range(start,start + 7)) # s,s+1,s+2,s+3,s+4,s+5,s+6
        feat_dates = list(range(1,start)) # s-7,s-6,s-5,s-4,s-3,s-2,s-1
        label = get_label(source,label_dates)
        launch_feat = get_launch_feat(source,feat_dates)
        activity_feat = get_activity_feat(source,feat_dates)
        video_feat = get_video_feat(source,feat_dates)
        register_feat = get_register_feat(source,feat_dates)
        tr = create_train(launch_feat,activity_feat,video_feat,register_feat,base_feat,label)
        off_tr = pd.concat([off_tr,tr],axis = 0)
        print(str(round(i/18*100)) + '%...',end = '')
        i += 1
    # va
    start = 24
    label_dates = list(range(start,start + 7)) # s,s+1,s+2,s+3,s+4,s+5,s+6
    feat_dates = list(range(1,start)) # s-7,s-6,s-5,s-4,s-3,s-2,s-1
    label = get_label(source,label_dates)
    launch_feat = get_launch_feat(source,feat_dates)
    activity_feat = get_activity_feat(source,feat_dates)
    video_feat = get_video_feat(source,feat_dates)
    register_feat = get_register_feat(source,feat_dates)
    va = create_train(launch_feat,activity_feat,video_feat,register_feat,base_feat,label)
    print(str(round(i/18*100)) + '%...',end = '')
    i += 1
    # on_tr
    on_tr = pd.DataFrame()
    for start in [23,22,21,20,19,18]:
        label_dates = list(range(start,start + 7)) # s,s+1,s+2,s+3,s+4,s+5,s+6
        feat_dates = list(range(1,start)) # s-7,s-6,s-5,s-4,s-3,s-2,s-1
        label = get_label(source,label_dates)
        launch_feat = get_launch_feat(source,feat_dates)
        activity_feat = get_activity_feat(source,feat_dates)
        video_feat = get_video_feat(source,feat_dates)
        register_feat = get_register_feat(source,feat_dates)
        tr = create_train(launch_feat,activity_feat,video_feat,register_feat,base_feat,label)
        on_tr = pd.concat([on_tr,tr],axis = 0)
        print(str(round(i/18*100)) + '%...',end = '')
        i += 1
    on_tr = pd.concat([va,on_tr,off_tr],axis = 0)
#    on_tr = va.copy()
    # te
    start = 31
    feat_dates = list(range(1,start)) # s-7,s-6,s-5,s-4,s-3,s-2,s-1
    launch_feat = get_launch_feat(source,feat_dates)
    activity_feat = get_activity_feat(source,feat_dates)
    video_feat = get_video_feat(source,feat_dates)
    register_feat = get_register_feat(source,feat_dates)
    te = create_test(launch_feat,activity_feat,video_feat,register_feat,base_feat)
    print(str(round(i/18*100)) + '%')
    i += 1
#    # save
#    pickle.dump(off_tr,open('tmp/off_train_2.pkl','wb'),protocol = 4)
#    pickle.dump(va,open('tmp/off_validate_2.pkl','wb'),protocol = 4)
#    pickle.dump(on_tr,open('tmp/on_train_2.pkl','wb'),protocol = 4)
#    pickle.dump(te,open('tmp/on_test_2.pkl','wb'),protocol = 4)
    # 返回
    return off_tr,va,on_tr,te

def xgb_for_va(off_tr,va):
    train = off_tr.copy()
    validate = va.copy()
    
    train_y = train['label'].values
    train_x = train.drop(['user_id','label'],axis=1).values
    validate_x = validate.drop(['user_id','label'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalidate = xgb.DMatrix(validate_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'error',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2 3
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=400)
    # 预测
    predict = bst.predict(dvalidate)
    validate_xy = validate[['user_id']]
    validate_xy['predicted_score'] = predict
    validate_xy.sort_values(['predicted_score'],ascending = False,inplace = True)
    # 返回
    return validate_xy

def xgb_for_te(on_tr,te):
    train = on_tr.copy()
    test = te.copy()
    
    train_y = train['label'].values
    train_x = train.drop(['user_id','label'],axis=1).values
    test_x = test.drop(['user_id'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'error',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2 3
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=400)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['user_id']]
    test_xy['predicted_score'] = predict
    test_xy.sort_values(['predicted_score'],ascending = False,inplace = True)
    # 返回
    return test_xy

if __name__ == '__main__':
    # 线下训练集、线下测试集、线上训练集、线上测试集
    print('模型2:')
    print('    预处理...')
    source = load_data()
    print('    构造训练集、测试集...',end = '')
    off_tr,va,on_tr,te = get_dataset(source)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = xgb_for_te(on_tr,te)
    print('完毕!')
    # 训练结果
    predict.to_csv(r'tmp/model_2.csv',index = False,header = None)