# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:53:32 2019

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy import sparse
from lightgbm.sklearn import LGBMClassifier
from scipy.stats import kurtosis
import time
import warnings
import gc
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
def com(df):
    l=df.keys()[df.dtypes=='int64']
    for i in l:
        df[i]=df[i].astype('int32')
    l=df.keys()[df.dtypes=='float64']
    for i in l:
        df[i]=df[i].astype('float32')
    gc.collect()
    print(df.info())
    return df

t = time.time()
###############################训练集转化为按天的
train_df = pd.read_csv('../data/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])
train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(
    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1
)
train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
newlist=pd.read_csv('newlist.csv',parse_dates=['auditing_date', 'due_date', 'repay_date'])
newlist=newlist.merge(train_df[['user_id', 'listing_id', 'auditing_date', 'due_date', 'due_amt']],on=['user_id', 'listing_id', 'auditing_date', 'due_date'],how='left')
newlist=newlist.merge(train_df[['user_id', 'listing_id', 'auditing_date', 'due_date', 'repay_date','repay_amt']],on=['user_id', 'listing_id', 'auditing_date', 'due_date', 'repay_date'],how='left')
print(newlist.repay_amt.sum())
newlist['l1'] = (newlist['due_date'] - newlist['repay_date']).dt.days
g=newlist.loc[newlist['repay_amt']==0]
g=g[['user_id', 'listing_id']]
g['t']=1
newlist=newlist.merge(g,on=['user_id', 'listing_id'],how='left')
newlist.loc[(newlist['l1'] < 0)&(newlist['t']==1), 'repay_amt'] = newlist.loc[(newlist['l1'] < 0)&(newlist['t']==1), 'due_amt']
print(newlist.repay_amt.sum())
newlist['repay_amt']=newlist['repay_amt'].fillna(0)
newlist.pop('t')
newlist.pop('l1')
train_df=newlist
del newlist
gc.collect()
##########################################整理训练集标签
amt_labels = train_df['repay_amt'].values
clf_labels = (train_df['repay_amt']>0).astype(int).values
train_due_amt_df = train_df[['due_amt']]
train_num = train_df.shape[0]
del  train_df['repay_amt']
############################################测试集加入逾期日期
test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
tc=test_df.copy()
test_df=test_df.merge(sub_example[['listing_id',  'repay_date']],on=['listing_id'],how='right')
tc['repay_date']=tc['due_date']+np.timedelta64(1 ,'D')
test_df = pd.concat([test_df, tc], axis=0, ignore_index=True)
#########################################################
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
df['l1'] = (df['due_date'] - df['repay_date']).dt.days#距离最后还款日
df['l2'] = (df['repay_date']-df['auditing_date']).dt.days#距离成交日
df.loc[df['l1'] <0,'l2']=32#逾期处理
df['adays']=(df['due_date'] -df['auditing_date']).dt.days#总日期
gx=df[['user_id', 'listing_id','repay_date']].copy()
hdays=pd.read_table('holidays_cn.txt',parse_dates=['date'])#节日
hdays['xiu']=hdays['holiday']!='no'
hdays['repay_date']=hdays['date']
df=df.merge(hdays[['repay_date','xiu']],on='repay_date',how='left')
###########################################筛选数据集，只用了后1/3
df=df.iloc[train_num-1*(train_num//3):]
clf_labels=clf_labels[train_num-1*(train_num//3):]
train_due_amt_df=train_due_amt_df.iloc[train_num-1*(train_num//3):]
amt_labels=amt_labels[train_num-1*(train_num//3):]
train_num=train_num//3

gx=df[['user_id', 'listing_id','repay_date']].copy()
###############################################################
i=0
df['last'+str(i)+'dayofweek']=(df['repay_date']).dt.dayofweek#当天星期几
df['last'+str(i)+'dayofmonth']=(df['repay_date']).dt.day#当天几号
df['last'+str(i)+'weekofmonth']=(df['repay_date']).dt.day//7#当天第几周
df['im']=df['l1']<7#是否最后一周
df['eweek']=df['l2']//7#第几周

xz=[]
xz.append('last'+str(i)+'dayofweek')
xz.append('last'+str(i)+'dayofmonth')
xz.append('last'+str(i)+'weekofmonth')
###########################结合listing数据
listing_info_df = pd.read_csv('../data/listing_info.csv', parse_dates=['auditing_date'])
listing_info_df['overpay']=listing_info_df['rate']*listing_info_df['principal']
#listing_info_df['all']=listing_info_df['rate']*listing_info_df['principal']*listing_info_df['term']
del listing_info_df['user_id'], listing_info_df['auditing_date']
df = df.merge(listing_info_df, on='listing_id', how='left')
print(1)

def get_age_bin(age):
    if age <= 18:
        return 'age<=18'
    elif age <= 22:
        return '18<age<=22'
    elif age <= 26:
        return '22<age<=26'
    elif age <= 30:
        return '26<age<=30'
    elif age <= 35:
        return '30<age<=35'
    elif age <= 40:
        return '35<age<=40'
    elif age <= 50:
        return '40<age<=50'
    else:
        return '50<age<=90'
#############user信息
user_info_df = pd.read_csv('../data/user_info.csv', parse_dates=['reg_mon', 'insertdate'])
user_info_df.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)
user_info_df['age_bin'] = user_info_df['age'].apply(lambda age:get_age_bin(age))#年龄分段
user_info_df['city_age']=user_info_df['id_city'].astype(str)+user_info_df['age_bin'].astype(str)#年龄分段交叉

g=user_info_df.groupby('user_id').size().reset_index(name='us')
user_info_df=user_info_df.merge(g,on='user_id',how='left')
#g=user_info_df.groupby(['user_id','cell_province']).nunique().reset_index(name='ucn')
#user_info_df=user_info_df.merge(g,on='user_id',how='left')
user_info_df['uec']=user_info_df['cell_province']==user_info_df['id_province']#电话和城市是否匹配
user_info_df = user_info_df.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
df = df.merge(user_info_df, on='user_id', how='left')


# tag信息
user_tag_df = pd.read_csv('../data/user_taglist.csv', parse_dates=['insertdate'])
user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
g=user_tag_df.groupby('user_id').size().reset_index(name='uts')
user_tag_df=user_tag_df.merge(g,on='user_id',how='left')
g=user_tag_df.groupby('user_id')['taglist'].sum().reset_index(name='taglist')#######多条合并
user_tag_df = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
user_tag_df.pop('taglist')
user_tag_df=user_tag_df.merge(g,on='user_id',how='left')
df = df.merge(user_tag_df, on='user_id', how='left')
##################这部分其实没用上
print(2)
# 历史记录表能做的特征远不止这些

listing_info_df = pd.read_csv('../data/listing_info.csv', parse_dates=['auditing_date'])
listing_info_df['overpay']=listing_info_df['rate']*listing_info_df['principal']
#listing_info_df['all']=listing_info_df['rate']*listing_info_df['principal']*listing_info_df['term']
del listing_info_df['user_id'], 
#################重点特征部分
repay_log_df = pd.read_csv('../data/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])

repay_log_df=repay_log_df.merge(listing_info_df,on='listing_id',how='left')

repay_log_df=repay_log_df.loc[repay_log_df['due_date'].dt.year!=2020]
#############################################给repay做标签来匹配数据集
repay_log_df['w']=repay_log_df['repay_date'].dt.dayofweek
repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days
repay_log_df['first']=repay_log_df['due_amt']>5000
repay_log_df['second']=(repay_log_df['due_amt']>1000)&(repay_log_df['due_amt']<=5000)

repay_log_df['l1'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days
repay_log_df['l2'] = (repay_log_df['repay_date'] - repay_log_df['auditing_date']).dt.days
print(repay_log_df['l1'].min())
repay_log_df.loc[repay_log_df['l1'] <0,'l1'] =-1
repay_log_df.loc[repay_log_df['l1'] <0,'l2']=32

print(repay_log_df['l1'].min())
repay_log_df['im']=repay_log_df['l1']<7
repay_log_df['eweek']=repay_log_df['l2']//7
repay_log_df['dd'] = (repay_log_df['due_date'] - repay_log_df['auditing_date']).dt.days
df['dd'] = (df['due_date'] - df['auditing_date']).dt.days
##########################################################
rpf=['w','first','second','l1','im','eweek','l2','dd']

r=repay_log_df.loc[(repay_log_df['early_repay_days']<=31)].copy()
def repayf(df,r,f,p,flag):
    #提取rate特征，具体参考ppt
    t = time.time()
    
    if flag==0:
        r['due']=r['early_repay_days']==0#最后一天还款
        gr=r.groupby(f)['due'].agg({f+p+'dsum':'sum',f+p+'dsize':'size'}).reset_index()
        gr[f+p+'drate']=(0.0000001+gr[f+p+'dsum'])/(0.0000001+gr[f+p+'dsize'])
        gr=gr.loc[gr[f+p+'dsize']>4]#因为没详细清洗数据，使用阈值来平滑
        
        df=df.merge(gr,on=f,how='left')
        r['late']=r['early_repay_days']<0#逾期
        gr=r.groupby(f)['late'].agg({f+p+'latesum':'sum',f+p+'latesize':'size'}).reset_index()
        gr[f+p+'laterate']=(0.0000001+gr[f+p+'latesum'])/(0.0000001+gr[f+p+'latesize'])
        gr=gr.loc[gr[f+p+'latesize']>4]
        
        df=df.merge(gr,on=f,how='left')
        r['tooearly']=r['early_repay_days']>24#头几天还款
        gr=r.groupby(f)['tooearly'].agg({f+p+'tooearlysum':'sum',f+p+'tooearlysize':'size'}).reset_index()
        gr[f+p+'tooearlyrate']=(0.0000001+gr[f+p+'tooearlysum'])/(0.0000001+gr[f+p+'tooearlysize'])
        gr=gr.loc[gr[f+p+'tooearlysize']>4]
        
        df=df.merge(gr,on=f,how='left')
     
    r['day']=r['repay_date'].dt.day
    r['wom']=r['repay_date'].dt.day//7
    xr=r.copy()
    ####################按天统计
    g=xr.groupby([f,'w']).size().reset_index(name='ws')
    #repay_log_df=repay_log_df.merge(g ,on=['user_id','w'],how='left')
    g1=r.groupby(f).size().reset_index(name='us')
    g=g.merge(g1 ,on=f,how='right')
    g['us'].fillna(0.0001)
    g['ws']=g['ws']-1
    g['us']=g['us']-1
    g['wus']=(0.000001+g['ws'])/(0.000001+g['us'])
    g=g.loc[g['us']>4]
    for i in range(1):
        gg=g.copy()
        gg.pop('us')
        gg.columns=[f,'last'+str(i)+'dayofmonth',f+p+'last'+str(i)+'daymonthtrickws',f+p+'last'+str(i)+'daymonthtrickwus']
        
        df=df.merge(gg,on=[f,'last'+str(i)+'dayofmonth'],how='left')
    #####################按星期几
    g=xr.groupby([f,'day']).size().reset_index(name='ws')
    g1=r.groupby(f).size().reset_index(name='us')
    g=g.merge(g1 ,on=f,how='right')
    g['us'].fillna(0.0001)
    g['ws']=g['ws']-1
    g['us']=g['us']-1
    g['wus']=(0.000001+g['ws'])/(0.000001+g['us'])
    g=g.loc[g['us']>4]
    for i in range(1):
        gg=g.copy()
        gg.pop('us')
        gg.columns=[f,'last'+str(i)+'dayofweek',f+p+'last'+str(i)+'daytrickws',f+p+'last'+str(i)+'daytrickwus']
        df=df.merge(gg,on=[f,'last'+str(i)+'dayofweek'],how='left')
    ############################按第几周
    g=xr.groupby([f,'wom']).size().reset_index(name='ws')
    g1=r.groupby(f).size().reset_index(name='us')
    g=g.merge(g1 ,on=f,how='right')
    g['us'].fillna(0.0001)
    g['ws']=g['ws']-1
    g['us']=g['us']-1
    g['wus']=(0.000001+g['ws'])/(0.000001+g['us'])
    g=g.loc[g['us']>4]
    for i in range(1):
        gg=g.copy()
        gg.pop('us')
        gg.columns=[f,'last'+str(i)+'weekofmonth',f+p+'last'+str(i)+'womtrickws',f+p+'last'+str(i)+'womtrickwus']
        df=df.merge(gg,on=[f,'last'+str(i)+'weekofmonth'],how='left')
    print(f)
    print('runtime: {}\n'.format(time.time() - t))
    
    gc.collect()
    df=com(df)
    return df
df['a']=df['age']//10
df=repayf(df,r,'user_id','',0)
c=list(df.keys())
print('count',c.count('user_idlast0daymonthtrickws'))
a=test_df.auditing_date.min()
r=r.loc[((r['repay_date'].astype('str')=='2200-01-01')&(r['due_date']<a))|((r['repay_date'].astype('str')!='2200-01-01')&(r['repay_date']<a))]
############根据faq筛选
r=r.merge(user_info_df,on='user_id',how='left')
df=repayf(df,r,'id_city','',0)
df=repayf(df,r,'l2','',1)
df=repayf(df,r,'l1','',1)


########针对第一期再筛选
repay_log_df = repay_log_df[repay_log_df['order_id'] == 1].reset_index(drop=True)
r=repay_log_df.loc[(repay_log_df['early_repay_days']<=31)].copy()
df=repayf(df,r,'user_id','j',0)
r=r.loc[((r['repay_date'].astype('str')=='2200-01-01')&(r['due_date']<a))|((r['repay_date'].astype('str')!='2200-01-01')&(r['repay_date']<a))]
r=r.merge(user_info_df,on='user_id',how='left')
r['a']=r['age']//10
df=repayf(df,r,'id_city','j',0)
df=repayf(df,r,'l2','j',1)

df=repayf(df,r,'dd','j',0)
df=repayf(df,r,'l1','j',1)
df=repayf(df,r,'im','j',1)
df=repayf(df,r,'eweek','j',1)



df=repayf(df,r,'rate','j',0)
df=repayf(df,r,'term','j',0)
####################################################################以上是全部rate特征
print(3)
#################################################################repay基于user的基础统计
repay_log_df['repay'] = repay_log_df['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days
repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
for f in ['listing_id', 'order_id', 'due_date', 'repay_date','auditing_date', 'repay_amt']:
    del repay_log_df[f]
group = repay_log_df.groupby('user_id', as_index=False)
repay_log_df = repay_log_df.merge(
    group['repay'].agg({'repay_mean': 'mean','repay_usize': 'size'}), on='user_id', how='left'
)
repay_log_df = repay_log_df.merge(
    group['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum','early_repay_days_n': 'nunique',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left'
)
repay_log_df = repay_log_df.merge(
    group['due_amt'].agg({
        'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
        'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
        'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
    }), on='user_id', how='left'
)
for i in rpf:
    repay_log_df.pop(i)
del repay_log_df['repay'], repay_log_df['early_repay_days'], repay_log_df['due_amt']
repay_log_df = repay_log_df.drop_duplicates('user_id').reset_index(drop=True)
for i in repay_log_df.copy():
    if i in df.keys() and i!='user_id':
        repay_log_df.pop(i)
    
df = df.merge(repay_log_df, on='user_id', how='left')
###########################################行为统计
user_bh=pd.read_csv('../data/user_behavior_logs.csv')
g=user_bh.groupby(['user_id'])['behavior_type'].agg({'bhsize':'size','bhuni':'nunique'})
gu=user_bh.groupby(['user_id','behavior_type'])['behavior_time'].size().to_dict()
df=df.merge(g,on='user_id',how='left')
bh=[]
for i in range(1,4):
    df['bh'+str(i)+'count']=df['user_id'].apply(lambda x:gu.get((x,i),0))
    df['bhtype'+str(i)]=df['bh'+str(i)+'count']>0
    bh.append('bh'+str(i)+'count')
df['bhsum']=df[bh].sum(axis=1)
for i in range(1,4):
    df['bh'+str(i)+'perc']=df['bh'+str(i)+'count']/df['bhsum']

#########################去除重复项
for i in list(df.keys().copy()):
    if '_y' in i:
        df.pop(i)
        df[i[:-2]]=df[i[:-1]+'x']
        df.pop(i[:-1]+'x')
##############################################年龄城市天数组合金额
df['city_age_l1']=df['id_city'].astype(str)+df['age_bin'].astype(str)+df['l1'].astype(str)

city_age_bin_mapping = df.groupby('city_age_l1').agg({'principal':'mean','due_amt':'mean'})
df['mean_principal_by_city_age_l1'] = df['city_age_l1'].map(city_age_bin_mapping['principal'])
df['principal_over_by_city_age_l1'] = df['principal'] / df['mean_principal_by_city_age_l1']
df['mean_due_amt_by_city_age_l1'] = df['city_age_l1'].map(city_age_bin_mapping['due_amt'])
df['due_amt_over_by_city_age_l1'] = df['due_amt'] /df['mean_due_amt_by_city_age_l1']
df.drop(['mean_principal_by_city_age_l1','mean_due_amt_by_city_age_l1'],inplace=True,axis=1)

df['due_amt_per_days'] = df['due_amt'] / (df['due_date'] - df['auditing_date']).dt.days
df['due_amt_per_sdays'] = df['due_amt'] / np.sqrt((df['due_date'] - df['auditing_date']).dt.days)
w=df['due_amt_per_days'].copy()
###################################################日期统计
date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']
for f in date_cols:
    if f in ['reg_mon', 'info_insert_date', 'tag_insert_date']:
        df[f + '_year'] = df[f].dt.year
    
    df[f + '_month'] = df[f].dt.month
    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
df.drop(columns=date_cols, axis=1, inplace=True)



del  df['user_id'],df['listing_id']
print('runtime: {}\n'.format(time.time() - t))
df['nantag']=df['taglist'].isnull()#是否有tag
#df.to_csv('dft2'+'.csv', index=False)
print(4)
#################################标签化数据编码
cate_cols = ['gender', 'cell_province', 'id_province', 'id_city','age_bin', 'city_age']
for f in cate_cols:
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')

for f in [ 'city_age_l1']:
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')
del df['repay_date']
del df['taglist']
df=df.astype('float32')

df.to_hdf('litedft6'+'.h5', key='df', mode='w')


###############################平均数编码特征，具体可参考ppt
def get_stratifiedkfold_ids(x,y,n_folds=5,random_state=42,shuffle=True):
    kfold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    fold = kfold.split(x, y)
    fold_ids = []
    for k, (train_in, test_in) in enumerate(fold):
        fold_ids.append([train_in,test_in])
    return fold_ids
def mean_encoding_feature_label(train,test,col_encode,col_label,alpha=10):
    for k in ['l1','l2']:
        for label in [1]:
            new_col_name = col_encode +k+ '_' + str(label) + '_mean_encoding'
            train[col_encode+'_'+k]=train[col_encode].astype(str)+train[k].astype(str)
       
            test[col_encode+'_'+k]=test[col_encode].astype(str)+test[k].astype(str)

    
            target_global_mean = train[train[col_label] == label][col_label].count() / train[col_label].count()
            #cat_count = train[train[col_label]==label].groupby(col_encode+k)[col_encode+k].count()
            
            fenzi = train[train[col_label]==label].groupby(col_encode+'_'+k).size()
            fenmu = train.groupby(col_encode+'_'+k).size()
       
            
            mean_encoding_mapping =(fenzi/fenmu * fenzi + target_global_mean * alpha) / (fenzi + alpha)
            test.loc[:,new_col_name] = test.loc[:,col_encode+'_'+k].map(mean_encoding_mapping)
            
            print('Encoding feature: ', col_encode, ' of label: ',label, end=', ')
            fold_ids=get_stratifiedkfold_ids(train[col_label],train[col_label],
                                                n_folds=5,random_state=2019,shuffle=True)
            for i,(trainid,validid) in enumerate(fold_ids):
                print(' fold :', i, end = ' ... ')
                trainfold = train.iloc[trainid,:]
             
                target_global_mean = trainfold[trainfold[col_label]==label][col_label].count() / trainfold[col_label].count()
                #cat_count = trainfold[trainfold[col_label]==label].groupby(col_encode+k)[col_encode+k].count()
                
                fenzi = trainfold[trainfold[col_label]==label].groupby(col_encode+'_'+k).size()
                fenmu = trainfold.groupby(col_encode+'_'+k).size()
              
            
                mean_encoding_mapping =(fenzi/fenmu * fenzi + target_global_mean * alpha) / (fenzi + alpha)
                train.loc[validid,new_col_name] = train.loc[validid,col_encode+'_'+k].map(mean_encoding_mapping)
                gc.collect()
            train.pop(col_encode+'_'+k)
         
            test.pop(col_encode+'_'+k)
          
            train=train.astype('float32')
            test=test.astype('float32')
            gc.collect()
    return train,test
#df = sparse.hstack((df.values, tag_cv), format='csr', dtype='float32')
#for i in df.keys():
   # if 'last' in i:
        #df[i]=df[i].fillna(0)
        #df.pop(i)
train_values, test_values = df[:train_num], df[train_num:]
train_values['label']=clf_labels
cl = ['gender','id_province','cell_province','id_city','city_age']
for i in cl:
    train_values, test_values =mean_encoding_feature_label(train_values,test_values,i,'label',alpha=10)
    print(train_values.shape,test_values.shape)
cl = ['term','rate','bhtype3','nantag']
for i in cl:
    train_values, test_values =mean_encoding_feature_label(train_values,test_values,i,'label',alpha=10)
    print(train_values.shape,test_values.shape)
cl = ['last0weekofmonth']
for i in cl:
    train_values, test_values =mean_encoding_feature_label(train_values,test_values,i,'label',alpha=10)
    print(train_values.shape,test_values.shape)

#######################################编码结束，特征工程结束
train_values.pop('label')
train_values=train_values.astype('float32')
test_values=test_values.astype('float32')
train_values.to_hdf('litetrain6'+'.h5', key='df', mode='w')
test_values.to_hdf('litetest6'+'.h5', key='df', mode='w')
train_values=pd.read_hdf('litetrain6'+'.h5')
test_values=pd.read_hdf('litetest6'+'.h5')

for i in train_values.keys().copy():
    if 'last0dayofweek' in i and 'cod' in i or i in ['usern','usernj']:
        train_values.pop(i)
        test_values.pop(i)
'''for i in train_values.keys().copy():
    if i in ['im','last0weekofmonth','gender','a','dd','id_citylatesize','id_citytooearlysize','term','nantag',
             'l2jdsum','l2jdsize','l2jlatesum','l2jlatesize','l2jtooearlysum','l2jtooearlysize','ddjdsize','ddjlatesum',
             'ddjlatesize','ddjtooearlysum','ddjtooearlysize','l1jdsum','l1jlatesum','l1jlatesize','l1jtooearlysum','l1jtooearlysize',
             'imjdsum','imjdsize','imjlatesum','imjlatesize','imjtooearlysum','imjtooearlysize','eweekjdsum','eweekjlatesum',
             'eweekjtooearlysum','eweekjtooearlysize','ratejdsize','ratejlatesum','ratejlatesize','ratejtooearlysum','ratejtooearlysize','termjlatesum',
             'termjlatesize','termjtooearlysum','termjtooearlysize']:
        train_values.pop(i)
        test_values.pop(i)'''
gx1,gx2= gx[:train_num], gx[train_num:]
classes=1
print(train_values.shape)
# 五折验证也可以改成一次验证，按时间划分训练集和验证集，以避免由于时序引起的数据穿越问题。
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, classes))
test_pred_prob = np.zeros((test_values.shape[0], ))
for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(i, 'fold...')
    t = time.time()

    #trn_x, trn_y = train_values.iloc[trn_idx], clf_labels[trn_idx]
    #val_x, val_y = train_values.iloc[val_idx], clf_labels[val_idx]
    val_repay_amt = amt_labels[val_idx]
    val_due_amt = train_due_amt_df.iloc[val_idx]
    valgx=gx1.iloc[val_idx]
    #val_due_av=train_due_av.iloc[val_idx]
    lgb_train = lgb.Dataset(train_values.iloc[trn_idx[:len(trn_idx)]], clf_labels[trn_idx[:len(trn_idx)]],free_raw_data=False)
    
    lgb_eval = lgb.Dataset(train_values.iloc[val_idx], clf_labels[val_idx])
    # LightGBM parameters found by Bayesian optimization
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'metric': 'logloss',
        #'num_leaves': 256,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed':0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha':1,
        'reg_lambda':2,
        #'objective':'multiclass',
        #'num_class':classes,
        'subsample':0.8,
        'subsample_freq':1,
        'colsample_bytree':0.8,
        'random_state':2019
        }
    
    gbm = lgb.train(params,lgb_train,num_boost_round=30000,                
                    verbose_eval=20,valid_sets=lgb_eval,early_stopping_rounds=100)
    '''a=gbm.predict(train_values,pred_leaf=True, num_iteration=gbm.best_iteration)
    b=gbm.predict(test_values,pred_leaf=True, num_iteration=gbm.best_iteration)
    c=pd.DataFrame()
    d=pd.DataFrame()
    for ix in range(a.shape[1]):
        c['f'+str(ix)]=a[:,ix]
        d['f'+str(ix)]=b[:,ix]
    c.to_hdf('trainl'+str(i)+'.h5', key='df', mode='w')
    d.to_hdf('testl'+str(i)+'.h5', key='df', mode='w')
    del c,d'''
    val_pred_prob_everyday = gbm.predict(train_values.iloc[val_idx], num_iteration=gbm.best_iteration)
    #prob_oof[val_idx] = val_pred_prob_everyday
    #val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]
    valgx['p']=val_pred_prob_everyday
    gg=valgx.groupby(['user_id', 'listing_id'])['p'].sum().reset_index(name='av')
    valgx=valgx.merge(gg,on=['user_id', 'listing_id'],how='left')
    val_pred_repay_amt = val_due_amt['due_amt'].values * (val_pred_prob_everyday/valgx['av'].values)
    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
    amt_oof[val_idx] = val_pred_repay_amt
    test_pred_prob += gbm.predict(test_values, num_iteration=gbm.best_iteration) / skf.n_splits

    print('runtime: {}\n'.format(time.time() - t))
    '''fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] =gbm.feature_name()
    fold_importance_df["importance"] = gbm.feature_importance(importance_type='gain')
    fold_importance_df.to_csv('f.csv',index=False)
    fold_importance_df=pd.read_csv('f.csv')
    f=fold_importance_df.loc[fold_importance_df["importance"]>0]
    kf=list(f["feature"].values)
    train_values=train_values[kf]
    test_values=test_values[kf]'''

    
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] =gbm.feature_name()
fold_importance_df["importance"] = gbm.feature_importance(importance_type='gain')
print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))
print('cv mae:', mean_absolute_error(amt_labels, amt_oof))
#print('cv logloss:', log_loss(clf_labels, prob_oof.argmax(axis=1)))
print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))

gx2['p']=test_pred_prob 
gg=gx2.groupby( 'listing_id')['p'].sum().reset_index(name='av')
gx2=gx2.merge(gg,on= 'listing_id',how='left')
gx2['pv']=gx2['p']/gx2['av']
sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
test_df = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_df[['listing_id', 'auditing_date','due_date', 'due_amt']]
sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example = sub_example.merge(gx2, on=['repay_date', 'listing_id'], how='left')


sub_example['repay_amt'] = sub_example['due_amt'] * sub_example['pv']
#sub_example[(sub_example['days']>10)&(sub_example['days']<21)]['repay_amt'] = 0
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('../03_submission/re.csv', index=False)
