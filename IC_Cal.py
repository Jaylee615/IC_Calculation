# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:20:08 2017
@author: jaylee
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import statsmodels.api as sm
from scipy.stats import pearsonr
import os

#%%
# 研究IC,IR等指标
# 中位数去极值法
def remove_outliers(data,ratio=10):
    s1=deepcopy(data)
    m1=np.nanmedian(s1)
    s2=np.abs(s1-m1)
    m2=np.nanmedian(s2)
    s1[np.where(s1>m1+ratio*m2)]=m1+ratio*m2
    s1[np.where(s1<m1-ratio*m2)]=m1-ratio*m2
    return s1

# 回归分析
def OLS_analysis(y_mat,x_mat,name='y-x'):
    # 计算最近的month个月
    y_mat=y_mat[:,1:]
    x_mat=x_mat[:,:-1]
    # 记录回归系数和p值
    sta_df=pd.DataFrame(index=range(y_mat.shape[1]),columns=['回归系数','p值','t值','IC'])
    # 对每个截面进行回归
    for n in range(y_mat.shape[1]):
        y=y_mat[:,n]
        x=x_mat[:,n]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        # 中位数去极值
        x_1=remove_outliers(x[~nas])
        y_1=remove_outliers(y[~nas])
        # 标准化：减去平均数再除以std
        x_scaled=preprocessing.scale(x_1)
        y_scaled=preprocessing.scale(y_1)
        #    corr=pearsonr(y_scaled,x_scaled)
        # 线性回归
        model = sm.OLS(y_scaled,x_scaled,missing='drop')
        results = model.fit()
        # IC 
        IC=pearsonr(y_scaled,x_scaled)
        
        sta_df.ix[n,'IC']=IC[0]
        sta_df.ix[n,'回归系数']=results.params[0]
        sta_df.ix[n,'p值']=results.pvalues[0]    
        sta_df.ix[n,'t值']=results.tvalues[0]  
    # 计算各统计量均值
    t_mean=sta_df['t值'].mean()
    t_strength=sta_df['t值'].mean()/sta_df['t值'].std()
    param_mean=sta_df['回归系数'].mean()
    t_abs_mean=sta_df['t值'].abs().mean()
    p_mean=sta_df['p值'].mean()
    param_mean_2=sta_df[sta_df['p值']<0.05]['回归系数'].mean()
    ic_mean=sta_df['IC'].mean()
    ic_std=sta_df['IC'].std()
    ir=sta_df['IC'].mean()/sta_df['IC'].std()
    a=sta_df['IC']
    ic_rate=len(a[a>0])/len(a)
    ic_rate2=len(a[abs(a)>0.02])/len(a)
    ic_abs=a.abs().mean()
    meanvalues=pd.DataFrame(columns=['t值均值','回归系数均值','|t|值均值','p均值','回归系数（p<0.05）均值','t值强度',\
                                     'IC均值','IC标准差','IR比率','IC>0占比','|IC|>0.02占比','|IC|均值'])
    meanvalues.loc[name,:]=[t_mean,param_mean,t_abs_mean,p_mean,param_mean_2,t_strength,\
                  ic_mean,ic_std,ir,ic_rate,ic_rate2,ic_abs]
    meanvalues=meanvalues[['t值均值','|t|值均值','p均值','t值强度','IC均值','IC标准差','IR比率','|IC|均值','|IC|>0.02占比','IC>0占比']]
    return meanvalues

#%%单个因子测试
startime = "2010-01-29"
endtime = "2013-12-31"#因子样本起止时间
ret=pd.read_excel('return.xlsx')#月收益率
ret_df=ret.ix[:,startime:endtime]
factor=pd.read_excel('pe_ttm.xlsx')#读取数据
factor_df=factor.ix[:,startime:endtime]
y_mat=ret_df.values
x_mat=factor_df.values
result_df=OLS_analysis(y_mat,x_mat,name='PE')
result_df.to_excel('IC.xlsx')



