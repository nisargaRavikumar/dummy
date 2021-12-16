#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install lightgbm


# In[2]:


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import itertools

import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


# In[3]:


train= pd.read_csv(r'C:\Users\nisar\Downloads\ADMProj\IFDtrain.csv',parse_dates=['Transaction_date'])
test= pd.read_csv(r'C:\Users\nisar\Downloads\ADMProj\IFDtest.csv',parse_dates=['Transaction_date'])
tr_df=train


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


df = pd.concat([train, test], sort=False) #put together test and train for data preprocessing
df.head()


# In[7]:


print("Train setinin boyutu:",train.shape)
print("Test setinin boyutu:",test.shape)


# In[8]:


df.shape


# In[9]:


df.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T


# In[10]:


df["Transaction_date"].min()


# In[11]:


df["Transaction_date"].max()


# In[12]:


df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])


# In[13]:


df["store_id"].nunique()


# In[14]:


df["book_id"].nunique()


# In[15]:


df.groupby(["store_id"])["book_id"].nunique()


# In[16]:


df.groupby(["store_id", "book_id"]).agg({"sales": ["sum", "mean", "median", "std"]})


# In[17]:


df['Transaction_date']=pd.to_datetime(df['Transaction_date'])


# In[18]:


# What month was the sale made
df['month'] = df['Transaction_date'].dt.month
# ayın hangi gününde satış yapılmış
df['day_of_month'] = df['Transaction_date'].dt.day
# yılın hangi gününde satış yapılmış
df['day_of_year'] = df['Transaction_date'].dt.dayofyear 
# yılın hangi haftasında satış yapılmış
df['week_of_year'] = df['Transaction_date'].dt.weekofyear
# haftanın hangi gününde satış yapılmış
df['day_of_week'] = df['Transaction_date'].dt.dayofweek
# hangi yılda satış yapılmış
df['year'] = df['Transaction_date'].dt.year
# haftasonu mu değil mi
df["is_wknd"] = df['Transaction_date'].dt.weekday // 4
# ayın başlangıcı mı
df['is_month_start'] = df['Transaction_date'].dt.is_month_start.astype(int)
# ayın bitişi mi
df['is_month_end'] = df['Transaction_date'].dt.is_month_end.astype(int) 


# In[19]:


df.head()


# In[20]:


# sales statistics in store-item-month breakdown
df.groupby(["store_id", "book_id", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


# In[21]:


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


# In[22]:


df.sort_values(by=['store_id', 'book_id', 'Transaction_date'], axis=0, inplace=True)
df.head()


# In[23]:


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store_id", "book_id"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


# In[24]:


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store_id", "book_id"])['sales'].                                                           transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546, 730])


# In[25]:


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] =                 dataframe.groupby(["store_id", "book_id"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
df.tail()


# In[26]:


df = pd.get_dummies(df, columns=['day_of_week', 'month'])


# In[27]:


df['sales'] = np.log1p(df["sales"].values)


# In[28]:


# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["Transaction_date"] < "2016-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["Transaction_date"] >= "2016-01-01") & (df["Transaction_date"] < "2016-04-01"), :]

# bağımsız değişkenler
cols = [col for col in train.columns if col not in ['Transaction_date', 'id', "sales", "year"]]


# In[29]:


val.shape
val.head()


# In[30]:


# train seti için bağımlı değişkenin seçilmesi
Y_train = train['sales']

# train seti için bağımsız değişkenin seçilmesi
X_train = train[cols]

# validasyon seti için bağımlı değişkenin seçilmesi
Y_val = val['sales']

# validasyon seti için bağımsız değişkenin seçilmesi
X_val = val[cols] 

# kontrol
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


# In[31]:


X_train.head()


# In[32]:


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# In[33]:


# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000, 
              'early_stopping_rounds': 200,
              'nthread': -1}


# In[34]:


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
print(lgbtrain)
print()
print(lgbval)
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=10000,
                  early_stopping_rounds=200,
                  feval=lgbm_smape, # hatyı gözlemliyoruz
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# percentage of validation error
smape(np.expm1(y_pred_val), np.expm1(Y_val))


# In[35]:


#Final Model

# test ve train bağımlı/bağımsız değişkenlerinin belirlenmesi

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]


# In[36]:


print(X_test)


# In[37]:


X_test.to_csv('X_test.csv', index=False)


# In[38]:


lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = model.predict(X_test, num_iteration=model.best_iteration)


# In[39]:


print(test_preds)


# In[40]:


print(test['Transaction_date'])


# In[46]:


#1. Estimates for the store's 1st products

forecast = pd.DataFrame({"date":test["Transaction_date"],
                        "store":test["store_id"],
                        "item":test["book_id"],
                        "sales":test_preds
                        })


# In[42]:


import joblib


# In[43]:


filename = 'finalized_model1.sav'
joblib.dump(model, filename)


# In[48]:


loaded_model = joblib.load(filename)
result = loaded_model.predict(X_test, num_iteration=model.best_iteration)
forecast = pd.DataFrame({"date":test["Transaction_date"],
                        "store":test["store_id"],
                        "item":test["book_id"],
                        "sales":test_preds
                        })

forecast[(forecast.store == 1) & (forecast.item == 121749)].set_index("date").sales.plot(color = "green",
                                                                                    figsize = (20,9),
                                                                                    legend=True, label = "Store 1 Item 1 Forecast");


# In[ ]:




