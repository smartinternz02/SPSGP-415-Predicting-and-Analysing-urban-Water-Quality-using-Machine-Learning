#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[2]:


data = pd.read_csv(r'C:\Users\rajes\Desktop\python project\internship\data_set\water_data.csv',encoding = 'ISO-8859-1', low_memory = False)
data.head()


# In[ ]:





# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().any()


# In[7]:


data.isnull().sum()


# In[8]:


data.dtypes


# In[9]:


data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')
data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
data['PH']=pd.to_numeric(data['PH'],errors='coerce')
data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
data.dtypes


# In[10]:


data.isnull().sum()


# In[11]:


data['Temp'].fillna(data['Temp'].mean(),inplace=True)
data['D.O. (mg/l)'].fillna(data['D.O. (mg/l)'].mean(),inplace=True)
data['PH'].fillna(data['PH'].mean(),inplace=True)
data['CONDUCTIVITY (µmhos/cm)'].fillna(data['CONDUCTIVITY (µmhos/cm)'].mean(),inplace=True)
data['B.O.D. (mg/l)'].fillna(data['B.O.D. (mg/l)'].mean(),inplace=True)
data['NITRATENAN N+ NITRITENANN (mg/l)'].fillna(data['NITRATENAN N+ NITRITENANN (mg/l)'].mean(),inplace=True)
data['TOTAL COLIFORM (MPN/100ml)Mean'].fillna(data['TOTAL COLIFORM (MPN/100ml)Mean'].mean(),inplace=True)
data.isnull().any()


# In[ ]:





# In[12]:


data.head()


# In[13]:


data.drop(['FECAL COLIFORM (MPN/100ml)'], axis = 1, inplace = True)
data.head()


# In[14]:


data=data.rename(columns = {'D.O. (mg/l)':'do'})
data=data.rename(columns = {'CONDUCTIVITY (µmhos/cm)':'co'})
data=data.rename(columns = {'B.O.D. (mg/l)':'bod'})
data=data.rename(columns = {'NITRATENAN N+ NITRITENANN (mg/l)':'na'})
data=data.rename(columns =  {'TOTAL COLIFORM (MPN/100ml)Mean':'tc'})
data=data.rename(columns =  {'STATION CODE':'station'})
data=data.rename(columns =  {'LOCATIONS':'location'})
data=data.rename(columns =  {'STATE':'state'})
data=data.rename(columns =  {'PH':'ph'})
data.head()


# In[15]:


data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))


# In[16]:


data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))


# In[17]:


data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))
data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))
data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))
data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))


# In[18]:


data.head()


# In[19]:


data['wph']=data.npH * 0.165
data['wdo']=data.ndo * 0.281
data['wbdo']=data.nbdo * 0.234
data['wec']=data.nec* 0.009
data['wna']=data.nna * 0.028
data['wco']=data.nco * 0.281
data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco


# In[20]:


average=data.groupby('year')['wqi'].mean()
average.head()


# In[21]:


data.head()


# In[22]:


data1=average.reset_index(level=0,inplace=False)
data1


# In[23]:


data1.boxplot(column='year')


# In[24]:


year=data1['year'].values
AQI=data1['wqi'].values
data1['wqi']=pd.to_numeric(data1['wqi'])
data1['year']=pd.to_numeric(data1['year'])

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(year,AQI, color='red')
plt.show()
data1


# In[25]:


cols =['year']
y = data1['wqi']
x=data1[cols]

plt.scatter(x,y)
plt.show()


# In[26]:


sns.countplot(data['year']) 
sns.countplot(data['do'])
sns.countplot(data['ph'])
sns.countplot(data['co'])
sns.countplot(data['bod'])
sns.countplot(data['na'])
sns.countplot(data['tc'])
sns.countplot(data['wqi'])


# In[27]:


data.drop(['Temp','station','location','state','nbdo',"nec","nna","wph","wdo","wbdo","wec","wna","wco","npH","ndo","nco"],axis = 1,inplace=True)


# In[28]:


x = data.iloc[:,0:7].values
y = data.iloc[:,7:].values
x


# In[29]:


x.shape


# In[30]:


y.shape


# In[31]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
x


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 10)


# In[33]:


X_train.shape


# In[34]:


X_test.shape


# In[35]:


y_train.shape


# In[36]:


y_test.shape


# In[38]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
y_pred


# In[39]:


reg_rf.score(X_train, y_train)


# In[40]:


reg_rf.score(X_test, y_test)


# In[41]:


y_test[10:15]


# In[42]:


y_pred[10:15]


# In[43]:


sns.distplot(y_test-y_pred)
plt.show()


# In[44]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[45]:


metrics.r2_score(y_test, y_pred)


# In[46]:


import pickle
pickle.dump(reg_rf,open('wqi.pkl','wb'))
model = pickle.load(open('wqi.pkl','rb'))


# In[47]:


pickle.dump(sc,open('sc.pkl','wb'))
model = pickle.load(open('sc.pkl','rb'))


# In[ ]:




