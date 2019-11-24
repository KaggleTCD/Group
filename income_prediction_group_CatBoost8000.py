#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor

# In[192]:


# Import the train and test dataset
#F:/MS/Masters/Machine Learning/Group/
#"F:/MS/Masters/Machine Learning/Group/
df = pd.read_csv("tcd-ml-1920-group-income-train.csv")
df_test = pd.read_csv("tcd-ml-1920-group-income-test.csv")


# In[ ]:





# In[193]:


df


# In[194]:


df.head()


# In[250]:


df_test


# In[196]:


# Cleaning the train and test dataset
df = df.drop("Instance",axis = 1)
df_test = df_test.drop("Instance",axis = 1)


# In[197]:


df = df.replace("#NUM!", np.nan)
df_test = df_test.replace("#NUM!", np.nan)


# In[ ]:





# In[198]:


#Function to calculate target encoding of a specific column
#df=dataframe,bycolumn = column,ontarget = target,w = weight
def calc_target_encoding(df,bycolumn,ontarget, w):
    # Calculate the overall mean
    mean = df[ontarget].mean()
    print("-------------Train and Test---------------------")
    print(df[ontarget])
    print("----------------------------------")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(mean)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # Calculate the number of values and the mean of each group
    agg = df.groupby(bycolumn)[ontarget].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the smoothed means
    smooth = (counts * means + w * mean) / (counts + w)

    # Replace each value by its smoothed mean
    return df[bycolumn].map(smooth)


# In[199]:


df.isnull().sum()


# In[200]:


#Renaming the columns having spaces in their names
df = df.rename(index=str, columns={"Year of Record": "YearofRecord"})
df_test = df_test.rename(index=str, columns={"Year of Record": "YearofRecord"})


# In[201]:


df = df.rename(index=str, columns={"Housing Situation": "HousingSituation"})
df_test = df_test.rename(index=str, columns={"Housing Situation": "HousingSituation"})


# In[202]:


df = df.rename(index=str, columns={"Crime Level in the City of Employement": "CrimeLevelintheCityofEmployement"})
df_test = df_test.rename(index=str, columns={"Crime Level in the City of Employement": "CrimeLevelintheCityofEmployement"})


# In[203]:


df = df.rename(index=str, columns={"Work Experience in Current Job [years]": "WorkExperience"})
df_test = df_test.rename(index=str, columns={"Work Experience in Current Job [years]": "WorkExperience"})


# In[204]:


df = df.rename(index=str, columns={"Satisfation with employer": "Satisfationwithemployer"})
df_test = df_test.rename(index=str, columns={"Satisfation with employer": "Satisfationwithemployer"})


# In[205]:


df = df.rename(index=str, columns={"Size of City": "SizeofCity"})
df_test = df_test.rename(index=str, columns={"Size of City": "SizeofCity"})


# In[206]:


df = df.rename(index=str, columns={"University Degree": "UniversityDegree"})
df_test = df_test.rename(index=str, columns={"University Degree": "UniversityDegree"})


# In[207]:


df = df.rename(index=str, columns={"Wears Glasses": "WearsGlasses"})
df_test = df_test.rename(index=str, columns={"Wears Glasses": "WearsGlasses"})


# In[208]:


df = df.rename(index=str, columns={"Hair Color": "HairColor"})
df_test = df_test.rename(index=str, columns={"Hair Color": "HairColor"})


# In[209]:


df = df.rename(index=str, columns={"Body Height [cm]": "BodyHeight"})
df_test = df_test.rename(index=str, columns={"Body Height [cm]": "BodyHeight"})


# In[210]:


df = df.rename(index=str, columns={"Yearly Income in addition to Salary (e.g. Rental Income)": "YearlyIncomeinadditiontoSalary"})
df_test = df_test.rename(index=str, columns={"Yearly Income in addition to Salary (e.g. Rental Income)": "YearlyIncomeinadditiontoSalary"})


# In[211]:


df = df.rename(index=str, columns={"Total Yearly Income [EUR]": "TotalYearlyIncome"})
df_test = df_test.rename(index=str, columns={"Total Yearly Income [EUR]": "TotalYearlyIncome"})


# In[212]:


df


# In[213]:


df_test


# In[214]:


data = pd.concat([df, df_test], sort=False)


# In[215]:


data


# In[216]:


data['YearofRecord'] = data['YearofRecord'].fillna((data['YearofRecord'].mode()[0]))
#df_test['YearofRecord'] = df_test['YearofRecord'].fillna((df_test['YearofRecord'].mode()[0]))


# In[217]:


data['YearlyIncomeinadditiontoSalary'] = data['YearlyIncomeinadditiontoSalary'].map(lambda x: x.rstrip('EUR'))
data['YearlyIncomeinadditiontoSalary'] = data['YearlyIncomeinadditiontoSalary'].astype(float)
#df_test['YearlyIncomeinadditiontoSalary'] = df_test['YearlyIncomeinadditiontoSalary'].map(lambda x: x.rstrip('EUR'))


# In[218]:


#Target encoding for HousingSituation
data['HousingSituation'] = calc_target_encoding(data,'HousingSituation','TotalYearlyIncome',48)


# In[219]:


#Target encoding for HousingSituation
#df_test['HousingSituation'] = calc_target_encoding(df,'HousingSituation','TotalYearlyIncome',48)


# In[ ]:





# In[220]:


#Target encoding for Satisfationwithemployer
data['Satisfationwithemployer'] = calc_target_encoding(data,'Satisfationwithemployer','TotalYearlyIncome',48)


# In[221]:


#Target encoding for Satisfationwithemployer
#df_test['Satisfationwithemployer'] = calc_target_encoding(df,'Satisfationwithemployer','TotalYearlyIncome',48)


# In[ ]:





# In[222]:


data['Satisfationwithemployer'] = data['Satisfationwithemployer'].fillna((data['Satisfationwithemployer'].mean()))
#df_test['Satisfationwithemployer'] = df_test['Satisfationwithemployer'].fillna((df_test['Satisfationwithemployer'].mean()))


# In[ ]:





# In[223]:


#Target encoding for Gender
data['Gender'] = calc_target_encoding(data,'Gender','TotalYearlyIncome',48)


# In[224]:


#Target encoding for Gender
#df_test['Gender'] = calc_target_encoding(df,'Gender','TotalYearlyIncome',48)


# In[ ]:





# In[225]:


data['Gender'] = data['Gender'].fillna((data['Gender'].mean()))
#df_test['Gender'] = df_test['Gender'].fillna((df_test['Gender'].mean()))


# In[ ]:





# In[226]:


data['Country'] = data['Country'].fillna((data['Country'].mode()[0]))


# In[227]:


#Target encoding for Country
data['Country'] = calc_target_encoding(data,'Country','TotalYearlyIncome',48)


# In[228]:


#Target encoding for Country
#df_test['Country'] = calc_target_encoding(df,'Country','TotalYearlyIncome',48)


# In[ ]:





# In[ ]:





# In[229]:


data = data.drop("HairColor",axis = 1)
#df_test = df_test.drop("HairColor",axis = 1)


# In[ ]:





# In[230]:


#Target encoding for Profession
data['Profession'] = calc_target_encoding(data,'Profession','TotalYearlyIncome',48)


# In[231]:


#df_test['Profession'] = calc_target_encoding(df,'Profession','TotalYearlyIncome',48)


# In[ ]:





# In[232]:


data['Profession'] = data['Profession'].fillna((data['Profession'].mean()))


# In[233]:


#df_test['Profession'] = df_test['Profession'].fillna((df_test['Profession'].mean()))


# In[ ]:





# In[234]:


data['UniversityDegree'] = calc_target_encoding(data,'UniversityDegree','TotalYearlyIncome',48)


# In[235]:


#df_test['UniversityDegree'] = calc_target_encoding(df,'UniversityDegree','TotalYearlyIncome',48)


# In[236]:


data.isnull().sum()


# In[237]:


data['UniversityDegree'] = data['UniversityDegree'].fillna((data['UniversityDegree'].mean()))


# In[238]:


#df_test['UniversityDegree'] = df_test['UniversityDegree'].fillna((df_test['UniversityDegree'].mean()))


# In[ ]:





# In[239]:


data['WorkExperience'] = calc_target_encoding(data,'WorkExperience','TotalYearlyIncome',48)


# In[240]:


#df_test['WorkExperience'] = calc_target_encoding(df,'WorkExperience','TotalYearlyIncome',48)


# In[ ]:





# In[241]:


data['WorkExperience'] = data['WorkExperience'].fillna((data['WorkExperience'].mean()))
#df_test['WorkExperience'] = df_test['WorkExperience'].fillna((df_test['WorkExperience'].mean()))


# In[242]:


data.isnull().sum()


# In[243]:


data


# In[244]:


data['Country'] = data['Country'].fillna((data['Country'].mean()))


# In[245]:


data.isnull().sum()


# In[246]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# In[247]:


df_new = data[0:len(df)]
df_test_new = data[len(df):]


# In[248]:


df_new


# In[249]:


df_test_new


# In[251]:


df_new = pd.DataFrame(df_new)
df_test_new = pd.DataFrame(df_test_new)


# In[252]:


df_new


# In[253]:


df_test_new


# In[254]:


X = df_new.drop("TotalYearlyIncome",axis=1)
Y = df_new.TotalYearlyIncome


# In[255]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)


# In[256]:


cb_model = CatBoostRegressor(iterations=8000,
                             learning_rate=0.01,
                             depth=12,
                             eval_metric='MAE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
cb_model.fit(X_train, Y_train,
             eval_set=(X_test,Y_test),
             use_best_model=True,
             verbose=True)


# In[257]:


#cb_model.fit(X_train,Y_train)


# In[258]:


Y_predicted = cb_model.predict(X_test)


# In[259]:


from sklearn.metrics import mean_absolute_error


# In[260]:


meanabsoluteerror = mean_absolute_error(Y_test,Y_predicted)


# In[261]:


print(meanabsoluteerror)


# In[262]:


X_question_test = df_test_new.drop("TotalYearlyIncome", axis=1)


# In[263]:

cb_model.fit(X,Y)
Y_question_pred = cb_model.predict(X_question_test)


# In[264]:


Y_question_pred = pd.DataFrame(Y_question_pred)


# In[265]:


Y_question_pred.to_csv("submission.csv", sep=',', index=False, header=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




