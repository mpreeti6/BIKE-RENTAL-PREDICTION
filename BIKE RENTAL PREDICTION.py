#!/usr/bin/env python
# coding: utf-8

# <html>
# <h1 style="text-align:center;color:red;background-color:powderblue;font-size:500%">Bike Rental Prediction</h1>
# </html>

# <html>
# <img src="https://cdn.dribbble.com/users/196525/screenshots/2689989/media/bf383b711ae4a38444363bd75cb01352.gif" alt="Computer man" style="width:600px;height:250px;"/>
# </html>

# <html>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>Data Set Information:</b><br><br>
#     Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort.
# It is important to make the rental bike available and accessible to the public at the right time as 
# it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes 
# a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply 
# of rental bikes.
# The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), 
# the number of bikes rented per hour and date information.<br><br>
#     <b>Attribute Information:</b><br>
#     🔶Date : year-month-day<br>
# 🔶Rented Bike count - Count of bikes rented at each hour<br>
# 🔶Hour - Hour of he day<br>
# 🔶Temperature-Temperature in Celsius<br>
# 🔶Humidity - %<br>
# 🔶Windspeed - m/s<br>
# 🔶Visibility - 10m<br>
# 🔶Dew point temperature - Celsius<br>
# 🔶Solar radiation - MJ/m2<br>
# 🔶Rainfall - mm<br>
# 🔶Snowfall - cm<br>
# 🔶Seasons - Winter, Spring, Summer, Autumn<br>
# 🔶Holiday - Holiday/No holiday<br>
# 🔶Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)</p>

# <html>
# <h1 style="text-align:left;color:red;font-size:200%">Table of Contents : </h1>
# </html>
# 
#   * [Data Manipulation](#sec1)
#        * [Importing Dataset](#sec1.1)
#        * [Dataset View](#sec1.2)
#        * [Dataset Information](#sec1.3)
#        * [Summary Statistics](#sec1.4)
#        * [Checking for unique values in integer type attribute](#sec1.5)
#        * [Checking for missing values in each column](#sec1.6)
#        * [percentage of missing values in each column](#sec1.7)
#        
#   * [Data Visualization](#sec2)
#        * [Missing Value Plot](#sec2.1)
#        * [Density Plot of Continuous Variables](#sec2.2)
#        * [Box plot for each continuous variavles](#sec2.3)
#        * [Heatmap](#sec2.4)
#        * [Density plot of each continuous variable after applying Power Transformer](#sec2.5)
#        * [Box plot of each continuous variable after applying Power Transformer](#sec2.6)
#        * [Bar Plot of each variable show label distribution of target variable](#sec2.7)
#        * [Pie chart of Categorical Variables](#sec2.8)
#        * [Count Plot of Categorical variable](#sec2.9)
#        
#   * [Variance Inflation Factor](#sec8)
#        
#   * [Feature Selection](#sec3)
#        * [Feature Importance Graph](#sec3.1)
#        
#   * [Splitting our dataset into train and test set](#sec4)
#   
#   * [Feature Scaling](#sec5)
#        
#   * [Modeling](#sec6)
#        * [linear Regression](#sec6.1)
#        * [Polynomial Regression](#sec6.2)
#        * [Decision Tree Regresion](#sec6.3)
#        * [Random Forest Regression](#sec6.4)
#        * [Bagging Regressor](#sec6.5)
#        * [Stacking Regressor](#sec6.6)
#        
#   * [Model Comparison](#sec7)
#        * [Maximum Accuracies in each Column ](#sec7.1)
#        * [Minimum Accuracies in each Column ](#sec7.2)

# ## Data Manipulation <a class="anchor" id="sec1"></a>

# ### Importing libraries 

# <html>
# <img src="https://newrelic.com/sites/default/files/wp_blog_inline_files/shutterstock_1352528811.jpg" alt="Computer man" style="width:150px;height:100px;"/>
# </html>

# In[231]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")
#imported different libraries where we will be working with.


# In[232]:


pd.set_option("display.max_rows", 100, "display.max_columns", 100)


# ### Importing dataset <a class="anchor" id="sec1.1"></a>

# In[233]:


df=pd.read_csv('SeoulBikeData.csv',encoding= 'unicode_escape',parse_dates=[0])


# ### Dataset View <a class="anchor" id="sec1.2"></a>

# In[234]:


df.head(5)


# ### Dataset Information <a class="anchor" id="sec1.3"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">Here we can see that all the data types are in <b>float</b> type and <b>object</b> type.</p>
# </html>

# In[235]:


df.info()


# In[236]:


df=df.astype({'Rented Bike Count':'float','Hour':'object'})


# In[237]:


df.info()


# In[238]:


df=df.rename(columns={'Temperature(°C)':'Temperature','Humidity(%)':'Humidity','Rainfall(mm)':'Rainfall','Snowfall (cm)':'Snowfall','Wind speed (m/s)':'Wind speed','Visibility (10m)':'Visibility','Solar Radiation (MJ/m2)':'Radiation','Dew point temperature(°C)':'Dew point temperature'})


# ### Summary Statistics <a class="anchor" id="sec1.4"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Brief Information of different descriptive statistics-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     *<b>Measures of Frequency</b>              :- Count, Percent, Frequency.<br>
#     *<b>Measures of Central Tendency</b>       :- Mean, Median, and Mode.<br>
#     *<b>Measures of Dispersion or Variation</b>:- Range(min,max),Variance, Standard Deviation.<br>
#     *<b>Measures of Position</b>               :- Percentile Ranks, Quartile Ranks.</p>
# </html>

# In[239]:


df.describe().style.background_gradient()


# ### Checking for unique values in all attribute <a class="anchor" id="sec1.5"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">Different numbers of distint values in each attribute.our target varibale is <b>Rented Bike Count</b> attribute.</p>
# </html>

# In[240]:


df.nunique().sort_values(ascending=True)


# ### Checking for missing values in each column <a class="anchor" id="sec1.6"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">No such missing values in our dataset.<br>

# In[241]:


df.isnull().sum()


# ### percentage of missing values in each column <a class="anchor" id="sec1.7"></a>

# In[242]:


pd.options.display.float_format = '{:,.2f} %'.format
print((df.isnull().sum()/len(df))*100)
pd.options.display.float_format = '{:,.2f}'.format


# ## Data Visualization <a class="anchor" id="sec2"></a>

# ### Missing Value Plot <a class="anchor" id="sec2.1"></a>

# In[243]:


import missingno as msno


# In[244]:


msno.matrix(df,labels=[df.columns],figsize=(30,16),fontsize=12)


# ### Bar Plot <a class="anchor" id="sec2.7"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">Here we can look at each plot and see the hours wise data distribution.</p>
# </html>

# In[245]:


plt.figure(figsize=(18, 18))
for i, col in enumerate(df.select_dtypes(include=['float64','int']).columns):
    plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.subplot(4,3, i+1)
    sns.barplot(data=df,x='Hour', y=col, ax=ax,edgecolor="black",palette='viridis_r')
plt.suptitle('Data distribution of continuous variables')
plt.tight_layout()


# ### Checking the data distribution of each Continuous variable  <a class="anchor" id="sec2.2"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Skewed Distribution-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <img src="https://www.ijamhrjournal.org/articles/2014/1/1/images/IntJAdvMedRes_2014_1_1_30_134449_u5.jpg" alt="Computer man" style="width:800px;height:300px;"/>
#     <b>What is skewed distribution?</b><br>
#     If one tail is longer than another, the distribution is skewed. These distributions are sometimes called asymmetric or asymmetrical distributions as they don’t show any kind of symmetry. Symmetry means that one half of the distribution is a mirror image of the other half. For example, the normal distribution is a symmetric distribution with no skew. The tails are exactly the same.<br>
#     <b>Left Skewed or Negatively Skewed</b>:- A left-skewed distribution has a long left tail. Left-skewed distributions are also called negatively-skewed distributions.(Mean&lt;Median&lt;Mode)<br>
#     <b>Right Skewed or Positively Skewed</b>:-A right-skewed distribution has a long right tail. Right-skewed distributions are also called positive-skew distributions.(Mean&gt;Median&gt;Mode)<br>
#     <b>Symmetric Distribution:-</b>A symmetric distribution is a type of distribution where the left side of the distribution mirrors the right side(Mean=Median=Mode).ex-Normal Distribution
#     </p>
# </html>

# In[246]:


plt.figure(figsize=(12, 12))
for i, col in enumerate(df.select_dtypes(include=['float64','int64']).columns):
    plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.subplot(5,2, i+1)
    sns.histplot(data=df, x=col, ax=ax,color='red',kde=True)
plt.suptitle('Data distribution of continuous variables')
plt.tight_layout()


# From the above graph we can see that there are a lot of attributes which are positively or negatively distributed.

# ### Box Plot <a class="anchor" id="sec2.3"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Box Plot-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>What is Box Plot?</b><br>
#     <img src="https://lh5.googleusercontent.com/Wz6lRE49LVUVq18MyNj6pEwDgdVcHhyDqaG5yGMQX36hy3ZGSyH7fs4A4nbJojGR58k=w2400" alt="Computer man" style="width:800px;height:300px;"/>
#     In descriptive statistics, a box plot or boxplot is a method for graphically demonstrating the locality, spread and skewness groups of numerical data through their quartiles.</p><br>
#     <p style="border:4px solid violet;color:teal;background-color:tan;">
#         <b>How to interpret boxplot</b><br>
#     *Median: In the box plot, the median is displayed rather than the mean.<br>
#     * Q1: The first quartile (25%) position.<br>
#   * Q3: The third quartile (75%) position.<br>
#   *  Interquartile range (IQR): a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles. It represents how 50% of the points were dispersed.<br>
# * Lower and upper 1.5*IQR whiskers: These represent the limits and boundaries for the outliers.<br>
#   *  Outliers: Defined as observations that fall below Q1 − 1.5 IQR or above Q3 + 1.5 IQR. Outliers are displayed as dots or circles.
# 
# </p>
# </html>

# In[247]:


plt.figure(figsize=(18, 18))
for i, col in enumerate(df.select_dtypes(include=['float64','int64']).columns):
    plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.subplot(5,2, i+1)
    sns.boxplot(data=df, x=col, ax=ax,color='blue')
plt.suptitle('Box Plot of continuous variables')
plt.tight_layout()


# In[248]:


#selecting variables that have data types float and int.
var=list(df.select_dtypes(include=['float64','int64']).columns)


# In[249]:


from sklearn.preprocessing import PowerTransformer
sc_X=PowerTransformer(method = 'yeo-johnson')
df[var]=sc_X.fit_transform(df[var])


# ### Data distribution after applying Power Transformer <a class="anchor" id="sec2.4"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">Now the Distribution plots look more symmetrical after treating the outliers.</p>
# </html>

# In[250]:


plt.figure(figsize=(18, 18))
for i, col in enumerate(df.select_dtypes(include=['float64','int64']).columns):
    plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.subplot(5,2, i+1)
    sns.histplot(data=df, x=col, ax=ax,color='red',kde=True)
plt.suptitle('Data distribution of continuous variables')
plt.tight_layout()


# ### Box Plot after applyig Power Transformer <a class="anchor" id="sec2.5"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">Now our Box plots look better after treating the outliers.</p>
# </html>

# In[251]:


plt.figure(figsize=(18, 18))
for i, col in enumerate(df.select_dtypes(include=['float64','int64']).columns):
    plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.subplot(5,2, i+1)
    sns.boxplot(data=df, x=col, ax=ax,color='blue')
plt.suptitle('Box Plot of continuous variables')
plt.tight_layout()


# ### Heatmap <a class="anchor" id="sec2.6"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Correlation Coefficient-</b></p>
#     <img src="https://lh6.googleusercontent.com/WJ-mqD3qf1j4DsE47HifHWf6d3H_2rrjbA0yVPpY-pIGapiZPX2uzM5l055oW-Nvp1U=w2400" alt="Computer man" style="width:800px;height:300px;"/>
#     <img src="https://lh5.googleusercontent.com/mJZDT-3QSQol0hs-opFs6NWUYMpFmiB7Hye-SQGaYJLiO-2LO2-a4358ljGymYUA4Yw=w2400" alt="Computer man" style="width:600px;height:200px;"/>
# </html>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">With the above heatmap plot we can interpret which variable is how much correlated to other variable.</p>
# </html>

# In[252]:


plt.figure(figsize=(8,8))
sns.heatmap(df.select_dtypes(include=['float']).corr(),annot=True,center = 0)
plt.show()


# ### Analysing Categorical Variable <a class="anchor" id="sec2.4"></a>

# ### Pie Chart .<a class="anchor" id="sec2.8"></a>

# <html>
# <p style="border:4px solid violet;color:indigo;background-color:tan">From the below graph it's clear that this is fully balanced data.</p>
# </html>

# In[253]:


season_var=pd.crosstab(index=df['Seasons'],columns='% observations')
plt.pie(season_var['% observations'],labels=season_var['% observations'].index,autopct='%.0f%%')
plt.title('Seasons')
plt.show()


# In[254]:


Functioning_Day_var=pd.crosstab(index=df['Functioning Day'],columns='% observations')
plt.pie(Functioning_Day_var['% observations'],labels=Functioning_Day_var['% observations'].index,autopct='%.0f%%')
plt.title('Functioning Day')
plt.show()


# In[255]:


holiday_var=pd.crosstab(index=df['Holiday'],columns='% observations')
plt.pie(holiday_var['% observations'],labels=holiday_var['% observations'].index,autopct='%.0f%%')
plt.title('Holiday')
plt.show()


# ### Count plot shows that the Seasons variable is balanced <a class="anchor" id="sec2.9"></a>

# In[256]:


sns.barplot(x=season_var.index,y=season_var['% observations'])
plt.title('Seasons')
plt.show()


# In[257]:


df=pd.get_dummies(df,columns=['Holiday','Seasons','Functioning Day','Hour'],drop_first=True)


# In[258]:


X=df.iloc[:,2:]
y=df.iloc[:,1]


# ### Variance Inflation Factor <a class="anchor" id="sec8"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Variance Inflation Factor-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>What is VIF?</b><br>
#     A variance inflation factor(VIF) detects multicollinearity in regression analysis. Multicollinearity is when there’s correlation between predictors (i.e. independent variables) in a model; it’s presence can adversely affect your regression results. The VIF estimates how much the variance of a regression coefficient is inflated due to multicollinearity in the model.<br>
#     <img src="https://www.statisticshowto.com/wp-content/uploads/2015/09/variance-inflation-factor.png" alt="Computer man" style="width:200px;height:100px;"/><br>
#     <b>A rule of thumb for interpreting the variance inflation factor:</b><br>
#     👉 1 = not correlated.<br>
#     👉 Between 1 and 5 = moderately correlated.<br>
#     👉 Greater than 5 = highly correlated.
#     </html>

# In[259]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[260]:


calc_vif(X.select_dtypes(include=['float','int']))


# In[261]:


#Dew Point Temperature is highly correlated .let's delete this variable and check the VIF score again.
del X['Dew point temperature']


# In[262]:


calc_vif(X.select_dtypes(include=['float','int']))
#Each variable is within the range between 1 and 5.


# ### Feature Selection <a class="anchor" id="sec3"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Feature Selection-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     Feature selection methods are intended to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable..</p>
# <img src="https://lh6.googleusercontent.com/exNj6JGWZNAzyB8XXd1LM5FrgMGbfyV09Qgts5bPJA14O7-8AqATF9suuWqYo6oYhOk=w2400" alt="Computer man" style="width:800px;height:300px;"/>
#  </html>

# <html>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     In our dataset we have numerical Input variable and Categorical Output variable.so we will use <b>ANOVA</b> for the feature selection.</p>

# In[263]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression#Correlation


# In[264]:


fs = SelectKBest(score_func=f_regression, k='all')
fs.fit(X, y)


# In[265]:


feature_contribution=(fs.scores_/sum(fs.scores_))*100


# ### Feature importance Graph <a class="anchor" id="sec3.1"></a>

# In[266]:


for i,j in enumerate(X.columns):
    print(f'{j} : {feature_contribution[i]:.2f}%')
plt.figure(figsize=(12,6))
sns.barplot(x=X.columns,y=fs.scores_)
plt.show()


# <html>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     From the above bar garph we can see the feature importance and we will include only those features which are more important for our model.</p>
#     </html>

# ### Splitting our dataset into train and test set <a class="anchor" id="sec4"></a>

# In[267]:


from sklearn.model_selection import train_test_split


# In[268]:


#splitting our dataset in 80% training and 20% testset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# ### Feature Scaling <a class="anchor" id="sec5"></a>

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>Feature Scaling-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>What is Normalization?</b><br>
#     Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.<br>
#     <img src="https://lh3.googleusercontent.com/q9a09LIGXoRO_1bdgFw0C3WcjyEhpDnJ3C8COL65yn0gWhRTtFm5US-Q33aAQujuETQ=w2400" alt="Computer man" style="width:800px;height:200px;"/>
#     <b>What is Standardization?</b><br>
#     Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
#         <img src="https://lh6.googleusercontent.com/_y4dtry_8ImYjqLTXcH68ZpB1--Iea2n2m08d-GLpbTQ4VSREwy3v1PcX8dGQwLE9PE=w2400" alt="Computer man" style="width:800px;height:200px;"/></p>

# <html>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     Here we are going to use <b>Standardization</b>.</p>
#     </html>

# In[269]:


from sklearn.preprocessing import StandardScaler


# In[270]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# ## Modeling <a class="anchor" id="sec6"></a>

# ### Performance Measures for Regression

# <html>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>R-Square:</b><br>
#     <img src="https://vitalflux.com/wp-content/uploads/2019/07/R-squared-formula-linear-regression-model-640x271.jpg" alt="Computer man" style="width:400px;height:100px;"/>
#     <br>
#     <b>Mean Square Error:</b><br>
#     <img src="https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG" alt="Computer man" style="width:400px;height:100px;"/>
#     </p>
#     </html>

# ### K-fold Cross Validation

# <html>
# <p style="color:chocolate;font-size:150%;">
#     <b>K-fold Cross validation-</b></p>
# <p style="border:4px solid violet;color:green;background-color:tan;">
#     <b>What is Cross Validation?</b><br>
#     Cross-validation is a technique in which we train our model using the subset of the data-set and then evaluate using the complementary subset of the data-set.<br>
#     <img src="https://lh6.googleusercontent.com/sp2oloxXrxErMlLFkU3p0TqWYUh4O-9OjhQxBk8RcbKyYrxfPSSfGqw4KU61Vw_Qq7A=w2400" alt="Computer man" style="width:800px;height:400px;"/>
#     </html>

# In[271]:


#importing different Regression models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score


# In[272]:


#creating dictionary for storing different models accuracy
model_comparison={}


# ### Linear Regression <a class="anchor" id="sec6.1"></a>

# In[291]:


model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Linear Regression']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# #### Linear Regression Summary

# In[274]:


import statsmodels.api as sm
lin_reg=sm.OLS(y_train,X_train).fit()
lin_reg.summary()


# #### Assumptions of linear regression 

# In[275]:


residuals = lin_reg.resid


# In[276]:


np.mean(residuals)


# #### Checking for normality of the residuals

# In[277]:


sm.qqplot(residuals)
plt.show()


# #### Checking for homoscedasticity

# In[278]:


plt.scatter(lin_reg.predict(X_train), residuals)
plt.plot(y_train, [0]*len(y_train),c='r')


# ### Polynomial Regression <a class="anchor" id="sec6.2"></a>

# In[292]:


poly_reg=PolynomialFeatures(degree=2)
model=LinearRegression()
model.fit(poly_reg.fit_transform(X_train),y_train)
y_pred=model.predict(poly_reg.fit_transform(X_test))
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Polynomial Regression']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# ### Decision Tree Regression <a class="anchor" id="sec6.3"></a>

# In[293]:


model=DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Decision Tree Regression']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# ### Random Forest Regression <a class="anchor" id="sec6.4"></a>

# In[294]:


model=RandomForestRegressor(n_estimators=10,random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Random forest Regression']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# ### Bagging Regressor <a class="anchor" id="sec6.5"></a>

# In[295]:


from sklearn.ensemble import BaggingRegressor
model= BaggingRegressor(RandomForestRegressor(n_estimators=10,random_state=0),random_state=0)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Bagging Regressor']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# ### Stacking Regressor <a class="anchor" id="sec6.6"></a>

# In[283]:


estimators=[('linear regression',LinearRegression()),('Decision Tree',DecisionTreeRegressor()),('random forest',RandomForestRegressor(n_estimators=10,random_state=0)),('bagging',BaggingRegressor(RandomForestRegressor(n_estimators=10,random_state=0),random_state=0))]


# In[296]:


model=StackingRegressor(estimators=estimators,final_estimator=model1,passthrough=True)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(f"Model R-Square : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"Model MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
model_comparison['Stacking Regressor']=[r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),(accuracies.mean()),(accuracies.std())]


# ### Model Comparison <a class="anchor" id="sec7"></a>

# In[297]:


Model_com_df=pd.DataFrame(model_comparison).T
Model_com_df.columns=['R-Square','MSE','CV Accuracy','CV std']
Model_com_df=Model_com_df.sort_values(by='R-Square',ascending=False)
Model_com_df.style.format("{:.2%}").background_gradient(cmap='Blues')


# #### Maximum Accuracies in each Column <a class="anchor" id="sec7.1"></a>

# In[298]:


Model_com_df.style.highlight_max().set_caption("Maximum Score in each Column").format("{:.2%}")


# #### Minimum Accuracies in each Column <a class="anchor" id="sec7.2"></a>

# In[299]:


Model_com_df.style.highlight_min().set_caption("Minimum Score in each Column").format("{:.2%}")


# In[ ]:




