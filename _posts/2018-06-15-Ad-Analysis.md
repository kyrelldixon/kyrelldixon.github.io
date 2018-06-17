---
title: Advertising Analysis Project
description: NexT is a high quality elegant Jekyll theme ported from Hexo Next. It is crafted from scratch, with love.
categories:
  - Jupyter Notebook
  - Machine Learning
  - Data Analysis
tags:
  - Machine Learning
  - K-Nearest Neighbors
  - Classification
  - Data Visualization
---

In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. Trying to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

- 'Daily Time Spent on Site': consumer time on site in minutes
- 'Age': cutomer age in years
- 'Area Income': Avg. Income of geographical area of consumer
- 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
- 'Ad Topic Line': Headline of the advertisement
- 'City': City of consumer
- 'Male': Whether or not consumer was male
- 'Country': Country of consumer
- 'Timestamp': Time at which consumer clicked on Ad or closed window
- 'Clicked on Ad': 0 or 1 indicated clicking on Ad

## Importing Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
%matplotlib inline
```

## Reading in Data

```python
ad = pd.read_csv('advertising.csv')
ad.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Ad Topic Line</th>
      <th>City</th>
      <th>Male</th>
      <th>Country</th>
      <th>Timestamp</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68.95</td>
      <td>35</td>
      <td>61833.90</td>
      <td>256.09</td>
      <td>Cloned 5thgeneration orchestration</td>
      <td>Wrightburgh</td>
      <td>0</td>
      <td>Tunisia</td>
      <td>2016-03-27 00:53:11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.23</td>
      <td>31</td>
      <td>68441.85</td>
      <td>193.77</td>
      <td>Monitored national standardization</td>
      <td>West Jodi</td>
      <td>1</td>
      <td>Nauru</td>
      <td>2016-04-04 01:39:02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69.47</td>
      <td>26</td>
      <td>59785.94</td>
      <td>236.50</td>
      <td>Organic bottom-line service-desk</td>
      <td>Davidton</td>
      <td>0</td>
      <td>San Marino</td>
      <td>2016-03-13 20:35:42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.15</td>
      <td>29</td>
      <td>54806.18</td>
      <td>245.89</td>
      <td>Triple-buffered reciprocal time-frame</td>
      <td>West Terrifurt</td>
      <td>1</td>
      <td>Italy</td>
      <td>2016-01-10 02:31:19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68.37</td>
      <td>35</td>
      <td>73889.99</td>
      <td>225.58</td>
      <td>Robust logistical utilization</td>
      <td>South Manuel</td>
      <td>0</td>
      <td>Iceland</td>
      <td>2016-06-03 03:36:18</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Data Summary

```python
ad.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
    Daily Time Spent on Site    1000 non-null float64
    Age                         1000 non-null int64
    Area Income                 1000 non-null float64
    Daily Internet Usage        1000 non-null float64
    Ad Topic Line               1000 non-null object
    City                        1000 non-null object
    Male                        1000 non-null int64
    Country                     1000 non-null object
    Timestamp                   1000 non-null object
    Clicked on Ad               1000 non-null int64
    dtypes: float64(3), int64(3), object(4)
    memory usage: 78.2+ KB

```python
ad.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Male</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.000200</td>
      <td>36.009000</td>
      <td>55000.000080</td>
      <td>180.000100</td>
      <td>0.481000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.853615</td>
      <td>8.785562</td>
      <td>13414.634022</td>
      <td>43.902339</td>
      <td>0.499889</td>
      <td>0.50025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.600000</td>
      <td>19.000000</td>
      <td>13996.500000</td>
      <td>104.780000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.360000</td>
      <td>29.000000</td>
      <td>47031.802500</td>
      <td>138.830000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68.215000</td>
      <td>35.000000</td>
      <td>57012.300000</td>
      <td>183.130000</td>
      <td>0.000000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78.547500</td>
      <td>42.000000</td>
      <td>65470.635000</td>
      <td>218.792500</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>91.430000</td>
      <td>61.000000</td>
      <td>79484.800000</td>
      <td>269.960000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis


```python
sns.set_context('notebook')
sns.set_style('white')
```


```python
ad['Age'].plot.hist(bins=40)
plt.xlabel('Age')
```




    <matplotlib.text.Text at 0x10d75b978>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_9_1.png)


The age counts follow a normal distribution with a spike around 40-42 years old. Let's see if age correlates with ad clicks.


```python
sns.countplot('Age',data=ad,hue='Clicked on Ad')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10d982fd0>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_11_1.png)



```python
sns.factorplot(x='Clicked on Ad',y='Age',data=ad,kind='swarm')
```

    <seaborn.axisgrid.FacetGrid at 0x110d15198>

![png](/assets/images/Ad_Analysis_files/Ad_Analysis_12_1.png)

Age doesn't seem to have a high correlation with ad clicks, but there is some grouping on the extreme ends. Now to check for general trends.

```python
sns.pairplot(ad,hue='Clicked on Ad',kind='scatter')
```

    <seaborn.axisgrid.PairGrid at 0x110f15240>

![png](/assets/images/Ad_Analysis_files/Ad_Analysis_14_1.png)

```python
sns.heatmap(ad.corr())
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1132cb630>

![png](/assets/images/Ad_Analysis_files/Ad_Analysis_15_1.png)

```python
sns.factorplot(x='Clicked on Ad',y='Daily Time Spent on Site',data=ad,kind='swarm')
```




    <seaborn.axisgrid.FacetGrid at 0x1131ff6d8>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_16_1.png)



```python
sns.factorplot(x='Clicked on Ad',y='Area Income',data=ad,kind='swarm')
```




    <seaborn.axisgrid.FacetGrid at 0x112ac8c18>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_17_1.png)



```python
sns.factorplot(x='Clicked on Ad',y='Daily Internet Usage',data=ad,kind='swarm')
```




    <seaborn.axisgrid.FacetGrid at 0x112ac8ac8>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_18_1.png)



```python
sns.factorplot(x='Clicked on Ad',y='Age',data=ad,kind='swarm')
```




    <seaborn.axisgrid.FacetGrid at 0x113789d68>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_19_1.png)


## Findings
Most of the groupings have overlap with another group when analyzing who clicked on an ad relative to each feature. The highest levels of grouping was found between Clicked on Ad and:

- Daily Internet Usage
- Daily Time Spent on Site
- Area Income

Being male didn't seem to have much affect on whether an ad was clicked. I'm going to model the data based on all columns and just the most relevant columns.


```python
all_columns = ['Male','Age','Daily Internet Usage', 'Daily Time Spent on Site', 'Area Income']
relevant_columns = ['Daily Internet Usage', 'Daily Time Spent on Site', 'Area Income']
```

# Prediction and Modeling


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
scaler.fit(ad[all_columns])
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
scaled_features = scaler.transform(ad[all_columns])
scaled_features[:5]
```




    array([[-0.96269532, -0.11490498,  1.73403   ,  0.24926659,  0.50969109],
           [ 1.03875025, -0.57042523,  0.31380538,  0.96113227,  1.00253021],
           [-0.96269532, -1.13982553,  1.28758905,  0.28208309,  0.35694859],
           [ 1.03875025, -0.79818535,  1.50157989,  0.57743162, -0.01445564],
           [-0.96269532, -0.11490498,  1.03873069,  0.21266356,  1.40886751]])




```python
scaled_ad = pd.DataFrame(scaled_features,columns=all_columns)
scaled_ad.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
      <th>Age</th>
      <th>Daily Internet Usage</th>
      <th>Daily Time Spent on Site</th>
      <th>Area Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.962695</td>
      <td>-0.114905</td>
      <td>1.734030</td>
      <td>0.249267</td>
      <td>0.509691</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.038750</td>
      <td>-0.570425</td>
      <td>0.313805</td>
      <td>0.961132</td>
      <td>1.002530</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.962695</td>
      <td>-1.139826</td>
      <td>1.287589</td>
      <td>0.282083</td>
      <td>0.356949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.038750</td>
      <td>-0.798185</td>
      <td>1.501580</td>
      <td>0.577432</td>
      <td>-0.014456</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.962695</td>
      <td>-0.114905</td>
      <td>1.038731</td>
      <td>0.212664</td>
      <td>1.408868</td>
    </tr>
  </tbody>
</table>
</div>



## Setting up data
I'm using two training and testing sets on the standardized data.


```python
X_all = scaled_ad
X_rel = scaled_ad[relevant_columns]
y = ad['Clicked on Ad']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)
X_rel_train, X_rel_test, y_rel_train, y_rel_test = train_test_split(X_rel, y, test_size=0.3, random_state=42)
```

## Logisitic Regression


```python
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
pred = logmodel.predict(X_test)
```


```python
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
```

                 precision    recall  f1-score   support
    
              0       0.96      0.98      0.97       146
              1       0.98      0.96      0.97       154
    
    avg / total       0.97      0.97      0.97       300
    
    
    
    [[143   3]
     [  6 148]]



```python
logmodel.fit(X_rel_train,y_rel_train)
pred_rel = logmodel.predict(X_rel_test)
```


```python
print(classification_report(y_rel_test,pred_rel))
print('\n')
print(confusion_matrix(y_rel_test,pred_rel))
```

                 precision    recall  f1-score   support
    
              0       0.93      0.97      0.95       146
              1       0.97      0.93      0.95       154
    
    avg / total       0.95      0.95      0.95       300
    
    
    
    [[141   5]
     [ 11 143]]


## K-Nearest Neighbors


```python
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
```


```python
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
```

                 precision    recall  f1-score   support
    
              0       0.92      0.97      0.95       146
              1       0.97      0.92      0.95       154
    
    avg / total       0.95      0.95      0.95       300
    
    
    
    [[142   4]
     [ 12 142]]



```python
knn.fit(X_rel_train,y_rel_train)
pred_rel = knn.predict(X_rel_test)
```


```python
print(classification_report(y_rel_test,pred_rel))
print('\n')
print(confusion_matrix(y_rel_test,pred_rel))
```

                 precision    recall  f1-score   support
    
              0       0.92      0.96      0.94       146
              1       0.96      0.92      0.94       154
    
    avg / total       0.94      0.94      0.94       300
    
    
    
    [[140   6]
     [ 12 142]]



```python
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```




    <matplotlib.text.Text at 0x11464c828>




![png](/assets/images/Ad_Analysis_files/Ad_Analysis_43_1.png)



```python
knn = KNeighborsClassifier(n_neighbors=14)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

knn.fit(X_rel_train,y_rel_train)
pred_rel = knn.predict(X_rel_test)
```


```python
print('FOR ALL COLUMNS:')
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')

print('FOR RELEVANT COLUMNS:')
print(classification_report(y_rel_test,pred_rel))
print('\n')
print(confusion_matrix(y_rel_test,pred_rel))
```

    FOR ALL COLUMNS:
                 precision    recall  f1-score   support
    
              0       0.92      0.99      0.96       146
              1       0.99      0.92      0.96       154
    
    avg / total       0.96      0.96      0.96       300
    
    
    
    [[145   1]
     [ 12 142]]
    
    
    FOR RELEVANT COLUMNS:
                 precision    recall  f1-score   support
    
              0       0.92      0.97      0.94       146
              1       0.97      0.92      0.94       154
    
    avg / total       0.94      0.94      0.94       300
    
    
    
    [[141   5]
     [ 12 142]]



```python

```
