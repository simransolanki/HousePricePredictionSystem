# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:59:54 2021

@author: simran
"""

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import metrics  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns 
from num2words import num2words
import altair as alt
sns.set(color_codes=True)



og_dataset = pd.read_csv('dataset/Maindataset.csv')
pd.options.display.max_columns = None

def display_original_dataset():
    st.write(og_dataset)
    st.write('The dataset contains ',og_dataset.shape[1],' columns and ',og_dataset.shape[0],' rows')
    
dataset=og_dataset.rename({'What is your age?':'Age',
          'What is your gender?':'Gender', 
          'In which area do you stay in western line in Mumbai?':'Location',
        'Your yearly income in INR':'Income',
           'Do you own a house?': 'Own_House',
           'Where is your house located?':'House_Located',   
          'What is the carpet area of your current house in square feet?':'Carpet_Area',
          'What is the current price of your house in INR?':'Price',
          'How many rooms do you have?':'No of Bedrooms',
          'Are you planning to buy a new house in Mumbai(Western Line)?':'New_House',
          'Would you take a housing loan if you had to buy a house?':'House_Loan',
        'In which area you would like to buy a new house in Mumbai(Western Line) if you had to?':'n_Location',
        'What type of house you would look for if you had to buy a house?':'n_House_Type',  
        'What would be your budget if you had to buy a house?':'Budget',
        'How much carpet area would you want(square feet) if you had to buy a house?':'n_Carpet_Area',
        'How many rooms do you want if you had to buy a house?':'n_Bedrooms',
        'If you had to buy a house what would you buy?':'New/Resale'},axis=1)

dataset.head()

dataset2=og_dataset.rename({'What is your age?':'Age',
          'What is your gender?':'Gender', 
          'In which area do you stay in western line in Mumbai?':'Location',
        'Your yearly income in INR':'Income',
           'Do you own a house?': 'Own_House',
           'Where is your house located?':'House_Located',   
          'What is the carpet area of your current house in square feet?':'Carpet_Area',
          'What is the current price of your house in INR?':'Price',
          'How many rooms do you have?':'No of Bedrooms',
          'Are you planning to buy a new house in Mumbai(Western Line)?':'New_House',
          'Would you take a housing loan if you had to buy a house?':'House_Loan',
        'In which area you would like to buy a new house in Mumbai(Western Line) if you had to?':'n_Location',
        'What type of house you would look for if you had to buy a house?':'n_House_Type',  
        'What would be your budget if you had to buy a house?':'Budget',
        'How much carpet area would you want(square feet) if you had to buy a house?':'n_Carpet_Area',
        'How many rooms do you want if you had to buy a house?':'n_Bedrooms',
        'If you had to buy a house what would you buy?':'New/Resale'},axis=1)

st.markdown("<h1 style='text-align: center;'>House Price Prediction System</h1>", unsafe_allow_html=True)
option = st.selectbox("",["Prediction Model","House Loan Prediction","Data Cleaning","Exploratory Data Analysis",
                          "Feature Selection","Data Transformation","Regression Model","Classification Model"])

dataset1=dataset.copy()

dataset_with_null = dataset
dataset = dataset.dropna()


data = dataset

def impute_carparking(cols):
    CarParking = cols[0]
    if("Car Parking" in CarParking):
        return 1
    else:
        return 0

def impute_watersupply(cols):
    watersupply = cols[0]
    if("24hr Water Supply" in watersupply):
        return 1
    else:
        return 0
    
def impute_maintenance_staff(cols):
    maintenance_staff = cols[0]
    if("Maintenance Staff" in  maintenance_staff ):
        return 1
    else:
        return 0
    
def impute_Lift(cols):
    lift = cols[0]
    if("Lift" in lift):
        return 1
    else:
        return 0
    
def impute_gaspipeline(cols):
    gas_pipeline = cols[0]
    if("Gas Pipeline" in gas_pipeline ):
        return 1
    else:
        return 0

def impute_gym(cols):
    gym = cols[0]
    if("Gym" in gym):
        return 1
    else:
        return 0
    
def impute_club_house(cols):
    gym = cols[0]
    if("Club house" in gym):
        return 1
    else:
        return 0

def impute_24hr_security(cols):
    security = cols[0]
    if("24Hr Security" in security):
        return 1
    else:
        return 0
    
def impute_school(cols):
    school = cols[0]
    if("School" in school):
        return 1
    else:
        return 0

def impute_college(cols):
    college = cols[0]
    if("College" in college):
        return 1
    else:
        return 0
    
def impute_medical(cols):
    medical = cols[0]
    if("Medical" in medical):
        return 1
    else:
        return 0
    
def impute_hospitals(cols):
    hospitals = cols[0]
    if("Hospitals" in hospitals):
        return 1
    else:
        return 0
    
def impute_railwayStation(cols):
    railway = cols[0]
    if("Railway Station" in railway):
        return 1
    else:
        return 0

def impute_market(cols):
    market = cols[0]
    if("Market" in market):
        return 1
    else:
        return 0
    
def changeRooms(cols):
    rooms = cols
    if("BHK" in rooms):
        return int(rooms.split(' ')[0])
    else:
        return int(0)
    
data['c_Car Parking'] = data[['c_Car Parking']].apply(impute_carparking,axis=1)
data['c_24hr Water Supply'] = data[['c_24hr Water Supply']].apply(impute_watersupply,axis=1)
data['c_Maintenance Staff'] = data[['c_Maintenance Staff']].apply(impute_maintenance_staff,axis=1)
data['c_Lift'] = data[['c_Lift']].apply(impute_Lift,axis=1)
data['c_Gas Pipeline'] = data[['c_Gas Pipeline']].apply(impute_gaspipeline,axis=1)
data['c_Gym'] = data[['c_Gym']].apply(impute_gym,axis=1)
data['c_24Hr Security'] = data[['c_24Hr Security']].apply(impute_24hr_security,axis=1)
data['c_Club house'] = data[['c_Club house']].apply(impute_24hr_security,axis=1)

data['c_School'] = data[['c_School']].apply(impute_school,axis=1)
data['c_College'] = data[['c_College']].apply(impute_college,axis=1)
data['c_Medical'] = data[['c_Medical']].apply(impute_medical,axis=1)
data['c_Hospitals'] = data[['c_Hospitals']].apply(impute_hospitals,axis=1)
data['c_Railway Station'] = data[['c_Railway Station']].apply(impute_railwayStation,axis=1)
data['c_Market'] = data[['c_Market']].apply(impute_market,axis=1)


data['n_Car Parking'] = data[['n_Car Parking']].apply(impute_carparking,axis=1)
data['n_24hr Water Supply'] = data[['n_24hr Water Supply']].apply(impute_watersupply,axis=1)
data['n_Maintenance Staff'] = data[['n_Maintenance Staff']].apply(impute_maintenance_staff,axis=1)
data['n_Lift'] = data[['n_Lift']].apply(impute_Lift,axis=1)
data['n_Gas Pipeline'] = data[['n_Gas Pipeline']].apply(impute_gaspipeline,axis=1)
data['n_Gym'] = data[['n_Gym']].apply(impute_gym,axis=1)
data['n_24Hr Security'] = data[['n_24Hr Security']].apply(impute_24hr_security,axis=1)
data['n_Club house'] = data[['n_Club house']].apply(impute_24hr_security,axis=1)


data['n_School'] = data[['n_School']].apply(impute_school,axis=1)
data['n_College'] = data[['n_College']].apply(impute_college,axis=1)
data['n_Medical'] = data[['n_Medical']].apply(impute_medical,axis=1)
data['n_Hospitals'] = data[['n_Hospitals']].apply(impute_hospitals,axis=1)
data['n_Railway Station'] = data[['n_Railway Station']].apply(impute_railwayStation,axis=1)
data['n_Market'] = data[['n_Market']].apply(impute_market,axis=1)
data['No of Bedrooms'] =  data['No of Bedrooms'].apply(changeRooms)
#data['No of Bedrooms'].apply(changeRooms)
data['n_Bedrooms'] = data['n_Bedrooms'].apply(changeRooms)

predictionDataset = data.iloc[:,2:-22]
predictionDataset.drop({'Price','Income','Own_House'},axis=1,inplace=True)

dependent_var = data.iloc[:,8].values

x = predictionDataset.iloc[:,:].values

dependent_var = dependent_var.reshape(len(dependent_var),1)

sc_y = StandardScaler()

y = sc_y.fit_transform(dependent_var)

dummies_location = pd.get_dummies(data.Location)
dummies_house_located = pd.get_dummies(data.House_Located)
dummies_location = dummies_location.drop(['Virar'],axis='columns')
dummies_house_located = dummies_house_located.drop(['Bungalow'],axis='columns')
dummies = pd.concat([dummies_location,dummies_house_located],axis="columns")

x = pd.concat([dummies,predictionDataset],axis='columns')
x = x.drop(['Location','House_Located'],axis='columns')
x = pd.DataFrame(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

feature = ['Age', 'Gender', 'Location', 'Income','House_Located', 'No of Bedrooms',  'c_Car Parking',
          'c_24hr Water Supply', 'c_Maintenance Staff', 'c_Gas Pipeline',
          'c_Lift', 'c_Club house', 'c_24Hr Security', 'c_Gym', 'c_School',
          'c_College', 'c_Hospitals', 'c_Medical', 'c_Railway Station',
          'c_Market', ]


list(enumerate(feature))
# data1 = data.copy()
def totalCount():
    plt.figure(figsize = (15, 100))
    for i in enumerate(feature):
        data1 = data.copy()
        plt.subplot(10, 2,i[0]+1).set_title("Total Data Count",fontsize=13)
        sns.countplot(i[1], data = data1)
        plt.xticks(rotation = 50)
        plt.tight_layout()
    st.pyplot()

def locationPrice():
    #st.write("House Prices In Mumbai Western Line")   
    c= alt.Chart(data).mark_bar(size=20).encode(
    x='Location',
    y='Price',
    color='No of Bedrooms',
    tooltip = ['Carpet_Area'],
    order=alt.Order(
      # Sort the segments of the bars by this field
      'No of Bedrooms',
      sort='ascending'
    )
    ).properties(width = 900,height = 500)
    st.altair_chart(c)
        
def locationCarpetArea():
    fig = plt.figure(figsize = (19,8))
    sns.barplot(x="Carpet_Area", y="Price",data=data,ci=None)
    plt.title('Location vs Price',fontsize=20)
    plt.xticks(rotation = 90)
    plt.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig)
    
le = LabelEncoder()
data['House_Loan'] = le.fit_transform(data['House_Loan'])
def budgetLoan():
    fig = plt.figure(figsize = (15,8))
    sns.barplot(x='Budget',y='House_Loan',data=data,ci=None)
    plt.title('Budget vs House Loan',fontsize=15)
    plt.xlabel("Budget",fontsize=15)
    plt.ylabel("House Loan",fontsize=15)
    plt.xticks(rotation = 40);
    st.pyplot(fig)
    
def budgetCarpet():
    #st.write("House Prices In Mumbai Western Line")   
    # c= alt.Chart(data).mark_bar(size=20).encode(
    # x='Budget',
    # y='Carpet_Area',
    # color='No of Bedrooms',
    # #tooltip = ['Carpet_Area'],
    # order=alt.Order(
    #   # Sort the segments of the bars by this field
    #   'Budget',
    #   sort='ascending'
    # )
    # ).properties(width = 900,height = 500)
    # st.altair_chart(c)
    fig = plt.figure(figsize = (15,8))
    sns.barplot(x='Budget',y='n_Carpet_Area',hue='n_Bedrooms',data=data,ci=None)
    plt.title('Budget vs Carpet_Area',fontsize=15)
    plt.xlabel("Budget",fontsize=15)
    plt.ylabel("Carpet Area",fontsize=15)
    plt.xticks(rotation = 40);
    st.pyplot(fig)
    
def incomeLoan():
    fig = plt.figure(figsize = (13,8))
    sns.barplot(x='Income',y='House_Loan',data=data,ci=None)
    plt.title('Income vs House Loan',fontsize=15)
    plt.xlabel("Income",fontsize=15)
    plt.ylabel("House Loan",fontsize=15)
    plt.xticks(rotation = 40);
    st.pyplot(fig)

def incomeCarpet():
    sns.set_context(font_scale=1.5)
    fig = plt.figure(figsize = (13,8))
    sns.barplot(x='Income',y='n_Carpet_Area',data=data,ci=None)
    plt.title('Income vs Carpet Area',fontsize=15)
    plt.xlabel("Income",fontsize=15)
    plt.ylabel("Carpet Area",fontsize=15)
    plt.xticks(rotation = 40);
    st.pyplot(fig)

carpet_area = data['Carpet_Area'].unique()
np.sort(carpet_area)
def CarpetArea():
    sns.set_context(font_scale=1.5)
    fig = plt.figure(figsize=(10,10))
    sns.distplot(data['Carpet_Area'])
    plt.xlabel("Carpet Area",fontsize=15)
    plt.title('Variation in Carpet Area',fontsize=15)
    st.pyplot(fig)
    
def countLoan():
 
    fig = plt.figure(figsize = (25,15))
    sns.set_context(font_scale=1.5)
    sns.countplot(x='Carpet_Area',hue='House_Loan',data=data)
    plt.xlabel("Carpet Area",fontsize=15)
    plt.ylabel("House Loan",fontsize=15)
    plt.title('Count of House Loan',fontsize=15)
    plt.xticks(rotation = 90);
    st.pyplot(fig)
    
def carpetNbedrooms():
    sns.set_context(font_scale=1.5)
    fig = sns.lmplot(x='Carpet_Area',y='No of Bedrooms',data=data,aspect=2)
    plt.xlabel("Carpet Area",fontsize=15)
    plt.ylabel("No of Bedrooms",fontsize=15)
    plt.title('Carpet Area Vs No of Bedrooms',fontsize=15)
    st.pyplot(fig)

def BedroomsCarpetArea():
    sns.set_context(font_scale=1.5)
    fig = plt.figure(figsize=(15,8))
    sns.boxplot(x='No of Bedrooms',y='Carpet_Area',data=data,palette='rainbow')
    plt.xlabel("No of Bedrooms",fontsize=15)
    plt.ylabel("Carpet Area",fontsize=15)
    plt.title('No of Bedrooms vs Carpet Area',fontsize=15)
    st.pyplot(fig)
    
def CarpetAreaPrice():
    sns.set_context('paper',font_scale=1.5)
    fig = sns.lmplot(x='Carpet_Area',y='Price',data=data,aspect=2)
    plt.xlabel("Carpet Area",fontsize=15)
    plt.title('Carpet Area vs Price',fontsize=15)
    st.pyplot(fig)
        
def HouseOwners():
    fig = plt.figure(figsize = (15,8))
    plt.subplot(1,2,1)
    plt.title('Genderwise Count w.r.t House Owners',fontsize=15)
    sns.countplot(x="Gender", data=dataset1, hue='Own_House')
    plt.subplot(1,2,2)
    plt.title('Agewise Count w.r.t House Owners',fontsize=15)
    sns.countplot(x="Age", data=dataset1, hue='Own_House')
    st.pyplot(fig)
    
def NewOwners():
    fig = plt.figure(figsize = (15,7))
    plt.subplot(1,2,1)
    plt.title('Gender wise Count w.r.t New House Buyers',fontsize=15)
    sns.countplot(x="Gender", data=dataset1, hue='New_House')
    plt.subplot(1,2,2)
    plt.title('Age wise Count w.r.t New House Buyers',fontsize=15)
    sns.countplot(x="Age", data=dataset1, hue='New_House')
    st.pyplot(fig)

compare_features=['House_Located','n_House_Type','c_Car Parking','n_Car Parking','c_24hr Water Supply','n_24hr Water Supply',
         'c_Maintenance Staff','n_Maintenance Staff','c_Gas Pipeline','n_Gas Pipeline',
         'c_Lift','n_Lift', 'c_Club house','n_Club house', 'c_24Hr Security','n_24Hr Security',
         'c_Gym','n_Gym', 'c_School', 'n_School','c_College','n_College' ,'c_Hospitals','n_Hospitals',
         'c_Medical','n_Medical' ,'c_Railway Station','n_Railway Station','c_Market','n_Market']
        
def OldFeatureNewFeature(compare_features):
    fig = plt.figure(figsize = (15,140))
    for i in enumerate(compare_features):
        data1 = data.copy()
        plt.subplot(15, 2,i[0]+1)
        sns.countplot(i[1], data = data1)
        plt.xticks(rotation = 50)
        plt.tight_layout()
    st.pyplot(fig)
   
FinalData = data.iloc[:,6:23] 
def FinalDataCorr():
    fig = plt.figure(figsize=(16,10))
    sns.heatmap(FinalData.corr(), annot=True)
    sns.set_style('white')
    plt.title('Correlation Between Features',fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    st.pyplot(fig)
        
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(abs(corr_matrix.iloc[i, j]) > threshold):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
  
# Mutiple Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# SVR
y_train1 = y_train.flatten()
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train,y_train1)

y_pred_svr = svr_regressor.predict(X_test)

# decision tree regression
tree_regressor = DecisionTreeRegressor(random_state = 1)
tree_regressor.fit(X_train, y_train)

#Random forest regression
forest_regressor = RandomForestRegressor(n_estimators = 20, random_state = 1)
forest_regressor.fit(X_train, y_train1)

y_pred_forest = forest_regressor.predict(X_test)

y_pred_tree = tree_regressor.predict(X_test)

#Classification
    
target_var = dataset1.iloc[:,0:4]
target_var = target_var.drop('Location',axis=1)
target_var1 = target_var
dummies_age = pd.get_dummies(target_var.Age)
dummies_age = dummies_age.drop(['56 - 65'],axis='columns')
dummies_income = pd.get_dummies(target_var.Income)
dummies_income = dummies_income.drop(['11lakh and above'],axis='columns')
target_dummies = pd.concat((dummies_age,dummies_income),axis='columns')
le = LabelEncoder()
target_var['Gender'] = le.fit_transform(target_var['Gender'])
target_var = pd.concat((target_dummies,target_var),axis='columns')
target_var = target_var.drop(['Age','Income'],axis='columns')

dataset1['House_Loan'] = le.fit_transform(dataset1['House_Loan'])
y_c = dataset1.iloc[:,24].values

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(target_var,y_c,test_size=0.2,random_state=0)

knn_classifier = KNeighborsClassifier(n_neighbors=23)
knn_classifier.fit(X_train_c,y_train_c)

knn_pred = knn_classifier.predict(X_test_c)


knn_accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,target_var,y_c,cv=10)
    knn_accuracy_rate.append(score.mean())

knn_error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,target_var,y_c,cv=10)
    knn_error_rate.append(1-score.mean())


randomforest_Classifier = RandomForestClassifier(n_estimators=20)
randomforest_Classifier.fit(X_train_c, y_train_c)


y_predicted_random = randomforest_Classifier.predict(X_test_c)
cm = confusion_matrix(y_test_c, y_predicted_random)




def predict_price(location,house_located,capet_area,No_of_Bedrooms, c_Car_Parking,c_24hr_Water_Supply,
       c_Maintenance_Staff, c_Gas_Pipeline, c_Lift, c_Club_house,
       c_24Hr_Security, c_Gym, c_School, c_College, c_Hospitals,
       c_Medical, c_Railway_Station, c_Market):
    l = []
    loc = dummies_location.columns==location
    rooms = 0
    for i in loc:
      if(i):
        l.append(1)
      else:
        l.append(0)
  
    house_loc = dummies_house_located.columns==house_located
    for i in house_loc:
      if(i):
        l.append(1)
      else:
        l.append(0)
    
    
    l.append(capet_area)
    
    if(No_of_Bedrooms == "1 RK"):
        l.append(0)
    else:
        if("BHK" in No_of_Bedrooms):
            rooms=int(No_of_Bedrooms.split(' ')[0])
            l.append(rooms)
    
    if(c_Car_Parking):
        l.append(1)
    else:
        l.append(0)
    
    if(c_24hr_Water_Supply):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Maintenance_Staff):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Gas_Pipeline):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Lift):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Club_house):
        l.append(1)
    else:
        l.append(0)
        
        
    if(c_24Hr_Security):
        l.append(1)
    else:
        l.append(0)

        
    if(c_Gym):
        l.append(1)
    else:
        l.append(0)
                
    if(c_School):
        l.append(1)
    else:
        l.append(0)
        
    if(c_College):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Hospitals):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Medical):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Railway_Station):
        l.append(1)
    else:
        l.append(0)
        
    if(c_Market):
        l.append(1)
    else:
        l.append(0)
    
    predicted_value = sc_y.inverse_transform(forest_regressor.predict([l]))
    return round(predicted_value[0],2)





def predict_knn(age,salary,gender):
    l = []
    age_list = dummies_age.columns==age
    sal_list = dummies_income.columns==salary
    
    for i in age_list:
      if(i):
        l.append(1)
      else:
        l.append(0)
    
    for i in sal_list:
      if(i):
        l.append(1)
      else:
        l.append(0)
        
    if(gender == "Male"):
        l.append(0)
    else:
        l.append(0)
        
    pred = knn_classifier.predict([l])
    
    return pred[0]

if(option == "Data Cleaning"):
    st.write("## Data Collected From The Survey")
    display_original_dataset()
    
    st.write("## Changing dataset columns")
    st.write(dataset2)
    
    col1, col2 = st.beta_columns(2)
        
    col1.write("## Count of null values")
    col1.text(dataset_with_null.isnull().sum())
   
    col2.write("## Removing all the null values")
    col2.text(dataset.isnull().sum())
    
    st.write("Since the missimg data cannot be manipulated by any means, we need to remove the null values")
    
    st.write("## Converting all the categorial data into integer")
    st.write(data)

    st.write('After removing null values the dataset contains ',dataset.shape[1],' columns and ',dataset.shape[0],' rows')

elif(option == "Exploratory Data Analysis"):
    st.sidebar.title("Graphs")
    
    selectbox = st.sidebar.radio(label="", options=["Comparision of Price w.r.t Features", "Total Data Count", "House Prices In Mumbai (Western Line)",
    "Boxplot Graph","Location vs Price","Analysis on Budget","Analysis on Income","Variation in Carpet Area","Count of House Loan",
    "Regression Graph","House Owners And House Buyer's strength",
    "Comaprision Between Current Features and New Features"])
    if selectbox == "Comparision of Price w.r.t Features":
       numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
       data[numerical_features].head()
       c_features = data[numerical_features].iloc[:,:-16]
       c_features.drop(['Price','Carpet_Area','House_Loan'],axis='columns',inplace=True)
       c_features.head()
       def hp_features():
           st.set_option('deprecation.showPyplotGlobalUse', False)
           for feature in c_features:
               data1=data.copy()
               data1.groupby(feature)['Price'].mean().plot.bar()
               plt.xlabel(feature)
               plt.ylabel('Price')
               plt.title("Comparision of Price w.r.t Features")
               st.pyplot()
       hp_features()
   
       st.write("From the above graphs we can conclude that features such as No of BedRooms, 24Hr Water Supply, Gas Pipeline, Lift and medical are highly influenced for increase in price whereas other features does not affect price as much ")
    
    elif selectbox == "Total Data Count":   
        
        st.write("Count Plot shows the counts of observations in each categorical data using bars")
        totalCount()
                
    elif selectbox == "House Prices In Mumbai (Western Line)":
        numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
        data[numerical_features].head()
        scatter_features = data[numerical_features].iloc[:,:-16]
        scatter_features['Location'] = data['Location']
        scatter_features.head()
        def housePrice():
            st.write("House Prices In Mumbai Western Line")
            c = alt.Chart(scatter_features).mark_point(size = 240).encode(
                x='No of Bedrooms', y='Price',color = 'Location', tooltip=['Carpet_Area','Location','Price','c_Car Parking','c_24hr Water Supply','c_Maintenance Staff','c_Lift','c_Gas Pipeline','c_Gym',
                             'c_24Hr Security','c_School','c_College','c_Hospitals','c_Medical','c_Railway Station','c_Market']).interactive().properties(width = 900,height= 600)
            st.altair_chart(c)
        housePrice()
                
    elif selectbox == "Location vs Price":

        locationPrice()
        
        st.markdown("""<style>
                        table {
                          border-collapse: collapse;
                          width: 100%;
                        }
                        th, td {
                          text-align: left;
                          padding: 8px;
                        }            
                        th {
                          background-color: #FCFCFC;
                          color: black;
                        }
                        #th_border th {
                          border-color: black;
                        }
                    </style>""",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center';>Analysis on Location with respect to Price</h2>",unsafe_allow_html=True)
        st.markdown("""<table style='width:100%' id='th_border'>
            <tr>
                <th colspan="3" style='text-align: center'>1 RK</th>
            </tr>
            <tr>
                <th>High</th>
                <th>Mid</th>
                <th>Low</th>
            </tr>
            <tr>
                <td>Marine Lines</td>
                <td>Mahim</td>
                <td>Dahisar</td>
            </tr>
             <tr>
                <td>Bandra</td>
                <td>Khar</td>
                <td>Bhayandar</td>
            </tr>
            <tr>
                <th colspan="3" style='text-align: center'>1 BHK</th>
            </tr>
            <tr>
                <th>High</th>
                <th>Mid</th>
                <th>Low</th>
            </tr>
             <tr>
                <td>Khar</td>
                <td>Churchgate</td>
                <td>Bhayandar</td>
            </tr>
             <tr>
                <td>Lower Parel</td>
                <td>Andheri</td>
                <td>Borivali</td>
            </tr>            
             <tr>
                <td rowspan ="9"></td>
                <td>Marine Lines</td>
                <td>Dahissar</td>                
            </tr>
            <tr>
                <td>Grant Road</td>
                <td>Jogeshwari</td>
            <tr>
                <td>Vile Parle</td>
                <td>Kandivali</td>
            </tr>
            <tr>
                <td>Bandra</td>
                <td>Mahalaxmi</td>
            </tr>
            <tr>
                <td>Prabhadevi</td>
                <td>Malad</td>
            </tr>
            <tr>
                <td>Dadar</td>
                <td>Mumbai Central</td>
            </tr>
            <tr>
                <td>Mahin</td>
                <td>Nallasopara</td>
            </tr>
            <tr>
                <td>Virar</td>       
                <td rowspan="2"></td>
            </tr>
            <tr>
                <td>Santacruz</td>               
            </tr>
            <tr>
                <th colspan="3" style='text-align: center'>2 BHK</th>
            </tr>
                <tr>
                    <th>High</th>
                    <th>Mid</th>
                    <th>Low</th>
                </tr>
                 <tr>
                    <td>Marine Lines</td>
                    <td>Bandra</td>
                    <td>Bhayandar</td>
                </tr>
                 <tr>
                    <td>Mahalaxmi</td>
                    <td>Dadar</td>
                    <td>Borivali</td>
                </tr>
                 <tr>
                    <td>Churchagte</td>
                    <td>Grant Road</td>
                    <td>Dahisar</td>
                </tr>
                 <tr>
                    <td>Khar</td>
                    <td>Jogeshwari</td>
                    <td>Kandivali</td>
                </tr>
                 <tr>
                    <td>Mahim</td>
                    <td>Lower Parel</td>
                    <td>Malad</td>
                </tr>
            <tr>
                <td rowspan ="4">
                <td>Matunga Road</td>
                <td>Miraroad</td>
            </tr>
                <tr>
                <td>Prabhadevi</td>
                <td>Santacruz</td>
                </tr>
                <tr>
                <td>Vile Parle</td>
                <td>Virar</td>
                </tr>
                <tr>
                <td>Goregaon</td>
                <td></td>
                </tr>
                 <tr>
                <th colspan="3" style='text-align: center'>3 BHK</th>
                </tr>
                <tr>
                    <th>High</th>
                    <th>Mid</th>
                    <th>Low</th>
                </tr>
                <tr>
                    <td>Dadar</td>
                    <td>Andheri</td>
                    <td>Bhayandar</td>
                </tr>
                <tr>
                    <td>Lower Parel</td>
                    <td>Borivali</td>
                    <td>Malad</td>
                </tr>
                <tr>
                    <td>Mahalaxmi</td>
                    <td>Goregaon</td>
                    <td>Nallasopara</td>
                </tr>
                <tr>
                    <td>Grant Road</td>
                    <td>Jogeshwari</td>
                    <td rowspan="5"></td>
                </tr>
                <tr>
                    <td>Matunga Road</td>
                    <td>Kandivali</td>
                </tr>
                <tr>
                    <td>Prabhadevi</td>
                    <td>Khar</td>
                </tr>
                <tr>
                    <td>Santacruz</td>  
                    <td rowspan="2"></td>
                </tr>
                <tr>
                    <td>Vile Parle</td>    
                </tr>
                <tr>
                <th colspan="3" style='text-align: center'>4 BHK</th>
                </tr>
                <tr>
                    <th>High</th>
                    <th>Mid</th>
                    <th>Low</th>
                </tr>
                <tr>
                    <td>Marine Lines</td>
                    <td>Dadar</td>
                    <td>Malad</td>
                </tr>
                <tr>
                    <td colspan="1" >
                    <td>Vile Parle</td>  
                    <td></td>
                </tr></table>""",unsafe_allow_html=True)
        # locationCarpetArea()
                    
    elif selectbox == "Analysis on Budget":

        budgetLoan()
        st.write("From the above graph we can observe that people have budget below 1 cr and above 6cr have the highest changes to take a house loan")
        
        budgetCarpet()
        st.write("From the above graph we conclude that:")
        st.write("Budget greater than 1Cr target 1,2,3 and 4 bhk with higher carpet area,ecxept budget range between 5Cr-7Cr target 3 and 4bhk with higher carpet area")
        st.write("Budget range between 50lakh and 90Lakh target 1,2 and 3bhk with higher carpet area ")
        st.write("Budget range less that 40lakh target 0,1 and 2bhk with medium carpet area")
    
    elif selectbox == "Analysis on Income":
        st.write("## Income vs House Loan")
        incomeLoan()
        st.write("From the above graph we can observe that people having income between 1 lakh - 3 lakh, 9 lakh - 11 lakh and 11 lakh and above have above 70% changes to take a house loan")
            
        st.write("## Income vs Carpet Area")
        incomeCarpet()
                        
    elif selectbox == "Variation in Carpet Area":

        CarpetArea()
        st.write("From the above graph we can observe that the dataset contains carpet area majority between  300 to 1000 sq.ft")
        
    elif selectbox == "Count of House Loan":
        
        countLoan()
        st.write("From the above graph we can observe that majority of the people where willing to take the house loan")

    elif selectbox == "Boxplot Graph":
        BedroomsCarpetArea()
        st.write("From the above graph we can observe that there are some outliers in carpet area with respect to rooms such as in 0 No. of BedRooms i.e 1 RK range of carpet area lies between 100 to 800 therefore house having carpet area 3500 is a outlier, in 1 BK range of carpet area lies between 200 and 760 therefore we have 3 higher outliers with carpet area 1000,1010 and 2500, in 2 BK range of carpet area lies between 480 and 1500 therefore we have 1 higher outliers with carpet area 1750, in 3 BK range of carpet area lies between 900 and 2050 therefore we have 1 higher outliers with carpet area 2400, in 4 BK range of carpet area lies between 1990 and 2350 therefore we have 1 higher outliers with carpet area 3500 and one lower outlier 100.")
       
    elif selectbox == "Regression Graph":
        st.write("No of Bedrooms vs Carpet Area")
        
        carpetNbedrooms()
        st.write("From the above graph we can conclude that there is a linear relation between the No. of Bedrooms and Carpet Area as No. of Bedrooms increases Carpet Area also increases but the points do not fit on the regression line.")
       
        
        st.write("## Carpet Area vs Price")
        CarpetAreaPrice()
        st.write("From the above graph we can conclude that there is a linear relation between the House Price and Carpet Area as House Price increases Carpet Area also increases but the points do not fit on the regression line.")
            
    elif selectbox == "House Owners And House Buyer's strength":

        HouseOwners()

        NewOwners()
        
    elif selectbox == "Comaprision Between Current Features and New Features":
        
        
        OldFeatureNewFeature(compare_features)
        
        st.write("From the above graphs we can conclude that except Gym and Club House majority of the people wanted all the rest of the features")
        
elif(option == "Feature Selection"):
    st.write("## Feature Selection- With Correlation")    
    corr_features = correlation(predictionDataset,0.85)
    st.write(" Highly Corelated Feature")    
    corr_features
    predictionDataset.drop('c_Club house',axis=1)
    st.write("Heat map to show Highly Correlated Data")    
    FinalDataCorr()
    st.write("From the graph we conclude that there is strong correlation between Club House and 24Hr Security so we drop Club House since club house is not a important feature to select")
    st.write("Droping the corelated feature")
    st.write(data.head())
    
elif(option == "Data Transformation"):
    st.title("Data Transformation for prediction model")
    st.write(" Dependent Variable (House Price)")
    st.write(dependent_var)
    st.write(" Scaling the Dependent Variable")
    st.write(y)
    st.write(" Prediction Dataset")
    st.write(predictionDataset.head())
    st.write(" Encoding the Dataset")
    st.write(x.head(15))
    st.write("## Splitting Data into Test and Trainning Set")
    X_train.shape , X_test.shape
    
    st.title("Data Transformation for classification model")
    st.write("## Dataset")
    st.write(dataset1.head(3))
    
    st.write("## Target Variable")    
    st.write(target_var.head())
    
    st.write("## Performaing one hot encodimg on age column")   
    st.write(dummies_age.head())
    
    st.write("## Performing one hot encoding on income column")    
    st.write(dummies_income.head())
    
    
    st.write("## Concatenationg the columns")
    st.write(target_dummies)
    
    
    st.write("## Label Encoding on Gender Column") 
    st.write(target_var)
    
    
    st.write("## Displaying the final Dataset")
    target_var
    
    
    st.write("## Label Encoding on House Loan Column")     
    st.write(y_c)
    
    y = dataset1.iloc[:,24].values
    
    st.write("## Splitting dataset into Test and Train Data")
    st.write(X_train_c.shape, X_test_c.shape)
        
    
    
elif(option == "Regression Model"):
    st.sidebar.title("Regression Models")
    selectbox = st.sidebar.radio(label="", options=["Multiple Linear Regression","Support Vector Regression",
                "Decision Tree Regression","Random Forest Regression"])
    
    if(selectbox == "Multiple Linear Regression"):
        st.title("Multiple Linear Regression")
               
        st.write("Multiple regression is an extension of simple linear regression. It is used when we want to predict the value of a variable based on the value of two or more other variables. The variable we want to predict is called the dependent variable (or sometimes, the outcome, target or criterion variable)")
        linear_y_pred = linear_regressor.predict(X_test)
        
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write("0 stands for actual value and 1 stands for predicted value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(linear_y_pred).reshape(len(linear_y_pred),1)),axis=1))
        st.write("From the above observation we conclude that there is a huge difference between the actual and predicted value.")         
        st.write('R^2 Score:', metrics.r2_score(y_test,linear_y_pred))
        st.write("The model score is 60% that means 60% of the data fit the regression model,since the r-squared score is not close to 1, so the model does not fit best")
            
    elif(selectbox == "Support Vector Regression"):
        st.title("Support Vector Regression")
        st.write("Supervised Machine Learning Models with associated learning algorithms that analyze data for classification and regression analysis are known as Support Vector Regression. SVR is built based on the concept of Support Vector Machine or SVM.")
                
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write("0 stands for actual value and 1 stands for predicted value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_svr).reshape(len(y_pred_svr),1)),axis=1))        
        st.write("From the above observation we conclude that there is a huge difference between the actual and predicted value.") 
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_svr))
        st.write("The model score is 50% that means only 50% of the data fit the regression mode, since the r-squared score is not close to 1, so the model does not fit best")       
    
    elif(selectbox == "Decision Tree Regression"):
        st.title("Decision Tree Regression")
        st.write("Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. Decision trees can handle both categorical and numerical data.")
        tree_regressor = DecisionTreeRegressor(random_state = 1)
        tree_regressor.fit(X_train, y_train1)
        y_pred_tree = tree_regressor.predict(X_test)


        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write("0 stands for actual value and 1 stands for predicted value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_tree).reshape(len(y_pred_tree),1)),axis=1))
        st.write("From the above observation we conclude that there is a huge difference between the actual and predicted value.") 
        #st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_tree))
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_tree))
        st.write("The model score is 40% that means only 40% of the data fit the regression model, the model score is less than 50%,since the r-squared score is not close to 1, so the model does not fit best ")      
    
    elif(selectbox == "Random Forest Regression"):
        st.title("Random Forest Regression")
        st.write("Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. A Random Forest operates by constructing several decision trees during training time and outputting the mean of the classes as the prediction of all the trees.")        
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write("0 stands for actual value and 1 stands for predicted value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_forest).reshape(len(y_pred_forest),1)),axis=1))
        st.write("From the above observation we conclude that there is a less difference between the actual and predicted value.") 
        #st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_forest))
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_forest))
        st.write("The model score is 75% that means 75% of the data fit the regression model, since the r-squared score is close to 1 the model fits best")
        
elif(option == "Classification Model"):
    st.sidebar.title("Classification Models")
   
    radio = st.sidebar.radio(label = "", options = ["Random Forest Classifier","K Nearest Neighbors Classifier"])
     
    if(radio == "Random Forest Classifier"):        
                
        st.write("# Random Forest Classifier")
        st.write("Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction")
        
        # st.write("## Confusion Matrix")
        # st.write(cm)
        
        st.write("## Accuracy Score")
        st.write(accuracy_score(y_test_c,y_predicted_random))
        
        st.write("## Classification Report")
        st.write(classification_report(y_test_c,y_predicted_random))
        
        st.write("## Heat Map to show Confusion Matirx")
        fig = plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        st.pyplot(fig)
        
    elif(radio == "K Nearest Neighbors Classifier"):
        
        # st.write("## Splitting dataset into Test and Train Data")
        # st.write(X_train_c.shape, X_test_c.shape)
        
        # st.write("## Model Score")
        # st.write(knn.score(X_test, y_test))
        st.write("# K Nearest Neighbors Classifier")
        st.write("K-Nearest Neighbors (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function). Classification is done by a majority vote to its neighbors")
        
        st.write("## Accuracy Score")
        st.write(accuracy_score(y_test_c, knn_pred))
        
        # st.write("## Confusion Matrix")
        # st.write(confusion_matrix(y_test_c,knn_pred))
        
        st.write("## Classification Report")
        st.write(classification_report(y_test_c,knn_pred))
        
        st.write("## Heat Map to show Confusion Matirx")
        fig = plt.figure(figsize=(8,4))
        sns.heatmap(confusion_matrix(y_test_c,knn_pred), annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        st.pyplot(fig)
    
        st.write("## Choosing the K value")
        st.write("K Value = 23")
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1,40),knn_error_rate,color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        st.pyplot(fig)
        
        st.write("## Checking the accuracy of K = 23")
        
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1,40),knn_accuracy_rate,color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Accuracy vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Accuracy Rate')
        st.pyplot(fig)
        
        # st.write("## Choosing the K value")
        # st.write("K Value = 23")
        # fig = plt.figure(figsize=(10,6))
        # plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
        #          markerfacecolor='red', markersize=10)
        # plt.title('Error Rate vs. K Value')
        # plt.xlabel('K')
        # plt.ylabel('Error Rate')
        # st.pyplot(fig)
        
        # st.write("## Checking the accuracy of K = 23")
        
        # fig = plt.figure(figsize=(10,6))
        # plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
        #          markerfacecolor='red', markersize=10)
        # plt.title('Accuracy vs. K Value')
        # plt.xlabel('K')
        # plt.ylabel('Accuracy Rate')
        # st.pyplot(fig)
   
elif (option == "Prediction Model"):
    st.write("## Location in Mumbai Western Line")
    with st.form(key='prediction_form'):
        locations = ["Andheri","Bandra","Bhayandar","Borivali","Churchgate","Dadar","Dahisar","Goregaon","Grant Road","Jogeshwari","Kandivali",	
        "Khar","Lower Parel","Mahalaxmi","Mahim","Malad","Marine Lines","Matunga Road","Mira Road","Mumbai Central","Nallasopara", "Prabhadevi","Santacruz","Vile Parle"]
        locations.sort()
        location = st.selectbox(label="", options=locations)
     
        
        st.write("## House Type")
        house_located = st.selectbox(label="", options=["Building","Bunglow","Chawl","Row House"])
        
        st.write("## Carpet Area")
        carpet_area = st.text_input(label = "Enter Carpet Area in Square Fit")
        
        st.write("## No of Bedrooms")
        No_of_Bedrooms = st.radio(label = "Select number of Rooms", options=["1 RK","1 BHK","2 BHK","3 BHK","4 BHK"])
        
        st.write("## Facilities")
        c_Car_Parking = st.checkbox("Car Parking")
        c_24hr_Water_Supply = st.checkbox("24hr Water Supply")
        c_Maintenance_Staff = st.checkbox("Maintenance Staff")
        c_Gas_Pipeline = st.checkbox("Gas Pipeline")
        c_Lift = st.checkbox("lift")
        c_Club_house = st.checkbox("Club house")
        c_24Hr_Security = st.checkbox("24Hr Security")	
        c_Gym = st.checkbox("Gym")
           
        st.write("## Area Facility")
        c_School = st.checkbox("School")
        c_College = st.checkbox("College")
        c_Hospitals = st.checkbox("Hospital")
        c_Medical = st.checkbox("Medical")
        c_Railway_Station = st.checkbox("Railway Station")
        c_Market = st.checkbox("Market")
        submit = st.form_submit_button('Predict Price')
        
    if(submit):

        try:        
            if(not carpet_area):
                st.write("Please enter the carpet area")
            elif(int(carpet_area)>=100 and int(carpet_area)<=6000):
                predicted_price = predict_price(location,house_located,carpet_area,No_of_Bedrooms, c_Car_Parking,c_24hr_Water_Supply,
               c_Maintenance_Staff, c_Gas_Pipeline, c_Lift, c_Club_house,
               c_24Hr_Security, c_Gym, c_School, c_College, c_Hospitals,
               c_Medical, c_Railway_Station, c_Market)
                
                price_in_words = num2words(predicted_price, to='currency', lang='en_IN')
                
                st.write("The cost of the house will be ",predicted_price," that is "+price_in_words.title().replace("Euro", "Rupees").replace("Cents", "Paise"))
                st.write('The accurace of the model is ', round(metrics.r2_score(y_test,y_pred_forest)*100),"%")
            else:
                st.write("Please enter a valid carpet area")
        except:
            st.write("Please enter a valid carpet area")
        
elif(option == "House Loan Prediction"):
    with st.form(key='classification_form'):
        
       
        age_grp_list = dummies_age.columns.values
        age_grp_list = np.append(age_grp_list,'56 - 65')
        
        income_list = dummies_income.columns.values
        income_list = np.append(income_list,'11lakh and above')
        
        st.write("## Age Group")
        age = st.selectbox(label="",options=age_grp_list)
        
        st.write('## Income Range')
        salary = st.selectbox(label='', options=income_list)
        
        st.write("## Gender")
        check_gender = st.radio(label="", options=["Male","Female"])
        
        submit = st.form_submit_button('Predict House Loan')
    
    if(submit):
       pred = predict_knn(age,salary,check_gender)
       if(pred == 1):
           st.write("The Buyer Will Take A Loan")
       else:
           st.write("The Buyer Will Not Take A Loan")

       
        
        
   
    
    
    
     
    
        

        
        
                    
        
        







        




        
 
       
      
     

