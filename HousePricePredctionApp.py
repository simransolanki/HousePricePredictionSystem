# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:59:54 2021

@author: Nimish
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
import plotly.express as px
from num2words import num2words
sns.set(color_codes=True)



og_dataset = pd.read_csv('Maindataset.csv')
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

st.markdown("<h1 style='text-align: center;'>House Price Prediction System</h1>", unsafe_allow_html=True)
option = st.selectbox("",["Prediction Model","Data Cleaning","Exploratory Dtata Analysis",
                          "Data Corelation","Data Transformation","Regression Model","Classification Model"])

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
data['No of Bedrooms'] = data['No of Bedrooms'].apply(changeRooms)
data['n_Bedrooms'] = data['n_Bedrooms'].apply(changeRooms)

predictionDataset = data.iloc[:,2:-22]
predictionDataset.drop({'Price','Income','Own_House'},axis=1,inplace=True)
dependent_var = data.iloc[:,8].values
x = predictionDataset.iloc[:,:].values
dependent_var = dependent_var.reshape(len(dependent_var),1)
sc_y = StandardScaler()
y = sc_y.fit_transform(dependent_var)


# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0,1])],remainder="passthrough")
# x = np.array(ct.fit_transform(x))

dummies_location = pd.get_dummies(data.Location)
dummies_house_located = pd.get_dummies(data.House_Located)
dummies_location = dummies_location.drop(['Virar'],axis='columns')
dummies_house_located = dummies_house_located.drop(['Bungalow'],axis='columns')
dummies = pd.concat([dummies_location,dummies_house_located],axis="columns")
x = pd.concat([dummies,predictionDataset],axis='columns')
x = x.drop(['Location','House_Located'],axis='columns')

x = pd.DataFrame(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

featureScale = preprocessing.StandardScaler()
featuresScale = featureScale.fit_transform(y_train)
featuresScale = featuresScale.flatten()

feature = ['Age', 'Gender', 'Location', 'Income','House_Located', 'No of Bedrooms',  'c_Car Parking',
          'c_24hr Water Supply', 'c_Maintenance Staff', 'c_Gas Pipeline',
          'c_Lift', 'c_Club house', 'c_24Hr Security', 'c_Gym', 'c_School',
          'c_College', 'c_Hospitals', 'c_Medical', 'c_Railway Station',
          'c_Market', ]

list(enumerate(feature))

def totalCount():
    plt.figure(figsize = (15, 100))
    for i in enumerate(feature):
        data1 = data.copy()
        plt.subplot(10, 2,i[0]+1).set_title("Total Data Count",fontsize=13)
        sns.countplot(i[1], data = data1)
        plt.xticks(rotation = 50)
        plt.tight_layout()
    st.pyplot()

def bedPrice():
    fig = sns.jointplot(x='No of Bedrooms',y='Price',data=data,kind='hex')
    plt.title('No of Bedrooms vs Price',fontsize=15)
    plt.tight_layout()
    st.pyplot(fig)
    
def locationPrice():
    fig = plt.figure(figsize = (15,8))
    sns.barplot(x="Location", y="Price",hue='No of Bedrooms',data=data,ci=None)
    plt.title('Location vs Price',fontsize=20)
    plt.xticks(rotation = 90);
    st.pyplot(fig)
    
def cl_nl():
    fig = sns.relplot(x="Location", y="n_Location",data=data,sizes=(40, 400), alpha=1, palette="muted",height=10)
    plt.title('Current Location vs New Location',fontsize=15)
    plt.xlabel("Current Location",fontsize=15)
    plt.ylabel("New Location",fontsize=15)
    plt.xticks(rotation = 40);
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
    sns.barplot(x='Income',y='Carpet_Area',data=data,ci=None)
    plt.title('Income vs Carpet Area',fontsize=15)
    plt.xlabel("Income",fontsize=15)
    plt.ylabel("Carpet Area",fontsize=15)
    plt.xticks(rotation = 40);
    st.pyplot(fig)

def incomeNlocation():
    sns.set_context(font_scale=1.5)
    fig = plt.figure(figsize=(10,10))
    plt.title('Income vs Location',fontsize=15)
    plt.xlabel("Income",fontsize=15)
    plt.ylabel("Location",fontsize=15)
    sns.scatterplot(x='Income',y='Location',data=data)
    st.pyplot(fig)

def incomeLocation():
    sns.set_context(font_scale=1.5)
    fig = plt.figure(figsize=(10,10))
    sns.scatterplot(x='Income',y='n_Location',data=data)
    plt.xlabel("Income",fontsize=15)
    plt.ylabel("New Location",fontsize=15)
    plt.title('Income vs New Location',fontsize=15)
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
    fig = plt.figure(figsize = (25, 110))
    for i in enumerate(compare_features):
        data1 = data.copy()
        plt.subplot(15, 2,i[0]+1)
        sns.countplot(i[1], data = data1).set_title("Comaprision Between Current Features and New Features",fontsize=15)
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
featureScale = preprocessing.StandardScaler()
featuresScale = featureScale.fit_transform(y_train)
featuresScale = featuresScale.flatten()

svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train,featuresScale)

y_pred_svr = svr_regressor.predict(X_test)

# decision tree regression
tree_regressor = DecisionTreeRegressor(random_state = 1)
tree_regressor.fit(X_train, featuresScale)

#Random forest regression
forest_regressor = RandomForestRegressor(n_estimators = 20, random_state = 1)
forest_regressor.fit(X_train, featuresScale)

y_pred_forest = forest_regressor.predict(X_test)

y_pred_tree = tree_regressor.predict(X_test)

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

if(option == "Data Cleaning"):
    st.write("## Data Collected From The Survey")
    display_original_dataset()
    
    st.write("## Changing dataset columns")
    st.write(dataset)
    
    st.write("## Dataset overview")
    st.write(dataset.info())
    
    col1, col2 = st.beta_columns(2)
    
    
    col1.write("## Count of null values")
    col1.text(dataset_with_null.isnull().sum())
   
    col2.write("## Removing all the null values")
    col2.text(dataset.isnull().sum())
    
    st.write("## Converting all the categorial data into integer")
    st.write(data)

    st.write('After removing null values the dataset contains ',dataset.shape[1],' columns and ',dataset.shape[0],' rows')

elif(option == "Exploratory Dtata Analysis"):
    st.sidebar.title("Pages")
    
    selectbox = st.sidebar.radio(label="", options=["Comparision of Price w.r.t Features", "Total Data Count", "House Prices In Mumbai (Western Line)",
    "No of Bedrooms vs Price","Location vs Price","Current Location vs New Location","Budget vs House Loan","Income vs Features",
    "Variation in Carpet Area","Count of House Loan",
    "Carpet Area Vs No of Bedrooms","No of Bedrooms vs Carpet Area","Carpet Area vs Price","House Owners And House Buyer's strength",
    "Comaprision Between Current Features and New Features","Correlation Between Features"])
    if selectbox == "Comparision of Price w.r.t Features":
        
       numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
       data[numerical_features].head()
       c_features = data[numerical_features].iloc[:,:-16]
       c_features.drop(['Price','Carpet_Area'],axis='columns',inplace=True)
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
   
    elif selectbox == "Total Data Count":   
        
        totalCount()
               
    elif selectbox == "House Prices In Mumbai (Western Line)":
        numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
        data[numerical_features].head()
        scatter_features = data[numerical_features].iloc[:,:-16]
        scatter_features['Location'] = data['Location']
        scatter_features.head()
        def housePrice():
            price_scatter= px.scatter(scatter_features, x='Price', y='No of Bedrooms',title='House Prices In Mumbai (Western Line)',
                    hover_name='Location',color='Location', size_max=80, 
                    hover_data=['c_Car Parking','c_24hr Water Supply','c_Maintenance Staff','c_Lift','c_Gas Pipeline','c_Gym',
                            'c_24Hr Security','c_School','c_College','c_Hospitals','c_Medical','c_Railway Station','c_Market'], 
                    width=900,height=800)
            price_scatter.show()
            #st.pyplot(price_scatter)
        housePrice()
    
    elif selectbox == "No of Bedrooms vs Price":

        bedPrice()
        
    elif selectbox == "Location vs Price":

        locationPrice()
        
    elif selectbox == "Current Location vs New Location":

        cl_nl()
        
    elif selectbox == "Budget vs House Loan":

        budgetLoan()
        
    elif selectbox == "Income vs Features":

        incomeLoan()
        incomeCarpet()
        incomeNlocation()
        incomeNlocation()
        incomeLocation()
                                 
    elif selectbox == "Variation in Carpet Area":

        CarpetArea()
        
    elif selectbox == "Count of House Loan":
        
        countLoan()

    elif selectbox == "Carpet Area Vs No of Bedrooms":

        carpetNbedrooms()
        
    elif selectbox == "No of Bedrooms vs Carpet Area":

        BedroomsCarpetArea()
        
    elif selectbox == "Carpet Area vs Price":

        CarpetAreaPrice()
        
    elif selectbox == "House Owners And House Buyer's strength":

        HouseOwners()

        NewOwners()
        
    elif selectbox == "Comaprision Between Current Features and New Features":
                
        OldFeatureNewFeature(compare_features)
                
elif(option == "Data Corelation"):
    
    corr_features = correlation(predictionDataset,0.85)
    st.write("## Highly Corelated Feature")
    corr_features
    predictionDataset.drop('c_Club house',axis=1)
    st.write("## HeatMap to show Correlation between Features")
    FinalDataCorr()
    st.write("## Droping the corelated feature")
    st.write(data.head())
        
elif(option == "Data Transformation"):
    st.write("## Dependent Variable (House Price)")
    st.write(dependent_var)
    st.write("## Scaling the Dependent Variable")
    st.write(y)
    st.write("## Prediction Dataset")
    st.write(predictionDataset.head())
    st.write("## Encoding the Dataset")
    st.write(x.head(15))
    st.write("## Splitting Data into Test and Trainning Set")
    X_train.shape , X_test.shape
    st.write("## Scaling Features")
    st.write(featuresScale)
    
elif(option == "Regression Model"):
    selectbox = st.sidebar.radio(label="", options=["Multiple Linear Regression","Support Vector Regression",
                "Decision Tree Regressor","Random Forest Regressor"])
    if(selectbox == "Multiple Linear Regression"):
               
        linear_y_pred = linear_regressor.predict(X_test)
        st.write("## Predicted Value")
        linear_y_pred
        
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(linear_y_pred).reshape(len(linear_y_pred),1)),axis=1))
         
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, linear_y_pred))
        st.write('R^2 Score:', metrics.r2_score(y_test,linear_y_pred))
        
        accuracies = cross_val_score(estimator = linear_regressor, X = x, y = y, cv = 10)
        st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100))       
                
    elif(selectbox == "Support Vector Regression"):

        st.write("## Scaling Features")
        st.write(featuresScale)
        
        st.write("## Predicted Value")
        y_pred_svr

        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_svr).reshape(len(y_pred_svr),1)),axis=1))
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_svr))
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_svr))
                
    elif(selectbox == "Decision Tree Regressor"):
       
        tree_regressor = DecisionTreeRegressor(random_state = 1)
        tree_regressor.fit(X_train, featuresScale)
        
        y_pred_tree = tree_regressor.predict(X_test)
        st.write("## Predicted Value")
        y_pred_tree
        
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_tree).reshape(len(y_pred_tree),1)),axis=1))
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_tree))
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_tree))
        
    elif(selectbox == "Random Forest Regressor"):
        
        st.write("## Predicted Value")
        y_pred_forest        
        np.set_printoptions(precision=2)
        st.write("## Actual Vs Predicted Value")
        st.write(np.concatenate((sc_y.inverse_transform(y_test).reshape(len(y_test),1),sc_y.inverse_transform(y_pred_forest).reshape(len(y_pred_forest),1)),axis=1))
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_forest))
        st.write('R^2 Score:', metrics.r2_score(y_test,y_pred_forest))
        
elif(option == "Classification Model"):
    radio = st.sidebar.radio(label = "", options = ["Random Forest Classifier","K Nearest Neighbors Classifier"])
    if(radio == "Random Forest Classifier"):        
        target_var = dataset1.iloc[:,0:4]
        target_var = target_var.drop('Location',axis=1)
        st.write("## Target Variable")
        st.write(target_var.head())
        
        le = LabelEncoder()
        target_var = target_var.apply(LabelEncoder().fit_transform)
        st.write("## Label Encoding on Target Variable")
        st.write(target_var.head())
        
        le = LabelEncoder()
        dataset1['House_Loan'] = le.fit_transform(dataset1['House_Loan'])
        st.write("## Dataset")        
        st.write(dataset1.head())
        
        y = dataset1.iloc[:,24].values
        st.write("## Label Encoding on House Loan")
        st.write(y)
        
        X_train, X_test, y_train, y_test = train_test_split(target_var,y,test_size=0.2)
        st.write("## Splitting dataset into Test and Train Data")
        st.write(X_train.shape, X_test.shape)
        
        model = RandomForestClassifier(n_estimators=20)
        model.fit(X_train, y_train)
        
        st.write("## Model Score")
        st.write(model.score(X_test, y_test))
        
        y_predicted = model.predict(X_test)
        cm = confusion_matrix(y_test, y_predicted)
        st.write("## Confusion Matrix")
        st.write(cm)
        
        st.write("## Accuracy Score")
        st.write(accuracy_score(y_test, y_predicted))
        
        st.write("## Classification Report")
        st.write(classification_report(y_test,y_predicted))
        
        st.write("## Heat Map to show Confusion Matirx")
        fig = plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        st.pyplot(fig)
        
    elif(radio == "K Nearest Neighbors Classifier"):
        target_var = dataset1.iloc[:,0:4]
        target_var = target_var.drop('Location',axis=1)
        st.write("## Target Variable")
        st.write(target_var.head())
        
        le = LabelEncoder()
        target_var = target_var.apply(LabelEncoder().fit_transform)
        st.write("## Label Encoding on Target Variable")
        st.write(target_var.head())
        
        le = LabelEncoder()
        dataset1['House_Loan'] = le.fit_transform(dataset1['House_Loan'])
        st.write("## Dataset")        
        st.write(dataset1.head())
        
        y = dataset1.iloc[:,24].values
        st.write("## Label Encoding on House Loan")
        st.write(y)
        
        X_train, X_test, y_train, y_test = train_test_split(target_var,y,test_size=0.2)
        st.write("## Splitting dataset into Test and Train Data")
        st.write(X_train.shape, X_test.shape)
        
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(X_train,y_train)
        st.write("## Model Score")
        st.write(knn.score(X_test, y_test))
        
        pred = knn.predict(X_test)
        st.write("## Accuracy Score")
        st.write(accuracy_score(y_test, pred))
        
        st.write("## Confusion Matrix")
        st.write(confusion_matrix(y_test,pred))
        
        st.write("## Classification Report")
        st.write(classification_report(y_test,pred))
    
        accuracy_rate = []
        for i in range(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            score=cross_val_score(knn,target_var,y,cv=10)
            accuracy_rate.append(score.mean())
        
        error_rate = []
        for i in range(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            score=cross_val_score(knn,target_var,y,cv=10)
            error_rate.append(1-score.mean())
        
        st.write("## Choosing the K value")
        st.write("K Value = 23")
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        st.pyplot(fig)
        
        st.write("## Checking the accuracy of K = 23")
        
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Accuracy vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Accuracy Rate')
        st.pyplot(fig)
   
elif (option == "Prediction Model"):
    st.write("## Location in Mumbai Western Line")
    with st.form(key='prediction_form'):
        location = st.selectbox(label="", options=["Marine Lines","Churchgate","Charni Road","Grant Road","Mumbai Central","Mahalaxmi","Lower Parel",
                                                     "Prabhadevi","Dadar","Matunga Road","Mahim Junction","Bandra","Khar Road","Santacruz","Vile Parle","Andheri","Jogeshwari","Ram Mandir",
                                                     "Goregaon","Malad","Borivali","Dahisar","Mira Road","Bhayandar","Naigaon","Vasai Road","Nallasopara","Virar"])
     
        st.write("## House Type")
        house_located = st.selectbox(label="", options=["Building","Bunglow","Chawl","Row House"])
        
        st.write("## Carpet Area")
        capet_area = st.text_input(label = "Enter Carpet Area in Square Fit")
        
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
        predicted_price = predict_price(location,house_located,capet_area,No_of_Bedrooms, c_Car_Parking,c_24hr_Water_Supply,
       c_Maintenance_Staff, c_Gas_Pipeline, c_Lift, c_Club_house,
       c_24Hr_Security, c_Gym, c_School, c_College, c_Hospitals,
       c_Medical, c_Railway_Station, c_Market)
        
        price_in_words = num2words(predicted_price, to='currency', lang='en_IN')
        
        st.write("The cost of the house will be ",predicted_price," that is "+price_in_words.title().replace("Euro", "Rupees").replace("Cents", "Paise"))
        st.write('The accurace of the model is ', round(metrics.r2_score(y_test,y_pred_forest)*100),"%")
        

    
    
    
     
    
        

        
        
                    
        
        







        




        
 
       
      
     

