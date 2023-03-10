import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, auc, log_loss, precision_recall_fscore_support, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


st.title("Telecom Churn prediction")
st.sidebar.header('User Input Parameters')

def user_input_features():
    account_length =st.sidebar.slider('Account_Length', 1, 243)
    voice_mail_plan=st.sidebar.radio('Voice_Mail_Plan 📱', [0,1])
    voice_mail_messages=st.sidebar.slider('Voice_Mail_Messages 🗣️📳', 0, 51)
    day_mins=st.sidebar.slider('Day_Mins',0.00,350.80)
    evening_mins=st.sidebar.slider('Evening_Mins🌇',0.00,363.70)
    night_mins=st.sidebar.slider('Night_Mins🌃',23.20,395.00)
    international_mins=st.sidebar.slider('International_Mins 🌎',0.00, 20.00)
    customer_service_calls=st.sidebar.slider('Insert Customer_Service_Calls',0, 9)
    international_plan=st.sidebar.radio('International_Plan', [0, 1])
    day_calls=st.sidebar.slider('Day_Calls 📞',0, 165)
    day_charge=st.sidebar.slider('Day_Charge',0.00, 59.64)
    evening_calls=st.sidebar.slider('Number of Evening_Calls 🌇📞',0 ,170)
    evening_charge=st.sidebar.slider('Evening_Charge🌇 ',0.00, 30.91)
    night_calls=st.sidebar.slider('Number of Night_Calls 🌃📞',33, 175)
    night_charge=st.sidebar.slider('Night_Charges🌃',1.04, 17.77)
    international_calls=st.sidebar.slider('Number of International_Calls 🌍📞',0, 20)
    international_charge=st.sidebar.slider('International_Charges 🌏📞',0.00,5.40)
    
    data={'account_length':account_length,
         'voice_mail_plan':voice_mail_plan,
         'voice_mail_messages':voice_mail_messages,
         'day_mins':day_mins,
         'evening_mins':evening_mins,
         'night_mins':night_mins,
         'international_mins':international_mins,
         'customer_service_calls':customer_service_calls,
         'international_plan':international_plan,
         'day_calls':day_calls,
         'day_charge':day_charge,
         'evening_calls':evening_calls,
         'evening_charge':evening_charge,
         'night_calls':night_calls,
         'night_charge':night_charge,
         'international_calls':international_calls,
         'international_charge':international_charge}
    features =pd.DataFrame(data,index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


churn_data  =pd.read_csv("C:/Users/suhail/deployment_3pm/EDA_final.csv")
churn_data.drop(["Unnamed: 0"],axis=1,inplace=True)
churn_data 

churn_data.drop(["state","area.code"],axis=1,inplace=True)


scaler = MinMaxScaler()
scaled_churn_data = scaler.fit_transform(churn_data)
scaled_df = pd.DataFrame(scaled_churn_data,columns=churn_data.columns)

X = scaled_df.drop("Churn",axis=1)
y = scaled_df[["Churn"]]


svc_model = SVC(C = 50, gamma =50, kernel='rbf')
svc_model.fit(X , y)

prediction = svc_model.predict(df)
prediction_proba = svc_model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'Customer will not Churn')




























 