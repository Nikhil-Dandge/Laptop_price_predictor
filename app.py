import streamlit as st
import pickle
import numpy as np

#model importing

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

#brand
company = st.selectbox('Brand',df['Company'].unique())

#type

type = st.selectbox('Type',df['TypeName'].unique())

#Ram

Ram = st.selectbox('RAM in GB',[2,4,6,8,12,16,24,32,64])

#weight

weight=st.number_input('Weight of the laptop')

#Touchscreen

Touchscreen =st.selectbox('TouchScreen',['No','Yes'])

#Ips

Ips= st.selectbox('IPS',['No','Yes'])

#Screensize

screen_size=st.number_input('Screensize')

#resolution


resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2340x1440'])

#cpu

Cpu =st.selectbox('CPU',df['Cpu brand'].unique())

#HDD

Hdd =st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

#SSD

Ssd =st.selectbox('SSD (in GB)',[0,8,128,256,512,1024,2048])

#GPU BRand

Gpu =st.selectbox('GPU',df['Gpu brand'].unique())

#OS

os =st.selectbox('os',df['os'].unique())

if st.button('Predict Price'):
    #query
    
    if Touchscreen=='Yes':
        Touchscreen=1
    else:
        Touchscreen=0
    
    if Ips=='Yes':
        Ips=1
    else:
        Ips=0
    
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size


    
    query = np.array([company,type,Ram,weight,Touchscreen,Ips,ppi,Cpu,Hdd,Ssd,Gpu,os])

    query = query.reshape(1,12)

    st.title("The Predicted price for this specifications is: " + str(int(np.exp(pipe.predict(query)[0]))))



