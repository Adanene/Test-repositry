2 / 2

#Here is an example Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

#python
#Copy code
# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs/edit#gid=1732295199")
data = pd.read_excel(xls , 'Used sheet' , header = 0)


# chnge some data into numeric


# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Select the features and target variable
features = ['L/B', 'Cb', 'Momen beban T','Displacement', 'Selisih beban', ]
target = 'Inclinement'

# Train a linear regression model
model = LinearRegression()
model.fit(train_data[features], train_data[target])

# Make predictions on the test set
test_predictions = model.predict(test_data[features])

# Evaluate the model performance
mse = mean_squared_error(test_data[target], test_predictions)
print('Mean squared error:', mse)

# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction")

st.write("""### We need some data to predict ship inclining angle""")

#input the new data here
#input some choose answer
beban = (
        "4",
        "6",
        )
# input some wrrited answer

Loa = st.number_input("Length Over All (m)", min_value= 0.00, step =0.01)
Lwl = st.number_input("Length Water Line (m)",min_value= 0.00, max_value= Loa)
Breadth = st.number_input("Breadth (m)", min_value= 0.00, step =0.01)
Depth = st.number_input("Depth (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
beban_A = st.number_input("Beban A (Ton)",min_value= 0.00,  step =0.01)
beban_B = st.number_input("Beban B (Ton)",min_value= 0.00,  step =0.01)
beban_C = st.number_input("Beban C (Ton)",min_value= 0.00,  step =0.01)
beban_D = st.number_input("Beban D (Ton)",min_value= 0.00,  step =0.01)
jumlah_beban = st.selectbox("Jumlah beban uji", beban)

if jumlah_beban == "4" :
        beban_E = 0
        beban_F = 0
        proses = st.slider("Proses incline", 0, 7, 0)
        if proses == 0:
            st.write("""### Beban di kiri : AB""") 
            st.write("""### Beban di kanan  : CD """)
            Mselisih = (beban_A + beban_B - beban_C - beban_D) * (-1) * ((Breadth) / 2)
        if proses == 1:
            st.write("""### Beban di kiri : B""") 
            st.write("""### Beban di kanan  : ACD """)
            Mselisih =  (beban_B - beban_A - beban_C - beban_D)  * (-1) * ((Breadth) / 2)
        if proses == 2:
            st.write("""### Beban di kiri : None """) 
            st.write("""### Beban di kanan  : ABCD """)
            Mselisih =  (0 - beban_B - beban_A - beban_C - beban_D) * (-1) * ((Breadth) / 2)    
        if proses == 3:
            st.write("""### Beban di kiri : C""") 
            st.write("""### Beban di kanan  : ABD """)
            Mselisih =  (beban_C - beban_B - beban_A -  beban_D) * (-1) * ((Breadth) / 2)
        if proses == 4:
            st.write("""### Beban di kiri : CD""") 
            st.write("""### Beban di kanan  : AB """)
            Mselisih =  (beban_C + beban_D - beban_A -  beban_B) * (-1) * ((Breadth) / 2)   
        if proses == 5:
            st.write("""### Beban di kiri : ABC""") 
            st.write("""### Beban di kanan  : D """)
            Mselisih =  (beban_A + beban_B + beban_C -  beban_D) * (-1) * ((Breadth) / 2)      
        if proses == 6:
            st.write("""### Beban di kiri : ABCD""") 
            st.write("""### Beban di kanan  : None """)
            Mselisih =  (beban_C + beban_D - beban_A +  beban_B) * (-1) * ((Breadth) / 2)  
        if proses == 7:
            st.write("""### Beban di kiri : ABD """) 
            st.write("""### Beban di kanan  : C """)
            Mselisih =  (beban_A + beban_B + beban_D -  beban_C) * (-1) * ((Breadth) / 2)  
    
if jumlah_beban == "6" :
        beban_E = st.number_input("Beban E (Ton)",min_value= 0.00, step =0.01)
        beban_F = st.number_input("Beban F (Ton)",min_value= 0.00, step =0.01)
        proses = st.slider("Proses incline", 0, 7, 0)
        if proses == 0:
            st.write("""### Beban di kiri : ABC""")
            st.write("""### Beban di kanan : DEF""") 
            Mselisih = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        if proses == 1:
            st.write("""### Beban di kiri : BC""") 
            st.write("""### Beban di kanan : ADEF""")
            Mselisih = (0 -beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        if proses == 2:
            st.write("""### Beban di kiri : C""") 
            st.write("""### Beban di kanan : ABDEF""")
            Mselisih = (0 - beban_A - beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        if proses == 3:
            st.write("""### Beban di kiri : None""") 
            st.write("""### Beban di kanan : ABCDEF""")
            Mselisih = (0 - beban_A - beban_B - beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        if proses == 4:
            st.write("""### Beban di kiri : ABC""") 
            st.write("""### Beban di kanan : DEF""")
            Mselisih = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        if proses == 5:
            st.write("""### Beban di kiri : ABCDE""") 
            st.write("""### Beban di kanan : F""")
            Mselisih = (beban_A + beban_B + beban_C + beban_D + beban_E - beban_F) * (-1) * ((Breadth) / 2)
        if proses == 6:
            st.write("""### Beban di kiri : ABCDEF""") 
            st.write("""### Beban di kanan : None""")
            Mselisih = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F) * (-1) * ((Breadth) / 2)    
        if proses == 7:
            st.write("""### Beban di kiri : ABCD""") 
            st.write("""### Beban di kanan : EF""")
            Mselisih = (beban_A + beban_B + beban_C + beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
#Calculation     

ok = st.button("Calculate Incline")
if ok:
        displacement = (Lwl * Breadth * Draft * Cb)
        LB = (Lwl /Breadth)
           
        new_test = pd.DataFrame({'Jenis Kapal': [Ship],'Cb': [Cb], 'Displacement': [displacement], 'Selisih beban': [Mselisih],'L/B': [LB]})
        predicted_Incline = model.predict(new_test)
        st.subheader(f" Ship will incline in {predicted_Incline} degrees")
        print('Inclining Prediction:', predicted_Incline)
   

