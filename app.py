
#Here is an example Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

#python
#Copy code
# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
data = pd.read_excel(xls , 'Used sheet' , header = 0)


# chnge some data into numeric


# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Select the features and target variable
features = ['L/B', 'Cb', 'Momen beban T','Displacement', ]
target = 'Inclinement'

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=300, random_state=50)
model.fit(train_data[features], train_data[target])

# Make predictions on the test set
test_predictions = model.predict(test_data[features])

# Evaluate the model performance
mse = mean_squared_error(test_data[target], test_predictions)
print('Mean squared error:', mse)

# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction Ver 0.031")

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
Breadth = st.number_input("Breadth Water Line (m)", min_value= 0.00, step =0.01)
Depth = st.number_input("Depth Water Line (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft Water Line (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
bebanA = st.number_input("Beban A (Kg)",min_value= 0.0000,  step =0.0001)
bebanB = st.number_input("Beban B (Kg)",min_value= 0.0000,  step =0.0001)
bebanC = st.number_input("Beban C (Kg)",min_value= 0.0000,  step =0.0001)
bebanD = st.number_input("Beban D (Kg)",min_value= 0.0000,  step =0.0001)
jumlah_beban = st.selectbox("Jumlah beban uji", beban)

##convert to anohter
beban_A = bebanA/1000
beban_B = bebanB/1000
beban_C = bebanC/1000
beban_D = bebanD/1000
if jumlah_beban == "4" :
        beban_E = 0
        beban_F = 0
        
        Mselisih1 = (beban_A + beban_B - beban_C - beban_D) * (-1) * ((Breadth) / 2)
        Mselisih2 =  (beban_B - beban_A - beban_C - beban_D)  * (-1) * ((Breadth) / 2)
        Mselisih3 =  (0 - beban_B - beban_A - beban_C - beban_D) * (-1) * ((Breadth) / 2)    
        Mselisih4 =  (beban_C - beban_B - beban_A -  beban_D) * (-1) * ((Breadth) / 2)
        Mselisih5 =  (beban_C + beban_D - beban_A -  beban_B) * (-1) * ((Breadth) / 2)   
        Mselisih6 =  (beban_A + beban_B + beban_C -  beban_D) * (-1) * ((Breadth) / 2)      
        Mselisih7 =  (beban_C + beban_D + beban_A +  beban_B) * (-1) * ((Breadth) / 2)  
        Mselisih8 =  (beban_A + beban_B + beban_D -  beban_C) * (-1) * ((Breadth) / 2)  
    
if jumlah_beban == "6" :
        bebanE = st.number_input("Beban E (Kg)",min_value= 0.00, step =0.01)
        bebanF = st.number_input("Beban F (Kg)",min_value= 0.00, step =0.01)

        beban_E = bebanE/1000
        beban_F = bebanF/1000

        Mselisih1 = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih2 = (0 -beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih3 = (0 - beban_A - beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih4 = (0 - beban_A - beban_B - beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih5 = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih6 = (beban_A + beban_B + beban_C + beban_D + beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih7 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih8 = (beban_A + beban_B + beban_C + beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
            
#Calculation     

ok = st.button("Calculate Incline")
if ok:
        displacement = (Lwl * Breadth * Draft * Cb)
                
        if Breadth == 0:
                LB = 0
        else :
                LB = (Lwl /Breadth) 
                
        new_test1 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih1],'L/B': [LB]})
        predicted_Incline1 = model.predict(new_test1)
        
        new_test2 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih2],'L/B': [LB]})
        predicted_Incline2 = model.predict(new_test2)
        
        new_test3 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih3],'L/B': [LB]})
        predicted_Incline3 = model.predict(new_test3)
        
        new_test4 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih4],'L/B': [LB]})
        predicted_Incline4 = model.predict(new_test4)
        
        new_test5 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih5],'L/B': [LB]})
        predicted_Incline5 = model.predict(new_test5)
        
        new_test6 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih6],'L/B': [LB]})
        predicted_Incline6 = model.predict(new_test6)
        
        new_test7 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih7],'L/B': [LB]})
        predicted_Incline7 = model.predict(new_test7)
        
        new_test8 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'Momen Beban T': [Mselisih8],'L/B': [LB]})
        predicted_Incline8 = model.predict(new_test8)
        
        dataS = pd.DataFrame({
                'No': ['1','2','3','4','5','6','7','8'],
                'Moment': [Mselisih1, Mselisih2, Mselisih3, Mselisih4, Mselisih5, Mselisih6, Mselisih7, Mselisih8],
                'incline': [predicted_Incline1, predicted_Incline2, predicted_Incline3, predicted_Incline4, predicted_Incline5, predicted_Incline6, predicted_Incline7, predicted_Incline8],
                        }
                )
        st.table(dataS)
        st.subheader(f"the accuracy of this inclinement model is {mse} " )
        # make graphics
        st.plotly_chart({
            'data': [{
                'x': dataS['Moment'],
                'y': dataS['incline'],
                'mode': 'markers',
                'marker': {'opacity': 0.5},
            }],
            'layout': {
                'title': 'Inclining moment',
                'xaxis': {'title': 'Moment'},
                'yaxis': {'title': 'Inclining Degree'},
            }})
