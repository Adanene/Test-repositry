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
sheet_id ='1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y/export?format-xlsx")
data = pd.read_excel(xls , 'Sheet2' , header = 0)


# chnge some data into numeric





# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Select the features and target variable
features = ['Jenis Kapal', 'Displacement', 'Selisih beban',]
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
Kapal = (
        "Kapal penumpang",
        "Kapal patroli",
        "Kapal kargo",
        "Kapal multipurpose",
        )
beban = (
        "4",
        "6",
        )


Kapal = st.selectbox("Jenis Kapal", Kapal)
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
            selisih = (beban_A + beban_B - beban_C - beban_D) * (-1)
        if proses == 1:
            st.write("""### Beban di kiri : B""") 
            st.write("""### Beban di kanan  : ACD """)
            selisih =  (beban_B - beban_A - beban_C - beban_D)  * (-1)
        if proses == 2:
            st.write("""### Beban di kiri : None """) 
            st.write("""### Beban di kanan  : ABCD """)
            selisih =  (0 - beban_B - beban_A - beban_C - beban_D) * (-1)    
        if proses == 3:
            st.write("""### Beban di kiri : C""") 
            st.write("""### Beban di kanan  : ABD """)
            selisih =  (beban_C - beban_B - beban_A -  beban_D) * (-1)
        if proses == 4:
            st.write("""### Beban di kiri : CD""") 
            st.write("""### Beban di kanan  : AB """)
            selisih =  (beban_C + beban_D - beban_A -  beban_B) * (-1)   
        if proses == 5:
            st.write("""### Beban di kiri : ABC""") 
            st.write("""### Beban di kanan  : D """)
            selisih =  (beban_A + beban_B + beban_C -  beban_D) * (-1)      
        if proses == 6:
            st.write("""### Beban di kiri : ABCD""") 
            st.write("""### Beban di kanan  : None """)
            selisih =  (beban_C + beban_D - beban_A +  beban_B) * (-1)  
        if proses == 7:
            st.write("""### Beban di kiri : ABD """) 
            st.write("""### Beban di kanan  : C """)
            selisih =  (beban_A + beban_B + beban_D -  beban_C) * (-1)  
    
if jumlah_beban == "6" :
        beban_E = st.number_input("Beban E (Ton)",min_value= 0.00, step =0.01)
        beban_F = st.number_input("Beban F (Ton)",min_value= 0.00, step =0.01)
        proses = st.slider("Proses incline", 0, 7, 0)
        if proses == 0:
            st.write("""### Beban di kiri : ABC""")
            st.write("""### Beban di kanan : DEF""") 
            selisih = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1)
        if proses == 1:
            st.write("""### Beban di kiri : BC""") 
            st.write("""### Beban di kanan : ADEF""")
            selisih = (0 -beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1)    
        if proses == 2:
            st.write("""### Beban di kiri : C""") 
            st.write("""### Beban di kanan : ABDEF""")
            selisih = (0 - beban_A - beban_B + beban_C - beban_D - beban_E - beban_F) * (-1)
        if proses == 3:
            st.write("""### Beban di kiri : None""") 
            st.write("""### Beban di kanan : ABCDEF""")
            selisih = (0 - beban_A - beban_B - beban_C - beban_D - beban_E - beban_F) * (-1)    
        if proses == 4:
            st.write("""### Beban di kiri : ABC""") 
            st.write("""### Beban di kanan : DEF""")
            selisih = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1)
        if proses == 5:
            st.write("""### Beban di kiri : ABCDE""") 
            st.write("""### Beban di kanan : F""")
            selisih = (beban_A + beban_B + beban_C + beban_D + beban_E - beban_F) * (-1)
        if proses == 6:
            st.write("""### Beban di kiri : ABCDEF""") 
            st.write("""### Beban di kanan : None""")
            selisih = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F) * (-1)    
        if proses == 7:
            st.write("""### Beban di kiri : ABCD""") 
            st.write("""### Beban di kanan : EF""")
            selisih = (beban_A + beban_B + beban_C + beban_D - beban_E - beban_F) * (-1)

#Calculation        
new_test = pd.DataFrame({'Jenis Kapal': [1], 'Displacement': [150], 'Selisih beban': [1],})
predicted_Incline = model.predict(new_test)
print('Predicted GM:', predicted_Incline)
