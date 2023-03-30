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
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
le_kapal = LabelEncoder()
data['Jenis Kapal'] = le_kapal.fit_transform(data['Jenis Kapal'])
data["Jenis Kapal"].unique()
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
proses = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
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

    
new_test = pd.DataFrame({'Jenis Kapal': [1], 'Displacement': [150], 'Selisih beban': [1],})
predicted_Incline = model.predict(new_test)
print('Predicted GM:', predicted_Incline)
