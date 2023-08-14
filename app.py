
#Here is an example Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

#python
#Copy code
# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache(allow_output_mutation=True)
def fetch_data():
        sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
        xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
        data = pd.read_excel(xls , 'Used sheet' , header = 0)
        return data

data = fetch_data()

# chnge some data into numeric

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=90)

# Select the features and target variable
features = ['L/B', 'Cb', 'MB','Displacement', ]
target = 'Inclinement'

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],  # Adjust as needed
    'max_depth': [None, 10, 20],       # Adjust as needed
    'min_samples_split': [2, 5],      # Adjust as needed
    'min_samples_leaf': [1, 2]        # Adjust as needed
}

# Create the RandomForestRegressor
rf = RandomForestRegressor(random_state=100)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the GridSearchCV to the training data
grid_search.fit(train_data[features], train_data[target])

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on the test set
test_predictions = best_model.predict(test_data[features])

# Evaluate the model performance
mse = mean_squared_error(test_data[target], test_predictions)
print('Mean squared error:', mse) 

# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction Ver 0.035")

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
Depth = st.number_input("Depth  (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
bebanA = st.number_input("Beban 1 (Kg)",min_value= 0.0000,  step =0.0001)
bebanB = st.number_input("Beban 2 (Kg)",min_value= 0.0000,  step =0.0001)
bebanC = st.number_input("Beban 3 (Kg)",min_value= 0.0000,  step =0.0001)
bebanD = st.number_input("Beban 4 (Kg)",min_value= 0.0000,  step =0.0001)
jumlah_beban = st.selectbox("Jumlah beban uji", beban)

##convert to anohter
beban_A = bebanA
beban_B = bebanB
beban_C = bebanC
beban_D = bebanD
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
        bebanE = st.number_input("Beban 5 (Kg)",min_value= 0.00, step =0.01)
        bebanF = st.number_input("Beban 6 (Kg)",min_value= 0.00, step =0.01)

        beban_E = bebanE
        beban_F = bebanF

        Mselisih1 = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih2 = (0 -beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih3 = (0 - beban_A - beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih4 = (0 - beban_A - beban_B - beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih5 = (beban_A + beban_B + beban_C - beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih6 = (beban_A + beban_B + beban_C + beban_D + beban_E - beban_F) * (-1) * ((Breadth) / 2)
        Mselisih7 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F) * (-1) * ((Breadth) / 2)    
        Mselisih8 = (beban_A + beban_B + beban_C + beban_D - beban_E - beban_F) * (-1) * ((Breadth) / 2)
            
#Calculation     

ok = st.button("Calculate Incline ")
if ok:
        displacement = (Lwl * Breadth * Draft * Cb)
                
        if Breadth == 0:
                LB = 0
        else :
                LB = (Lwl /Breadth) 
                
        new_test1 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih1],'L/B': [LB]})
        predicted_Incline1 = best_model.predict(new_test1)
        
        new_test2 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih2],'L/B': [LB]})
        predicted_Incline2 = best_model.predict(new_test2)
        
        new_test3 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih3],'L/B': [LB]})
        predicted_Incline3 = best_model.predict(new_test3)
        
        new_test4 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih4],'L/B': [LB]})
        predicted_Incline4 = best_model.predict(new_test4)
        
        new_test5 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih5],'L/B': [LB]})
        predicted_Incline5 = best_model.predict(new_test5)
        
        new_test6 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih6],'L/B': [LB]})
        predicted_Incline6 = best_model.predict(new_test6)
        
        new_test7 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih7],'L/B': [LB]})
        predicted_Incline7 = best_model.predict(new_test7)
        
        new_test8 = pd.DataFrame({'Cb': [Cb], 'Displacement': [displacement], 'MB': [Mselisih8],'L/B': [LB]})
        predicted_Incline8 = best_model.predict(new_test8)
        
        dataS = pd.DataFrame({
                'No': ['1','2','3','4','5','6','7','8'],
                'Moment Beban': [Mselisih1, Mselisih2, Mselisih3, Mselisih4, Mselisih5, Mselisih6, Mselisih7, Mselisih8],
                'incline': [predicted_Incline1[0], predicted_Incline2[0], predicted_Incline3[0], predicted_Incline4[0], predicted_Incline5[0], predicted_Incline6[0], predicted_Incline7[0], predicted_Incline8[0]],
                        }
                )
        st.table(dataS)
        st.subheader(f"the accuracy of this inclinement model is {mse} " )
        # make graphics
        fig, ax = plt.subplots()

        # Create a scatter plot
        scatter = ax.scatter(dataS['Moment Beban'], dataS['incline'], color='blue', label='Incliing result')
        
        # Set title, labels, and legend
        ax.set_title("Inclining graphic")
        ax.set_xlabel('Moment Beban')
        ax.set_ylabel('incline')
        ax.legend()

        # Add annotations
        for i in range(len(dataS)):
                ax.annotate(i+1, (dataS['Moment Beban'].iloc[i], dataS['incline'].iloc[i])) # i+1 because Python's indexing starts at 0

        ax.set_xlabel('Moment Beban')
        ax.set_ylabel('incline')

        # Customization: draw a vertical line (you can adjust this as per your requirement)
        threshold = dataS['Moment Beban'].mean()  # example threshold using mean, adjust as needed
        ax.axvline(x=threshold, color='red', linestyle='--', label=" 0,0 coordinate")
        ax.legend()

        # Customization: draw a horizontal  line (you can adjust this as per your requirement)
        thresholds = dataS['incline'].mean()  # example threshold using mean, adjust as needed
        ax.axhline(y=thresholds, color='red', linestyle='--', label="")
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)
