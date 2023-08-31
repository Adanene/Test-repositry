#Here is an Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

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

# Define a dictionary to store the session state values
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False
    
# Load the dataset
@st.cache(allow_output_mutation=True)

        
def fetch_data():
        sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
        xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
        data = pd.read_excel(xls , 'Used sheet' , header = 0)
        return data

data = fetch_data()



# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction Ver 0.86")

st.write("""### We need some data to predict ship inclining angle""")

#input the new data here
#input some choose answer
beban = (
        "0",
        "4",
        "6",
        )
# input some wrrited answer


Loa = st.number_input("Length Over All (m)", value= 0.00, step=0.01)
Lwl = st.number_input("Length Water Line (m)",min_value= 0.00, max_value= Loa)
Breadth = st.number_input("Breadth Water Line (m)", min_value= 0.00, step =0.01)
Depth = st.number_input("Depth  (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
jumlah_beban = st.selectbox("Number Weight", beban)

##convert to anohter

if jumlah_beban == "4" :
        bebanA = st.number_input("Beban 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Beban 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Beban 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Beban 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = 0
        bebanF = 0

if jumlah_beban == "6" :
        bebanA = st.number_input("Beban 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Beban 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Beban 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Beban 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = st.number_input("Beban 5 (Kg)",min_value= 0.0000, step =0.0001)
        bebanF = st.number_input("Beban 6 (Kg)",min_value= 0.0000, step =0.0001)
        
        

ok = st.button("Calculate Incline")       
if ok:
    st.session_state.button_pressed = True

    #start machine learning process
    # chnge some data into numeric

    # Split the dataset into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=90)

    # Select the features and target variable
    features = ['B/T', 'Cb', 'D/T', 'kirkan', 'beban/disp',]
    target = 'Inclinement'
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 250, 500, 750],  # Adjust as needed
        'max_depth': [None, 10, 25, 50, 75],       # Adjust as needed
        'min_samples_split': [ 2, 3, 4, 5],      # Adjust as needed
        'min_samples_leaf': [ 2, 3, 4]        # Adjust as needed
    }

    # Create the RandomForestRegressor
    rf = RandomForestRegressor(random_state=600)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit the GridSearchCV to the training data
    grid_search.fit(train_data[features], train_data[target])

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    
    # Extract feature importances
    importances = best_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]


    # Make predictions on the test set
    test_predictions = best_model.predict(test_data[features])

    # Evaluate the model performance
    mse = mean_squared_error(test_data[target], test_predictions)
    print('Mean squared error:', mse) 

    # Make predictions on the test set
    test_predictions = best_model.predict(test_data[features])

    # Evaluate the model performance
    mse = mean_squared_error(test_data[target], test_predictions)
    print('Mean squared error:', mse) 

if st.session_state.button_pressed:
        if jumlah_beban =="0" :
                st.subheader(f"the accuracy of this inclinement model is {mse}  " )
        
        else:
        
                #transfer weight
                beban_A = bebanA
                beban_B = bebanB
                beban_C = bebanC
                beban_D = bebanD
                beban_E = bebanE
                beban_F = bebanF
                totalB = (bebanA + bebanB + bebanC + bebanD + bebanE + bebanF)
        
                #calculate displacement
                displacement = (Lwl * Breadth * Draft * Cb)

                totdisp = (totalB / displacement)
                

                #calculate moment
                if jumlah_beban == "4" :

                        kiri1 = (beban_A + beban_B)
                        kiri2 = (beban_B)
                        kiri3 = 0
                        kiri4 = (beban_C)
                        kiri5 = (beban_C + beban_D)
                        kiri6 = (beban_A + beban_B + beban_C)
                        kiri7 = (beban_A + beban_B + beban_C + beban_D)
                        kiri8 = (beban_A + beban_B + beban_D)

                        kanan1 = (beban_C + beban_D)
                        kanan2 = (beban_A + beban_C + beban_D)
                        kanan3 = (beban_A + beban_B + beban_C + beban_D)
                        kanan4 = (beban_A + beban_B + beban_D)
                        kanan5 = (beban_A + beban_B)
                        kanan6 = (beban_D)
                        kanan7 = 0
                        kanan8 = (beban_C)
                        
                        
                
                if jumlah_beban == "6" :

                        kiri1 = (beban_A + beban_C + beban_E)
                        kiri2 = (beban_A + beban_B + beban_C + beban_E)
                        kiri3 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kiri4 = (beban_A + beban_B + beban_C + beban_D + beban_E )
                        kiri5 = (beban_A + beban_C + beban_E)
                        kiri6 = (beban_E)
                        kiri7 = 0
                        kiri8 = (beban_C + beban_E)

                        kanan1 = (beban_B + beban_D + beban_F)
                        kanan2 = (beban_D + beban_F)
                        kanan3 = 0
                        kanan4 = (beban_F)
                        kanan5 = (beban_B + beban_D + beban_F)
                        kanan6 = (beban_A + beban_B + beban_C + beban_D + beban_F )
                        kanan7 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kanan8 = (beban_A + beban_B + beban_D + beban_F)
                    
                     
                #finding ratio
                if Breadth == 0:
                        LB = 0
                else :
                        LB = (Lwl /Breadth) 

                if Draft == 0:
                        BT = 0
                        DT = 0
                else :
                        BT = (Breadth /Draft) 
                        DT = (Depth / Draft)
                    
                if kiri1 < kanan1:
                    kikan1 = ((kiri1 / kanan1))
                else :
                    kikan1 = (-(kanan1 / kiri1)) 
                    
                if kiri2 < kanan2:
                    kikan2 = ((kiri2 / kanan2))
                else :
                    kikan2 = (-(kanan2 / kiri2)) 
                    
                if kiri3 < kanan3:
                    kikan3 = ((kiri3 / kanan3))
                else :
                    kikan3 = (-(kanan3 / kiri3))
                    
                if kiri4 < kanan4:
                    kikan4 = ((kiri4 / kanan4))
                else :
                    kikan4 = (-(kanan4 / kiri4))
                    
                if kiri5 < kanan5:
                    kikan5 = ((kiri5 / kanan5))
                else :
                    kikan5 = (-(kanan5 / kiri5))
                    
                if kiri6 < kanan6:
                    kikan6 = ((kiri6 / kanan6))
                else :
                    kikan6 = (-(kanan6 / kiri6))
                    
                if kiri7 < kanan7:
                    kikan7 = ((kiri7 / kanan7))
                else :
                    kikan7 = (-(kanan7 / kiri7)) 

                if kiri8 < kanan8:
                    kikan8 = ((kiri8 / kanan8))
                else :
                    kikan8 = (-(kanan8 / kiri8))
                    
                    
                Mselisih1 =  (kiri1 - kanan1) * (-1) * ((Breadth) / 2)
                Mselisih2 =  (kiri2 - kanan2)  * (-1) * ((Breadth) / 2)
                Mselisih3 =  (kiri3 - kanan3) * (-1) * ((Breadth) / 2)    
                Mselisih4 =  (kiri4 - kanan4) * (-1) * ((Breadth) / 2)
                Mselisih5 =  (kiri5 - kanan5) * (-1) * ((Breadth) / 2)   
                Mselisih6 =  (kiri6 - kanan6) * (-1) * ((Breadth) / 2)      
                Mselisih7 =  (kiri7 - kanan7) * (-1) * ((Breadth) / 2)  
                Mselisih8 =  (kiri8 - kanan8) * (-1) * ((Breadth) / 2)        

                
                new_test1 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan1], 'beban/disp' : [totdisp], })
                predicted_Incline1 = model.predict(new_test1)
        
                new_test2 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan2], 'beban/disp' : [totdisp], })
                predicted_Incline2 = model.predict(new_test2)
        
                new_test3 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan3], 'beban/disp' : [totdisp], })
                predicted_Incline3 = model.predict(new_test3)
        
                new_test4 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan4], 'beban/disp' : [totdisp], })
                predicted_Incline4 = model.predict(new_test4)
        
                new_test5 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan5], 'beban/disp' : [totdisp], })
                predicted_Incline5 = model.predict(new_test5)
        
                new_test6 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan6], 'beban/disp' : [totdisp], })
                predicted_Incline6 = model.predict(new_test6)
        
                new_test7 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan7], 'beban/disp' : [totdisp], })
                predicted_Incline7 = model.predict(new_test7)
        
                new_test8 = pd.DataFrame({ 'B/T' :[BT], 'Cb': [Cb], 'D/T' :[DT] , 'kirkan': [kikan8], 'beban/disp' : [totdisp], })
                predicted_Incline8 = model.predict(new_test8)
        
                dataS = pd.DataFrame({
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
                        ax.annotate(i, (dataS['Moment Beban'].iloc[i], dataS['incline'].iloc[i])) # i+1 because Python's indexing starts at 0

                ax.set_xlabel('Moment Beban')
                ax.set_ylabel('incline')

                # Customization: draw a vertical line (you can adjust this as per your requirement)
                threshold = dataS['Moment Beban'].mean()  # example threshold using mean, adjust as needed
                ax.axvline(x=0, color='red', linestyle='--', label=" 0,0 coordinate")
                ax.legend()

                # Customization: draw a horizontal  line (you can adjust this as per your requirement)
                thresholds = dataS['incline'].mean()  # example threshold using mean, adjust as needed
                ax.axhline(y=0, color='red', linestyle='--', label="")
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)
            
                # Plotting feature importances
                imp, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(importances)), importances[sorted_indices], align='center')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(np.array(features)[sorted_indices])
                ax.set_title("Feature Importances")
                ax.set_ylabel('Importance')
                ax.set_xlabel('Features')

                st.pyplot(imp)  # Pass the figure object to st.pyplot()
