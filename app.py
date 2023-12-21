#Jupyter code for machine learning to predict ship stability during an inclining test using Python and XG-BOOST library:

#python
# Import necessary libraries
import pandas as pd
import csv
import os
import json
import joblib
import pickle
import requests
import math
import base64
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import BytesIO

# Define a dictionary to store the session state values
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False
    
# Load the dataset
@st.cache(allow_output_mutation=True)

 
def fetch_data():
        sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
        xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
        data = pd.read_excel(xls , 'Usesheet' , header = 0)
        return data

data = fetch_data()
download = 0
# Specify the shareable link of your JSON file
# Specify the URL of your model on GitHub
# URL of the raw file on GitHub
github_raw_url = 'https://raw.githubusercontent.com/Adanene/Test-repositry/main/your_model.pkl'
file_path = 'your_model.pkl'



# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction Ver 1.6 (XGB)")
st.write("""### How to use: """)
st.write("""##### 1. this only applicable to monohull""")
st.write("""##### 2. only 4 and 6 weight method used on this inclining test  """)
st.write("""##### 3. each weight must have simmiliar (or close enough) weight """)
st.write("""##### 4. inclining weight must placed symetrical and no further than ship's half breadth """)
st.write("""##### 5. the result of this app is inclining angle during 4 or 6 weight method process """)
st.write("""##### 6. To prevent error, dont put "0" on ship dimension unless "Number Weight" also "0"  """)
st.write("""##### 7. Inclining test is based on BKI rules "Guidance for inclining test 2015" """)
st.markdown("For more information about BKI rules, [you can click on here](https://rules-api.bki.co.id/v1/publication?path=cHVibGljYXRpb25zL0d1aWRhbmNlL1BhcnQgNi4gU3RhdHV0b3J5LyggVm9sIEMgKSBQZXR1bmp1ayBQZW5ndWppYW4gS2VtaXJpbmdhbiBkYW4gUGVyaW9kZSBPbGVuZyBLYXBhbC8yMDE1Lzc3X1BldHVuanVrIFBlbmd1amlhbiBLZW1pcmluZ2FuIGRhbiBQZXJpb2RlIE9sZW5nIEthcGFsXzIwMTUtNl8zLnBkZg,,&act=view&app=252f31d48ff053e3a7bba35251ad78d1).")
st.write("""### We need some data to predict ship inclining angle""")

#input the new data here
#input some choose answer
beban = (
        "0",
        "4",
        "6",
        )
# input some wrrited answer

Lwl = st.number_input("Length Water Line (m)",min_value= 0.00, step =0.01)
Breadth = st.number_input("Breadth Water Line (m)", min_value= 0.00, step =0.01)
Depth = st.number_input("Depth  (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
jumlah_beban = st.selectbox("Number Weight", beban)

##convert to anohter

if jumlah_beban == "4" :
        st.image('https://drive.google.com/uc?id=14WooQDOBkHLHqVkCy6wFwLfzb6Jxq8YS')
        st.write("""##### please make sure if every weight at one side, at least it inclined 1 degree""")
        bebanA = st.number_input("Weight 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Weight 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Weight 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Weight 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = 0
        bebanF = 0
        

if jumlah_beban == "6" :
        st.image('https://drive.google.com/uc?id=1BqM-jtRUqNR5w9NNU2teF4R5qYJ2GI7D')
        st.write("""##### please make sure if every weight at one side, at least it inclined 1 degree""")
        bebanA = st.number_input("Weight 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Weight 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Weight 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Weight 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = st.number_input("Weight 5 (Kg)",min_value= 0.0000, step =0.0001)
        bebanF = st.number_input("Weight 6 (Kg)",min_value= 0.0000, step =0.0001)
        
ok = st.button("Calculate Incline")       


if ok:
    # URL of the raw file on GitHub
    github_raw_url = 'https://github.com/Adanene/Test-repositry/tree/main/your_model.pkl'
    #Load the model if exist
    if os.path.exists(file_path):
        loaded_model = joblib.load(file_path)
    else:
        print(f"The file {file_path} does not exist.")
        # Download the file
        response = requests.get(github_raw_url)

    # Check if the download was successful
    if response.status_code == 200:
    # Check if the file exists on GitHub
        response = requests.head(github_raw_url)
        file_exists = response.status_code == 200
        # Now you can use the loaded_model as needed
        st.success("Model loaded successfully!")
    else:
        print(f"Failed to download the model. Status code: {response.status_code}")    


    #start machine learning process
    def calculate_mape(actual, predicted):
            errors = np.abs(actual - predicted)
            denominator = np.abs(actual)
    
            # Handle cases where denominator is zero
            denominator[denominator == 0] = 0.01  # Convert zeros to other value to avoid division by zero

            # Calculate MAPE
            mape = np.nanmean(errors / denominator) * 100

            # Convert mape to a string
            mape_str = f"{mape:.2f}"

            return mape_str

    # Saving the model
    def save_model(model, file_path='your_model.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

    # Loading the model
    def load_model(file_path='your_model.pkl'):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

    # Training or loading the model
    def train_or_load_model(X, y):
        github_raw_url = 'https://raw.githubusercontent.com/Adanene/Test-repositry/main/your_model.pkl'

        if download == 1:
            # Download the model from GitHub
            response = requests.get(github_raw_url)
            response.raise_for_status()
    
            # Save the downloaded content to a local file
            with open('your_model.pkl', 'wb') as f:
                f.write(response.content)

            # Load the model using pickle
            loaded_model = load_model('your_model.pkl')
        else:
            # Train the model (your existing training code)
            features = ['beban/disp', 'Cb', 'cogm', 'B/T']
            target = 'Inclinement'
            X = data[features]
            y = data[target]
            xgboost_model = xgb.XGBRegressor(random_state=400, objective="reg:squarederror")
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10],
                'learning_rate': [1.25],
                'subsample': [1],
                'colsample_bytree': [1.0],
                'reg_alpha': [1],
                'reg_lambda': [1],
                'gamma': [0],
                'min_child_weight': [6],
                'scale_pos_weight': [1]
            }
            grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid,
                                       cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', error_score='raise')
            grid_search.fit(X, y, eval_metric='rmse', eval_set=[(X, y)], early_stopping_rounds=100)
            loaded_model = grid_search.best_estimator_

            # Save the trained model to a local file using pickle
            save_model(loaded_model, 'your_model.pkl')

        return loaded_model

    
    st.session_state.button_pressed = True                 
    # Train the model (your existing training code)

    #Select the features and target variable
    features = ['beban/disp', 'Cb' , 'cogm', 'B/T',]
    target = 'Inclinement'

    # Split the dataset into features (X) and target variable (y)
    X = data[features]
    y = data[target]

    # Train or load the model
    best_model = train_or_load_model(X, y)

    # Make predictions on all data points using the model
    print(f"Type of best_model: {type(best_model)}")
    print(f"Shape of X: {X.shape}")
    all_predictions_best_model = best_model.predict(X)

    
    # Now, all_predictions_best_model contain the predictions for all data points in your dataset

    # Apply the threshold to predicted values
    threshold = 0.001  # You can adjust this value based on your domain knowledge
    all_predictions_best_model[all_predictions_best_model < threshold] = 0.01
   
    # MAPE Prediction for best_model
    mape_best_model = calculate_mape(y, all_predictions_best_model)

    # Evaluate the MSE performance for best_model
    mse_best_model = mean_squared_error(y, all_predictions_best_model)
    print('Mean squared error for best_model:', mse_best_model)

    # Note: XGBoost also provides feature importances similar to Random Forest
    importances_best_model = best_model.feature_importances_
    sorted_indices_best_model = np.argsort(importances_best_model)[::-1]
    
   
    ## this is for model section
    # Create another XGBoost regressor (model) with fixed parameters
    model = xgb.XGBRegressor(n_estimators=300, max_depth=11, learning_rate=1.0, random_state=400)

    # Fit the model to the training data
    model.fit(X, y)
    
    # Make predictions on all data points using the model
    all_predictions_model = model.predict(X)
    
    #Applying Treshold to model 
    all_predictions_model[all_predictions_model < threshold] = 0.01
    
    # Evaluate the model performance for "model"
    mse_model = mean_squared_error(y, all_predictions_model)
    print('Mean squared error for model:', mse_model)
    
    # MAPE Prediction for model
    mape_model = calculate_mape(y, all_predictions_model)
    
    #Feature importances for model
    importances_model = model.feature_importances_
    sorted_indices_model = np.argsort(importances_model)[::-1]
    
    
if st.session_state.button_pressed:
        if jumlah_beban =="0" :

                # Create a download link
                # Get the values from the 'groups' column
                # Load the specific sheet
                xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
                worksheet_name = 'Usesheet'  # Replace with the actual name of your sheet
                worksheet = xls.parse(sheet_name=worksheet_name)

                groups = worksheet['groups'].tolist()

               # Predict data from datasheet
                new_test0 = worksheet[['beban/disp', 'Cb', 'cogm', 'B/T',]]
                predicted_Incline0 = best_model.predict(new_test0) ### the one that will learn and predict the data
                datap = {'Actual': y, 'Predicted': predicted_Incline0}
             # Calculate mape and MSE on datasheet
                # MAPE and MSE Prediction for model
                mape_datap = calculate_mape(y, predicted_Incline0)
                mse_datap = mean_squared_error(y, predicted_Incline0)
            # print the MAPE and MSE
                st.subheader(f"Mean squared error for predicting datasheet is {mse_datap}  " )
                st.subheader(f"Mean Absolute Percentage Error for predicting datasheet is {mape_datap}")
             # Preapare the .csv files
                dg = pd.DataFrame(datap)
                predictions_dg = pd.DataFrame({'Group' : groups, 'Actual':y, 'Predicted':predicted_Incline0})
                predictions_dg.to_csv( index=False, sep='|')
                st.success("Predictions saved to predictions.csv")
                
                def create_download_link(dg, filename="predictions.csv"):
                    csv_content = dg.to_csv(index=False, sep='|')  # Assuming '|' as separator
                    b64 = base64.b64encode(csv_content.encode()).decode()  # Encoding the CSV file
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
                    return href
                
                # Creating the .pkl download link
                def create_download_link(file_path='your_model.pkl', filename="your_model.pkl"):
                    with open(file_path, 'rb') as f:
                        pkl_content = f.read()
                        b64 = base64.b64encode(pkl_content).decode()
                        href = f'<a href="data:file/pkl;base64,{b64}" download="{filename}">Download Model</a>'
                    return href
            
                # Display the link
                st.markdown(create_download_link(predictions_dg), unsafe_allow_html=True)
                if download == 1 :
                     st.markdown(create_download_link(your_model.pkl), unsafe_allow_html=True)
                
                # Plotting feature importances
                imp, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(importances_best_model)), importances_best_model[sorted_indices_best_model], align='center')
                ax.set_xticks(range(len(importances_best_model)))
                ax.set_xticklabels(np.array(features)[sorted_indices_best_model])
                ax.set_title("Feature Importances")
                ax.set_ylabel('Importance')
                ax.set_xlabel('Features')

                st.pyplot(imp)  # Pass the figure object to st.pyplot()

                # Plotting Actual vs predicted value
                # make graphics
                fig, ax = plt.subplots()
                # Create a scatter plot
                scatter = ax.scatter(datap['Actual'], datap['Predicted'] , color='blue', label='Incliing result')
        
                # Set title, labels, and legend
                ax.set_title("Actual vs Predicted")
                ax.set_xlabel('Actual data')
                ax.set_ylabel('Prediction data')
                ax.legend()

                # Add annotations
                for i in range(len(datap)):
                                actual_value = pd.to_numeric(datap['Actual'].iloc[i], errors='coerce')
                                # Check if 'Predicted' is a DataFrame or NumPy array
                                if isinstance(datap['Predicted'], pd.DataFrame):
                                    predicted_value = pd.to_numeric(datap['Predicted'].iloc[i, 0], errors='coerce')
                                else:
                                    predicted_value = pd.to_numeric(datap['Predicted'][i], errors='coerce')
                    
                ax.annotate(i, (actual_value, predicted_value))
                ax.set_xlabel('Actual)')
                ax.set_ylabel('Predicted')
                  # Pass the figure object to st.pyplot()
                st.pyplot(fig)

                # make the histogram datasheet
                predicted_degrees = datap['Predicted']
                # Create a histogram with 1-degree bins
                freq, bins, _ = plt.hist(predicted_degrees, bins=np.arange(min(predicted_degrees), max(predicted_degrees) + 1, 1), edgecolor='black')

                # Set title and labels
                plt.title('Frequency Histogram of Predicted Inclining Angles')
                plt.xlabel('Inclining Angle (degrees)')
                plt.ylabel('Frequency')

                # Show the plot
                st.pyplot(plt.gcf())

                # Clear the current figure
                plt.clf()
                
    
        else:
                halfBreadth = Breadth/2
                #transfer weight
                beban_A = bebanA * halfBreadth
                beban_B = bebanB * halfBreadth
                beban_C = bebanC * halfBreadth
                beban_D = bebanD * halfBreadth
                beban_E = bebanE * halfBreadth
                beban_F = bebanF * halfBreadth
                totalB = (bebanA + bebanB + bebanC + bebanD + bebanE + bebanF)
        
                #calculate displacement
                displacement = (Lwl * Breadth * Draft * Cb)

                totdisp = (totalB / displacement)
                

                #calculate moment
                if jumlah_beban == "4" :
                # 4 pembebanan
                        kanan1 = (beban_B + beban_D)
                        kanan2 = (beban_D)
                        kanan3 = 0
                        kanan4 = (beban_A)
                        kanan5 = (beban_A + beban_C)
                        kanan6 = (beban_A + beban_B + beban_C)
                        kanan7 = (beban_A + beban_B + beban_C + beban_D)
                        kanan8 = (beban_B + beban_C + beban_D)
                        kanan9 = (beban_B + beban_D) 

                        kiri1 = (beban_A + beban_C)
                        kiri2 = (beban_A + beban_B + beban_C)
                        kiri3 = (beban_A + beban_B + beban_C + beban_D)
                        kiri4 = (beban_B + beban_C + beban_D)
                        kiri5 = (beban_B + beban_D)
                        kiri6 = (beban_D)
                        kiri7 = 0
                        kiri8 = (beban_A)
                        kiri9 = (beban_A + beban_C)
                        
                        
                
                if jumlah_beban == "6" :

                        kanan1 = (beban_A + beban_C + beban_E)
                        kanan2 = (beban_A + beban_B + beban_C + beban_E)
                        kanan3 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kanan4 = (beban_A + beban_B + beban_C + beban_D + beban_E )
                        kanan5 = (beban_A + beban_C + beban_E)
                        kanan6 = (beban_E)
                        kanan7 = 0
                        kanan8 = (beban_C + beban_E)
                        kanan9 = (beban_A + beban_C + beban_E)

                        kiri1 = (beban_B + beban_D + beban_F)
                        kiri2 = (beban_D + beban_F)
                        kiri3 = 0
                        kiri4 = (beban_F)
                        kiri5 = (beban_B + beban_D + beban_F)
                        kiri6 = (beban_A + beban_B + beban_C + beban_D + beban_F )
                        kiri7 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kiri8 = (beban_A + beban_B + beban_D + beban_F)
                        kiri9 = (beban_B + beban_D + beban_F)
                    
                     
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

                    
                Cogm1 =  ((kiri1 - kanan1) / totalB )* Breadth/2
                Cogm2 =  ((kiri2 - kanan2) / totalB )* Breadth/2 
                Cogm3 =  ((kiri3 - kanan3) / totalB )* Breadth/2
                Cogm4 =  ((kiri4 - kanan4) / totalB )* Breadth/2
                Cogm5 =  ((kiri5 - kanan5) / totalB )* Breadth/2
                Cogm6 =  ((kiri6 - kanan6) / totalB )* Breadth/2 
                Cogm7 =  ((kiri7 - kanan7) / totalB )* Breadth/2
                Cogm8 =  ((kiri8 - kanan8) / totalB )* Breadth/2
                Cogm9 =  ((kiri9 - kanan9) / totalB )* Breadth/2
                
                new_test1 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm1],'B/T' :[BT],})
                predicted_Incline1 = best_model.predict(new_test1)
        
                new_test2 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm2],'B/T' :[BT],})
                predicted_Incline2 = best_model.predict(new_test2)
        
                new_test3 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm3],'B/T' :[BT],})
                predicted_Incline3 = best_model.predict(new_test3)
        
                new_test4 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm4],'B/T' :[BT],})
                predicted_Incline4 = best_model.predict(new_test4)
        
                new_test5 = pd.DataFrame({ 'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm5],'B/T' :[BT],})
                predicted_Incline5 = best_model.predict(new_test5)
        
                new_test6 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm6],'B/T' :[BT],})
                predicted_Incline6 = best_model.predict(new_test6)
        
                new_test7 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm7],'B/T' :[BT],})
                predicted_Incline7 = best_model.predict(new_test7)
        
                new_test8 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm8],'B/T' :[BT],})
                predicted_Incline8 = best_model.predict(new_test8)

                new_test9 = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm9],'B/T' :[BT],})
                predicted_Incline9 = best_model.predict(new_test9)

                ##Convert into rad and tan θ
                radians1 = math.radians(predicted_Incline1[0])
                radians2 = math.radians(predicted_Incline2[0])
                radians3 = math.radians(predicted_Incline3[0])
                radians4 = math.radians(predicted_Incline4[0])
                radians5 = math.radians(predicted_Incline5[0])
                radians6 = math.radians(predicted_Incline6[0])
                radians7 = math.radians(predicted_Incline7[0])
                radians8 = math.radians(predicted_Incline8[0])
                radians9 = math.radians(predicted_Incline9[0])

                tantheta1 = math.tan(radians1)
                tantheta2 = math.tan(radians2)
                tantheta3 = math.tan(radians3)
                tantheta4 = math.tan(radians4)
                tantheta5 = math.tan(radians5)
                tantheta6 = math.tan(radians6)
                tantheta7 = math.tan(radians7)
                tantheta8 = math.tan(radians8)
                tantheta9 = math.tan(radians9)
                
                st.subheader(f"Mean squared error is {mse_best_model}  " )
                st.subheader(f"Mean Absolute Percentage Error is {mape_model}")
            
                dataS = pd.DataFrame({
                        'Moment Beban (Kg.m)': [Mselisih1, Mselisih2, Mselisih3, Mselisih4, Mselisih5, Mselisih6, Mselisih7, Mselisih8, Mselisih9],
                        'incline (degrees)': [predicted_Incline1[0], predicted_Incline2[0], predicted_Incline3[0], predicted_Incline4[0], predicted_Incline5[0], predicted_Incline6[0], predicted_Incline7[0], predicted_Incline8[0], predicted_Incline9[0]],
                        'incline (tan θ)' : [tantheta1, tantheta2, tantheta3, tantheta4, tantheta5, tantheta6, tantheta7, tantheta8, tantheta9]
                                   
                        })
                dataS_display = dataS.copy()
                dataS_display['Moment Beban (Kg.m)'] = dataS['Moment Beban (Kg.m)']
                dataS_display['incline (degrees)'] = dataS['incline (degrees)'].apply(lambda x: '{:.2f}'.format(x))
                dataS_display['incline (tan θ)'] = dataS['incline (tan θ)'].apply(lambda x: '{:.3f}'.format(x))

                st.table(dataS_display)
            
               
                # make graphics
                fig, ax = plt.subplots()

                # Create a scatter plot
                scatter = ax.scatter(dataS['Moment Beban (Kg.m)'], dataS['incline (tan θ)'], color='blue', label='Incliing result')
        
                # Set title, labels, and legend
                ax.set_title("Inclining graphic")
                ax.set_xlabel('Moment Beban (Kg.m)')
                ax.set_ylabel('incline (tan θ)')
                ax.legend()

                # Add annotations
                for i in range(len(dataS)):
                        ax.annotate(i, (dataS['Moment Beban (Kg.m)'].iloc[i], dataS['incline (tan θ)'].iloc[i])) # i+1 because Python's indexing starts at 0

                ax.set_xlabel('Moment Beban (Kg.m)')
                ax.set_ylabel('incline (tan θ)')

                # Customization: draw a vertical line (you can adjust this as per your requirement)
                threshold = dataS['Moment Beban (Kg.m)'].mean()  # example threshold using mean, adjust as needed
                ax.axvline(x=0, color='red', linestyle='--', label=" 0,0 coordinate")
                ax.legend()

                # Customization: draw a horizontal  line (you can adjust this as per your requirement)
                thresholds = dataS['incline (tan θ)'].mean()  # example threshold using mean, adjust as needed
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

            ###0.24554285714285714285714285714286

                st.pyplot(imp)  # Pass the figure object to st.pyplot()
    
