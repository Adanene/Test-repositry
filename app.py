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
hydros = (
        "No",
        "Predict Automatic (Not Quite Accurate)", 
        "Yes I have My KM from Hydrostatic",
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


Stab = st.selectbox("Do you need to calculate Stability Point? (Required KM from hydrostatic)", hydros)
if Stab == "Yes I have My MG from Hydrostatic" :
        KM = st.number_input("Write KM From your Hydrostatic",min_value= 0.0000,  step =0.0001)

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
        download = 1
    else:
        print(f"Failed to download the model. Status code: {response.status_code}")    
        download = 0

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
                'subsample': [ 1],
                'colsample_bytree': [1.0],
                'reg_alpha': [1],
                'reg_lambda': [1],
                'gamma': [0],
                'min_child_weight': [6],
                'scale_pos_weight': [1]
            }
            grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid,
                                       cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', error_score='raise')
            grid_search.fit(X, y, eval_metric='rmse', eval_set=[(X, y)], early_stopping_rounds=50)
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
                st.subheader(f"Mean squared error for predicting datasheet is {mse_datap:.3f}  " )
                st.subheader(f"Mean Absolute Percentage Error for predicting datasheet is {mape_datap:.3f}")
             # Preapare the .csv files
                dg = pd.DataFrame(datap)
                predictions_dg = pd.DataFrame({'Group' : groups, 'Actual':y, 'Predicted':predicted_Incline0})
                predictions_dg.to_csv( index=False, sep='|')
                st.success("Predictions saved to predictions.csv")
                
                # Creating the CSV download link
                def create_csv_download_link(dg, filename="predictions.csv"):
                    csv_content = dg.to_csv(index=False, sep='|')  # Assuming '|' as separator
                    b64 = base64.b64encode(csv_content.encode()).decode()  # Encoding the CSV file
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
                    return href

                # Creating the .pkl download link
                def create_model_download_link(model_file_path='your_model.pkl', filename="your_model.pkl"):
                    with open(model_file_path, 'rb') as f:
                        pkl_content = f.read()
                        b64 = base64.b64encode(pkl_content).decode()
                        href = f'<a href="data:file/pkl;base64,{b64}" download="{filename}">Download Model</a>'
                    return href

                # Display the links
                st.markdown(create_csv_download_link(predictions_dg), unsafe_allow_html=True)
                
                st.markdown(create_model_download_link('your_model.pkl'), unsafe_allow_html=True)

                 # Create a scatter plot
                
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
                fig, aa = plt.subplots()
                scatter = aa.scatter(datap['Actual'], datap['Predicted'] , color='blue', label='Inclining result')
                # Set title, labels, and legend
                aa.set_title("Actual vs Predicted")
                aa.set_xlabel('Actual data')
                aa.set_ylabel('Prediction data')
                aa.legend()

                # Add annotations
                for i in range(len(datap)):
                                actual_value = pd.to_numeric(datap['Actual'].iloc[i], errors='coerce')
                                # Check if 'Predicted' is a DataFrame or NumPy array
                                if isinstance(datap['Predicted'], pd.DataFrame):
                                    predicted_value = pd.to_numeric(datap['Predicted'].iloc[i, 0], errors='coerce')
                                else:
                                    predicted_value = pd.to_numeric(datap['Predicted'][i], errors='coerce')
                    
                aa.annotate(i, (actual_value, predicted_value))
                aa.set_xlabel('Actual)')
                aa.set_ylabel('Predicted')
                  # Pass the figure object to st.pyplot()
                st.pyplot(fig)
            

                # Make the histogram datasheet
                predicted_degrees = datap['Predicted']
                  # Make the histogram datasheet
                fig_hist, ax_hist = plt.subplots()
                predicted_degrees = datap['Predicted']
                # Create a histogram with 1-degree bins
                freq, bins, _ = ax_hist.hist(predicted_degrees, bins=np.arange(min(predicted_degrees), max(predicted_degrees) + 1, 1), edgecolor='black')

                # Set title and labels
                ax_hist.set_title('Frequency Histogram of Predicted Inclining Angles')
                ax_hist.set_xlabel('Inclining Angle (degrees)')
                ax_hist.set_ylabel('Frequency')

                # Show the plot
                st.pyplot(fig_hist)
            
                # Optionally, you can clear the current figure after displaying all plots
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

                    
                Cogm1 =  ((kiri1 - kanan1) / totalB ) 
                Cogm2 =  ((kiri2 - kanan2) / totalB ) 
                Cogm3 =  ((kiri3 - kanan3) / totalB ) 
                Cogm4 =  ((kiri4 - kanan4) / totalB ) 
                Cogm5 =  ((kiri5 - kanan5) / totalB )
                Cogm6 =  ((kiri6 - kanan6) / totalB ) 
                Cogm7 =  ((kiri7 - kanan7) / totalB ) 
                Cogm8 =  ((kiri8 - kanan8) / totalB ) 
                Cogm9 =  ((kiri9 - kanan9) / totalB ) 

                new_testmid = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm1],'B/T' :[BT],})
                predicted_Inclineid = best_model.predict(new_testmid)

                new_testpt = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm3],'B/T' :[BT],})
                predicted_Inclinept = best_model.predict(new_testpt)

                new_testst = pd.DataFrame({'beban/disp': [totdisp], 'Cb': [Cb], 'cogm' :[Cogm7],'B/T' :[BT],})
                predicted_Inclinest = best_model.predict(new_testst)
            
                predicted_Incline1 = predicted_Inclineid
        
                predicted_Incline2 = Cogm2/Cogm3 * predicted_Inclinept
        
                predicted_Incline3 = predicted_Inclinept
        
                predicted_Incline4 = Cogm4/Cogm3 * predicted_Inclinept
        
                predicted_Incline5 = predicted_Inclineid
        
                predicted_Incline6 = Cogm6/Cogm7 * predicted_Inclinest
        
                predicted_Incline7 = predicted_Inclinest

                predicted_Incline8 = Cogm8/Cogm7 * predicted_Inclinest

                predicted_Incline9 = predicted_Inclineid

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
                
                st.subheader(f"Mean squared error is {mse_best_model:.3f}" )
                st.subheader(f"Mean Absolute Percentage Error is {float(mape_model):.3f}")
            
                dataS = pd.DataFrame({
                        'Posisi Cog Momen (m)': [Cogm1, Cogm2, Cogm3, Cogm4, Cogm5, Cogm6, Cogm7, Cogm8, Cogm9],
                        'incline (degrees)': [predicted_Incline1[0], predicted_Incline2[0], predicted_Incline3[0], predicted_Incline4[0], predicted_Incline5[0], 
                                                predicted_Incline6[0], predicted_Incline7[0], predicted_Incline8[0], predicted_Incline9[0]],
                        'incline (tan θ)' : [tantheta1, tantheta2, tantheta3, tantheta4, tantheta5, tantheta6, tantheta7, tantheta8, tantheta9]
                                   
                        })
                dataS_display = dataS.copy()
                dataS_display['Posisi Cog Momen (m)'] = dataS['Posisi Cog Momen (m)']
                dataS_display['incline (degrees)'] = dataS['incline (degrees)'].apply(lambda x: '{:.2f}'.format(x))
                dataS_display['incline (tan θ)'] = dataS['incline (tan θ)'].apply(lambda x: '{:.3f}'.format(x))

                st.table(dataS_display)
            
               
                # make graphics
                fig, ax = plt.subplots()

                # Create a scatter plot
                scatter = ax.scatter(dataS['Posisi Cog Momen (m)'], dataS['incline (tan θ)'], color='blue', label='Incliing result')
        
                # Set title, labels, and legend
                ax.set_title("Inclining graphic")
                ax.set_xlabel('Posisi Cog Momen (m)')
                ax.set_ylabel('incline (tan θ)')
                ax.legend()

                # Add annotations
                for i in range(len(dataS)):
                        ax.annotate(i, (dataS['Posisi Cog Momen (m)'].iloc[i], dataS['incline (tan θ)'].iloc[i])) # i+1 because Python's indexing starts at 0

                ax.set_xlabel('Posisi Cog Momen (m)')
                ax.set_ylabel('incline (tan θ)')

                # Customization: draw a vertical line (you can adjust this as per your requirement)
                threshold = dataS['Posisi Cog Momen (m)'].mean()  # example threshold using mean, adjust as needed
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
                ax.bar(range(len(importances_best_model)), importances_best_model[sorted_indices_best_model], align='center')
                ax.set_xticks(range(len(importances_best_model)))
                ax.set_xticklabels(np.array(features)[sorted_indices_best_model])
                ax.set_title("Feature Importances")
                ax.set_ylabel('Importance')
                ax.set_xlabel('Features')

                st.pyplot(imp)  # Pass the figure object to st.pyplot()

                #### Now for hydrostatic part
                ###Calulate KG
            
                if Stab == "Yes I have My KM from Hydrostatic" :
                    MG1 = (Cogm1 * totalB) / (displacement * 9.81 * tantheta1)
                    MG2 = (Cogm2 * totalB) / (displacement * 9.81 * tantheta2)
                    MG3 = (Cogm3 * totalB) / (displacement * 9.81 * tantheta3)
                    MG4 = (Cogm4 * totalB) / (displacement * 9.81 * tantheta4)
                    MG5 = (Cogm5 * totalB) / (displacement * 9.81 * tantheta5)
                    MG6 = (Cogm6 * totalB) / (displacement * 9.81 * tantheta6)
                    MG7 = (Cogm7 * totalB) / (displacement * 9.81 * tantheta7)
                    MG8 = (Cogm8 * totalB) / (displacement * 9.81 * tantheta8)
                    MG9 = (Cogm9 * totalB) / (displacement * 9.81 * tantheta9)  
                    
                    KM1 = KM
                    KM2 = KM
                    KM3 = KM
                    KM4 = KM
                    KM5 = KM
                    KM6 = KM
                    KM7 = KM
                    KM8 = KM
                    KM9 = KM

                    KG1 = MG1 + KM1
                    KG2 = MG2 + KM2
                    KG3 = MG3 + KM3
                    KG4 = MG4 + KM4
                    KG5 = MG5 + KM5
                    KG6 = MG6 + KM6
                    KG7 = MG7 + KM7
                    KG8 = MG8 + KM8
                    KG9 = MG9 + KM9
                    
                if Stab == "Predict Automatic (Not Quite Accurate)" :
                    MG1 = (Cogm1 * totalB) / (displacement * 9.81 * tantheta1)
                    MG2 = (Cogm2 * totalB) / (displacement * 9.81 * tantheta2)
                    MG3 = (Cogm3 * totalB) / (displacement * 9.81 * tantheta3)
                    MG4 = (Cogm4 * totalB) / (displacement * 9.81 * tantheta4)
                    MG5 = (Cogm5 * totalB) / (displacement * 9.81 * tantheta5)
                    MG6 = (Cogm6 * totalB) / (displacement * 9.81 * tantheta6)
                    MG7 = (Cogm7 * totalB) / (displacement * 9.81 * tantheta7)
                    MG8 = (Cogm8 * totalB) / (displacement * 9.81 * tantheta8)
                    MG9 = (Cogm9 * totalB) / (displacement * 9.81 * tantheta9)
                    
                    KG1 = (Cogm1 * totalB) / (totalB / halfBreadth)* tantheta1
                    KG2 = (Cogm2 * totalB) / (totalB / halfBreadth)* tantheta2
                    KG3 = (Cogm3 * totalB) / (totalB / halfBreadth)* tantheta3
                    KG4 = (Cogm4 * totalB) / (totalB / halfBreadth)* tantheta4
                    KG5 = (Cogm5 * totalB) / (totalB / halfBreadth)* tantheta5
                    KG6 = (Cogm6 * totalB) / (totalB / halfBreadth)* tantheta6
                    KG7 = (Cogm7 * totalB) / (totalB / halfBreadth)* tantheta7
                    KG8 = (Cogm8 * totalB) / (totalB / halfBreadth)* tantheta8
                    KG9 = (Cogm9 * totalB) / (totalB / halfBreadth)* tantheta9
                    
                    KM1 = -MG1 + KG1
                    KM2 = -MG2 + KG2
                    KM3 = -MG3 + KG3
                    KM4 = -MG4 + KG4
                    KM5 = -MG5 + KG5
                    KM6 = -MG6 + KG6
                    KM7 = -MG7 + KG7
                    KM8 = -MG8 + KG8
                    KM9 = -MG9 + KG9
                    
                #calculate average data
                AvKG = (KG1 +KG2 +KG3 +KG4 +KG5 +KG6 +KG7 +KG8 +KG9)
                AvMG = (MG1 +MG2 +MG3 +MG4 +MG5 +MG6 +MG7 +MG8 +MG9)
                AvKm = (KM1 +KM2 +KM3 +KM4 +KM5 +KM6 +KM7 +KM8 +KM9)
                rounded_AVKG = round(AvKG, 3)
                rounded_AvMG = round(AvMG, 3)
                rounded_AvKm = round(AvKm, 3)

                 # Build the table
                
                KGdata = [KG1, KG2, KG3, KG4, KG5, KG6, KG7, KG8, KG9]
                MGdata = [MG1, MG2, MG3, MG4, MG5, MG6, MG7, MG8, MG9]
                KMdata = [KM1, KM2, KM3, KM4, KM5, KM6, KM7, KM8, KM9]
                Cogmdata = [Cogm1, Cogm2, Cogm3, Cogm4, Cogm5, Cogm6, Cogm7, Cogm8, Cogm9]
                # Assuming KG, MG, and KM are your original data

                # Round each value to 0.001 depth
                rounded_KG = [round(value, 3) for value in KGdata]
                rounded_MG = [round(value, 3) for value in MGdata]
                rounded_KM = [round(value, 3) for value in KMdata]
                rounded_Cogm = [round(value, 3) for value in Cogmdata]
                # Create the DataFrame
                dataK = pd.DataFrame({
                    'Cogm (m)' : rounded_Cogm,
                    'KG (m)' : rounded_KG,
                    'MG (m)' : rounded_MG,
                    'KM (m)' : rounded_KM,
                })
                st.write("""##### Hydrostatic Point""")
                st.table(dataK)
                  # Plotting line diagram with switched X-axis and Y-axis
                fig, aa = plt.subplots()

                # Create line plot for KG vs MG with switched axes
                aa.plot(dataS['Posisi Cog Momen (m)'], dataK['KG (m)'], label='KG (m)', marker='o')

                # Create line plot for KM vs MG with switched axes
                aa.plot(dataS['Posisi Cog Momen (m)'], dataK['KM (m)'], label='KM (m)', marker='o')

                # Create line plot for MG vs MG with switched axes
                aa.plot(dataS['Posisi Cog Momen (m)'], dataK['MG (m)'], label='MG (m)', marker='o')

                # Set title, labels, and legend
                aa.set_title("Hydrostatic per moment")
                aa.set_xlabel('Portside                            Starboardside')
                aa.set_ylabel('KM, KG, MG (m)')
                aa.legend()

                # Add annotations
                for i in range(len(dataS)):
                    aa.annotate(i, (dataS['Posisi Cog Momen (m)'].iloc[i], dataS['incline (tan θ)'].iloc[i]))

                # Customization: draw a horizontal line
                threshold = dataS['Posisi Cog Momen (m)'].mean()
                aa.axhline(y=threshold, color='red', linestyle='--')
                aa.legend()

                # Customization: draw a vertical line
                thresholds = dataS['incline (tan θ)'].mean()
                aa.axvline(x=thresholds, color='red', linestyle='--')
                aa.legend()

                # Display the plot using Streamlit
                st.pyplot(fig)
                st.subheader(f"Average KG on this ship is {AvKG:.3f}  " )
                st.subheader(f"Average MG on this ship is {AvMG:.3f}  " )
                st.subheader(f"Average KM on this ship is {AvKm:.3f}  " )
