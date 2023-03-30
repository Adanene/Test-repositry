2 / 2

Here is an example Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

python
Copy code
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
sheet_id ='1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format-xlsx")
df = pd.read_excel(xls , 'Sheet2' , header = 0)

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Select the features and target variable
features = ['displacement', 'Selisih beban', 'Jenis kapal',]
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
new_test = pd.DataFrame({'displacement': [1000], 'Selisih beban': [100], 'Jenis kapal': [Kapal kargo],})
predicted_Incline = model.predict(new_test)
print('Predicted GM:', predicted_Incline)
