from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset from CSV file
df = pd.read_csv('real_dataset.csv')

# One-hot encoding of categorical features
df_encoded = pd.get_dummies(df, columns=['Product', 'Region', 'Production Method'])

# Separating features (X) and target variable (y)
X = df_encoded.drop('Water Footprint', axis=1)
y = df_encoded['Water Footprint']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    product = request.form['Product']
    quantity = int(request.form['Quantity'])
    region = request.form['Region']
    method = request.form['Production Method']

    result = predict_water_footprint(product, quantity, region, method)
    
    return render_template('index.html', result=result)

def predict_water_footprint(product, quantity, region, method):
    # Create a DataFrame with all zeros for the input
    input_data = pd.DataFrame(columns=X_train.columns)
    input_data.loc[0] = 0  # Initialize all columns to zero

    # Set the specific values for the input
    input_data['Quantity'] = quantity
    
    # Set the corresponding one-hot encoded column to 1
    if 'Product_' + product in input_data.columns:
        input_data['Product_' + product] = 1
    if 'Region_' + region in input_data.columns:
        input_data['Region_' + region] = 1
    if 'Production Method_' + method in input_data.columns:
        input_data['Production Method_' + method] = 1

    # Predict the water footprint
    prediction = model.predict(input_data)
    return round(prediction[0], 2)

if __name__ == '__main__':
    app.run(debug=True)
