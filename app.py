from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from flask import jsonify
from flask import render_template
model1 = DecisionTreeClassifier()





app = Flask(__name__)
df = pd.DataFrame()  
X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global df  
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        try:
            df = pd.read_csv(file)
            table_html = df.head().to_html()
            return render_template('index.html', table=table_html)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

@app.route('/preprocess')
def preprocess():
    global df
    if df.empty:
        return render_template('index.html', message='DataFrame is empty. Upload a file first.')

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category').cat.codes


    table_html = df.head().to_html()
    return render_template('index.html', table=table_html, message='Preprocessing completed.')

@app.route('/pie_chart')
def pie_chart():
    global df
    if df.empty:
        return render_template('index.html', message='DataFrame is empty. Upload a file first.')

    type_counts = df['type'].value_counts()
    transaction_types = type_counts.index
    quantities = type_counts.values
    fig = px.pie(df, values=quantities, names=transaction_types, hole=0.4, title='Distribution of transaction types')
    chart_html = fig.to_html(full_html=False)
    return render_template('index.html', chart=chart_html)

@app.route('/split')
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None and not df.empty:
        try:
            X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
            Y = df["isFraud"]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
            message = f'Split completed successfully. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}'
            return render_template('index.html', message=message)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

    else:
        return render_template('index.html', message='Error: Data not loaded or empty. Please click "Show" first.')
@app.route('/logistic_regression')
def logistic_regression():
    global df, X_train, X_test, y_train, y_test
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) * 100
        return render_template('accuracy.html', accuracy=accuracy)

    else:
        return render_template('index.html', message='Error: Split data first before running Logistic Regression.')
    
@app.route('/decision_tree')
def decision_tree():
    global df, X_train, X_test, y_train, y_test, model1
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        model1 = DecisionTreeClassifier()
        model1.fit(X_train, y_train)
        accuracy = model1.score(X_test, y_test) * 100
        return render_template('accuracy1.html', accuracy=accuracy, model_name='Decision Tree')

    else:
        return render_template('index.html', message='Error: Split data first before running Decision Tree.')

@app.route('/make_prediction', methods=['GET', 'POST'])
def make_prediction():
    global model1 
    if request.method == 'POST':
        type_value = int(request.form['type'])
        amount_value = float(request.form['amount'])
        oldbalanceOrg_value = float(request.form['oldbalanceOrg'])
        newbalanceOrig_value = float(request.form['newbalanceOrig'])

        input_values = [type_value, amount_value, oldbalanceOrg_value, newbalanceOrig_value]
        print(input_values)
        prediction = model1.predict([input_values])

        print("Prediction:", prediction)

        return render_template('prediction_result.html', prediction=prediction)

    return render_template('make_prediction.html')



if __name__ == '__main__':
    app.run(debug=True)
