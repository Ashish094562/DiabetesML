from flask import Flask,render_template,request, jsonify
import pickle
from flask_cors import CORS
import pandas as pd
app= Flask(__name__)
model = pickle.load(open('diabetes_prediction_model.pkl','rb'))
CORS(app)
@app.route('/')
def home():
    result=''
    return render_template('index.html',**locals())
@app.route('/predict',methods=['POST','GET'])
def predict():
    gender = str(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history = str(request.form['smoking_history'])
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])
    input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
    })
    result = model.predict(input_data)[0]
    if result == 0:
        result = 'Low risk'
    else:
        result = 'High risk'
    
    return jsonify({"result": result})
if __name__=="__main__":
    app.run(debug=True)