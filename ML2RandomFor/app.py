#importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor




app=Flask(__name__) #initializing a flask app

@app.route('/',methods=['GET']) #route to display the home
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) #route to show predicting result in a web UI
@cross_origin()
def index():
    if request.method=='POST':
        try:
            #reading the inputs given by the user

            crim = np.log(float(request.form['CRIM']))
            zn = np.sqrt(float(request.form['ZN']))
            indus = float(request.form['INDUS'])
            #chas = float(request.form['CHAS'])
            nox = float(request.form['NOX'])
            rm = float(request.form['RM'])
            age = float(request.form['AGE'])
            dis = float(request.form['DIS'])
            rad = np.log(float(request.form['RAD']))
            tax = float(request.form['TAX'])
            ptratio = float(request.form['PTRATIO'])
            b = np.log(float(request.form['B']))
            lstat = np.log(float(request.form['LSTAT']))

            filename_model='modelForPrediction.sav'
            loaded_model=pickle.load(open(filename_model,'rb'))
            scaler = pickle.load(open('standardscaler.pickle', 'rb'))

            prediction=loaded_model.predict(scaler.transform([[crim,zn,indus,nox,rm,age,dis,rad,tax,ptratio,b,lstat]]))
            #print('prediction is',prediction[0])
            return render_template('results.html',prediction=round(prediction[0],2))

        except Exception as e:
            return 'Error showing: {}'.format(e)

    else:
        return  render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

