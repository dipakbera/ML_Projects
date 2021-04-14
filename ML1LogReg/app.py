#importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

import pickle
from sklearn.linear_model import LogisticRegression




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

            rate_marriage = float(request.form['rate_marriage'])
            yrs_married= float(request.form['yrs_married'])
            children = int(request.form['children'])
            religious = float(request.form['religious'])
            education = float(request.form['education'])
            occupation = float(request.form['occupation'])
            occupation_hus = float(request.form['occupation_hus'])
            extra_marital_time = float(request.form['extra_marital_time'])


            filename_model='finalModel.sav'
            loaded_model=pickle.load(open(filename_model,'rb'))
            scaler = pickle.load(open('standardScaler.sav', 'rb'))

            prediction=loaded_model.predict(scaler.transform([[rate_marriage,yrs_married,children,religious,education,occupation,occupation_hus,extra_marital_time]]))
            val=prediction[0]
            if val==0:
                result='she does not have any extra-marital affair.'
            else:
                result='she has atleast one extra-marital affair.'
            return render_template('results.html',prediction=result) #round(prediction[0],2))

        except Exception as e:
            return 'Error showing: {}'.format(e)

    else:
        return  render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

