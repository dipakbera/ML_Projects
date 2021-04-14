#importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

import pickle
from sklearn.tree import DecisionTreeClassifier




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

            Pclass = int(request.form['Pclass'])
            Age = float(request.form['Age'])
            SibSp = int(request.form['SibSp'])
            Fare = float(request.form['Fare'])
            Sex = str(request.form['Sex'])
            em = str(request.form['Embarked'])


            saved_model='modelForPrediction.sav'
            loaded_model=pickle.load(open(saved_model,'rb'))
            d1,d2,d3=0,0,0
            if Sex=='Male':
                d1=1

            if em == 'S':
                d3=1

            if em=='Q':
                d2=1


            prediction=loaded_model.predict([[Pclass,Age,SibSp,Fare,d1,d2,d3]])
            x=prediction[0]
            if x==0:
                pred='the passenger did not survive.'
            elif x==1:
                pred='the passenger survived.'

            return render_template('results.html',prediction=pred)

        except Exception as e:
            return 'Error showing: {}'.format(e)

    else:
        return  render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

