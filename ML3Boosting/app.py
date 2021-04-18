#importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

import pickle
from xgboost import XGBClassifier




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

            age = int(request.form['age'])
            workclass = str(request.form['workclass'])
            fnlwgt = int(request.form['fnlwgt'])
            edu = int(request.form['edu'])
            mstatus=str(request.form['marital-status'])
            occ=str(request.form['occupation'])
            rel=str(request.form['relationship'])
            race=str(request.form['race'])
            gender=str(request.form['gender'])
            cgain=int(request.form['capital-gain'])
            closs=int(request.form['capital-loss'])
            hpw=int(request.form['hours-per-week'])
            ncntry=str(request.form['native-country'])


            #converting entries into numerical
            workclass_map=[('Federal-gov', 0),('Local-gov', 1),('Private', 2),
                     ('Self-emp-inc', 3),
                     ('Self-emp-not-inc', 4),
                     ('State-gov', 5),
                     ('Without-pay', 6)]
            dict1={k[0]:k[1] for k in workclass_map}
            workclass=dict1[workclass]


            mstatus_map=[('Divorced', 0),
                            ('Married-AF-spouse', 1),
                            ('Married-civ-spouse', 2),
                            ('Married-spouse-absent', 3),
                            ('Never-married', 4),
                            ('Separated', 5),
                            ('Widowed', 6)]
            dict2 = {k[0]: k[1] for k in mstatus_map}
            mstatus = dict2[mstatus]

            occ_map=[('Adm-clerical', 0),
                         ('Armed-Forces', 1),
                         ('Craft-repair', 2),
                         ('Exec-managerial', 3),
                         ('Farming-fishing', 4),
                         ('Handlers-cleaners', 5),
                         ('Machine-op-inspct', 6),
                         ('Other-service', 7),
                         ('Priv-house-serv', 8),
                         ('Prof-specialty', 9),
                         ('Protective-serv', 10),
                         ('Sales', 11),
                         ('Tech-support', 12),
                         ('Transport-moving', 13)]
            dict3={k[0]:k[1] for k in occ_map}
            occ=dict3[occ]

            rel_map=[('Husband', 0),
                     ('Not-in-family', 1),
                     ('Other-relative', 2),
                     ('Own-child', 3),
                     ('Unmarried', 4),
                     ('Wife', 5)]
            dict4={k[0]:k[1] for k in rel_map}
            rel=dict4[rel]

            race_map=[('Amer-Indian-Eskimo', 0),
                         ('Asian-Pac-Islander', 1),
                         ('Black', 2),
                         ('Other', 3),
                         ('White', 4)]
            dict5={k[0]:k[1] for k in race_map}
            race=dict5[race]

            gender_map=[('Female', 0), ('Male', 1)]
            dict6={k[0]:k[1] for k in gender_map}
            gender=dict6[gender]

            ncntry_map=[('United-States',38),('Mexico',25),('Philippines',29),('Germany',10),('Puerto-Rico',32),('Canada',1),
                        ('El-Salvador',7),('India',18),('Cuba',4),('England',8),('China',2),('South',34),('Jamaica',22),
                        ('Italy',21),('Dominican-Republic',5),('others',100)]
            dict7 = {k[0]: k[1] for k in ncntry_map}
            ncntry = dict7[ncntry]


            if ncntry==100:
                pred='not available right now as model is not developed for other countries as well'

            else:
                    saved_model='final_model.pickle'
                    loaded_model=pickle.load(open(saved_model,'rb'))

                    scalar=pickle.load(open('standardScaler.pickle','rb'))

                    prediction=loaded_model.predict(scalar.transform([[age,workclass,fnlwgt,edu,mstatus,occ,rel,race,gender,cgain,closs,hpw,ncntry]]))
                    x=prediction[0]
                    if x==0:
                        pred='<=50K'
                    elif x==1:
                        pred='>50K'

            return render_template('results.html',prediction=pred)

        except Exception as e:
            return 'Error showing: {}'.format(e)

    else:
        return  render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)