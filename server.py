from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import OneClassSVM
from numpy import where
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump, load
import csv

diabdata = pd.read_csv("static/diabetes_dataset__2019.csv") # get absolute path or relative path, whatever works on your computer
diabdata.dropna(subset = ['BMI', 'Sleep', 'SoundSleep'], inplace=True) #drops nan values from quantitative variables for now
diabdata = diabdata.dropna(axis=0)
diabdata = diabdata.drop(['Age','highBP','RegularMedicine','JunkFood','Stress','Pregancies','Pdiabetes','UriationFreq'],axis=1)
diab_quant = diabdata[['BMI', 'Sleep', 'SoundSleep']] #the quantiative variables we want
target = diabdata['Diabetic'] #the label we want to predict
lab_encoder = LabelEncoder() #encode target labels with value between 0 and n_classes-1
target = lab_encoder.fit_transform(target) #transform target lab encoded

app = Flask(__name__, template_folder=".") # get absolute path or relative path, whatever works on your computer
app.static_folder = 'static'
@app.route('/')
def startup():
    # print("uh")
    return render_template('templates/test2.html')

@app.route("/predict",methods=['POST','GET'])
def test():
    a = request.args.get('bmi')
    b = request.args.get('slp')
    c = request.args.get('sslp')
    d = request.args.get('g')
    e = request.args.get('fh')
    f = request.args.get('pa')
    g = request.args.get('smk')
    h = request.args.get('alc')
    i = request.args.get('jf')
    j = request.args.get('bp')
    header = ['Age','Gender','Family_Diabetes','highBP','PhysicallyActive','BMI','Smoking','Alcohol','Sleep','SoundSleep','RegularMedicine','JunkFood','Stress','BPLevel','Pregancies','Pdiabetes','UriationFreq']
    data = [0,d,e,j,f,a,g,h,b,c,0,i,0,j,0,0,0]
    data2 = [0,d,e,j,f,a,g,h,b,c,0,i,0,j,0,0,0]
    with open('temp.csv', 'w', encoding='UTF8') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerow(data)
        write.writerow(data2)
    XMLP = diabdata.drop('Diabetic', axis=1)
    yMLP = diabdata['Diabetic']

    X_trainMLP, X_testMLP, y_trainMLP, y_testMLP = train_test_split(XMLP, yMLP, test_size=0.2, random_state=12)
    cat_MLP = ['Gender', 'Family_Diabetes', 'PhysicallyActive', 'Smoking', 'Alcohol', 'JunkFood', 'BPLevel']
    # cat_MLP = ['Gender', 'Family_Diabetes', 'PhysicallyActive', 'Smoking', 'Alcohol', 'JunkFood', 'BPLevel','Age','highBP','RegularMedicine','JunkFood','Stress','Pregancies','Pdiabetes','UriationFreq']
    num_MLP = ['BMI','Sleep','SoundSleep']
    #encode categorical variables and scale quantitative
    XMLP_preproc = pd.DataFrame(OrdinalEncoder().fit_transform(XMLP), columns=XMLP.columns)

    scaler = StandardScaler()
    nums = [col for col in XMLP_preproc.columns if col not in cat_MLP]
    scaler.fit(XMLP_preproc[nums])
    XMLP_preproc[nums] = scaler.transform(XMLP_preproc[nums])
    # print(XMLP_preproc)
    #train-test 80-20
    ordscaler = OrdinalEncoder()
    ordscaler.fit_transform(XMLP)
    X_trainMLP, X_testMLP, y_trainMLP, y_testMLP = train_test_split(XMLP_preproc, yMLP, test_size=0.2, random_state=12)
    #using standard MLPClassifier activation and solver, as well as hidden layer sizes of 100 neurons for 2 layers
    #mlp = MLPClassifier(hidden_layer_sizes = (18,18), activation = 'tanh', solver = 'sgd', random_state = 42, max_iter=1000, batch_size = 25)
    mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'tanh', learning_rate_init = 0.01, batch_size = 75, hidden_layer_sizes = (20, 20), max_iter = 1000)
    mlp.fit(X_trainMLP, y_trainMLP)
    y_predMLP = mlp.predict(X_testMLP)
    # print(np.mean(y_predMLP == y_testMLP))



    # https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html
    # https://joblib.readthedocs.io/en/stable/generated/joblib.load.html
    # Not sure if this one was talked about in class, the idea is that this scales the data in the way the training data was scaled.
    dump(scaler, 'stdscaler', compress=True)
    scaler=load('stdscaler')

    dump(ordscaler, 'ord', compress=True)
    ordscaler=load('ord')

    tempdata = pd.read_csv('temp.csv')
    tempdata = tempdata.drop(['Age','highBP','RegularMedicine','JunkFood','Stress','Pregancies','Pdiabetes','UriationFreq'],axis=1)
    XMLP_preproc = pd.DataFrame(ordscaler.transform(tempdata), columns=tempdata.columns)
    nums = [col for col in XMLP_preproc.columns if col not in cat_MLP]
    scaler.transform(XMLP_preproc[nums])
    XMLP_preproc[nums] = scaler.transform(XMLP_preproc[nums])
    XMLP_preproc, X_whatever = train_test_split(XMLP_preproc, test_size=0.5, random_state=12)
    # print(XMLP_preproc)
    prediction = str(mlp.predict(XMLP_preproc))

    prediction = prediction.replace('[\'', '')
    prediction = prediction.replace('\']', '')


    return render_template('templates/predict_temp.html', prediction = "Yes" if (prediction=="yes") else "No")
    

if(__name__=='__main__'):
    app.run(debug=True)   
