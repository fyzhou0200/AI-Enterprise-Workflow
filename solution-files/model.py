import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"
IMAGE_DIR="images"

from flask import Flask
app = Flask(__name__)

@app.route('/')

def get_preprocessor():
    """
    return the preprocessing pipeline
    """

    ## preprocessing pipeline
    numeric_features = ['recent_views','recent_invoices','previous_7','previous_14','previous_28','previous_70']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])

    categorical_features = []
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])


    return(preprocessor)

def _model_train(df,tag,test=False):
    """
    example funtion to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    """


    ## start timer for runtime
    time_start = time.time()

    X,y,dates = engineer_features(df)

    rs=42

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)

    ## build models
    regressor_names = ["SGDRegressor","RandomForestRegressor", "GradientBoostingRegressor", "AdaBoostRegressor"]

    regressors = (SGDRegressor(random_state=rs),
                  RandomForestRegressor(random_state=rs),
                  GradientBoostingRegressor(random_state=rs),
                  AdaBoostRegressor(random_state=rs))

    params = [
        {"reg__penalty":["l1","l2","elasticnet"],
         "reg__learning_rate":["constant","optimal","invscaling"]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__bootstrap":[True, False]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]},
        {"reg__n_estimators":[10,30,50],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]}]

    ## train models
    models = {}
    total = len(regressor_names)
    for iteration, (name,regressor,param) in enumerate(zip(regressor_names, regressors, params)):

        pipe = Pipeline(steps=[('scaler', StandardScaler()),
                               ("reg", regressor)])

        grid = GridSearchCV(pipe, param_grid=param,
                            scoring="neg_mean_squared_error",
                            cv=5, n_jobs=-1, return_train_score=True)
        grid.fit(X_train, y_train)
        models[name] = grid, grid.best_estimator_["reg"].get_params()

    ## evaluation on the validation set
    val_scores = []
    for key, model in models.items():
        y_pred = model[0].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        val_scores.append(rmse)

    ## select best model
    bm = regressor_names[np.argmin(val_scores)]
    opt_model, params = models[bm]

    print("cuurent optimal model is: ", bm)

    ## retrain best model using all data
    opt_model.fit(X, y)

    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))

        print("... saving latest data")
        data_file = os.path.join("models", 'latest-train.pickle')
        with open(data_file, 'wb') as tmp:
            pickle.dump({'y':y, 'X':X}, tmp)

    joblib.dump(opt_model,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## plot the figure of rmse
    model_names = ['SGD', 'RF','GBM', 'ADA']
    model_rmses = val_scores

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model_names, model_rmses, width = 0.4)

    plt.xlabel("Model Names")
    plt.ylabel("Model Errors")
    plt.title("Model Training RMSE Comparisons")
    plt.show()

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':min(val_scores)},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)


def model_train(data_dir,test=False):
    """
    funtion to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():

        if test and country not in ['all','united_kingdom']:
            continue

        _model_train(df,country,test=test)

def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model

    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("data","cs-train")

    models = [f for f in os.listdir(os.path.join(".","models")) if re.search("sl",f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}

    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data,all_models = model_load(training=False)

    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")

    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]

    ## sainty check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)

    return({'y_pred':y_pred,'y_proba':y_proba})

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("data","cs-train")
    model_train(data_dir,test=False)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load()
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day)
    print(result)
