import mlflow
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(filename):
   
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


df_train = read_dataframe("https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-07.parquet")
df_val = read_dataframe("https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-08.parquet")


categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)



target = "duration"
y_train = df_train[target].values 
y_val = df_val[target].values


import xgboost as xgb

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val,label=y_val)


mlflow.xgboost.autolog(disable=True)
with mlflow.start_run():
    
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label = y_val)
    
    best_params = {'learning_rate' : 0.10638129396766673,
                    'max_depth' : 29,
                    'min_child_weight': 5.759957543179141,
                    'objective' : 'reg:linear',
                    'reg_alpha' : 0.01851315389500763,
                    'reg_lambda' : 0.003994721106614769,
                    'seed' : 42}
    
    
    mlflow.log_params(best_params)
    
    booster = xgb.train(
    params=best_params, 
    dtrain=train,
    num_boost_round=1000,
    evals=[(valid, 'validation')],
    early_stopping_rounds=50) 
    
    
    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse) 
    
    with open("models/preprocessor.b", "wb") as f_out:
        
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor.b")
    mlflow.xgboost.log_model(booster,artifact_path="models_mlflow")

