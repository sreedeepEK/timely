import mlflow
import pandas as pd 
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    print(f"Reading dataframe for {year}-{month:02d}...")
    
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    print(f"   Using URL: {url}")
    
    df = pd.read_parquet(url)

    print(f"   Raw shape: {df.shape}")
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    print(f"   Filtered shape (1–60 min trips): {df.shape}")

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    print(f"Finished processing dataframe for {year}-{month:02d}")
    return df


def create_X(df, dv=None):
    print("Creating feature matrix")

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        print(" No DictVectorizer found → fitting a new one")
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        print("  Using existing DictVectorizer to transform")
        X = dv.transform(dicts)
    
    print(f"   Feature matrix shape: {X.shape}")
    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    print("Starting model training...")
  
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        best_params = {
            'learning_rate': 0.10638129396766673,
            'max_depth': 29,
            'min_child_weight': 5.759957543179141,
            'objective': 'reg:linear',
            'reg_alpha': 0.01851315389500763,
            'reg_lambda': 0.003994721106614769,
            'seed': 42
        }
        print(f"   Training with params: {best_params}")
        
        mlflow.log_params(best_params)
        
        booster = xgb.train(
            params=best_params, 
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )
        
        print("  Training finished")

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"Validation RMSE: {rmse:.2f}")
        
        mlflow.log_metric("rmse", rmse) 
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor.b")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        print("Model and preprocessor saved and logged to MLflow")
        
        return run.info.run_id


def main(year, month): 
    print(f"Starting pipeline for {year}-{month:02d}")

    df_train = read_dataframe(year=year, month=month)
    
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    
    df_val = read_dataframe(year=next_year, month=next_month)
    
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    target = "duration"
    y_train = df_train[target].values 
    y_val = df_val[target].values
    
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    run_id = main(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(str(run_id))
    print("Run ID written to run_id.txt")