import predict


ride = {
    "PULocationID": 10, 
    "DOLocationID": 35,
    "trip_distance": 40 
    }


pred = predict.predict(ride)

print(pred)