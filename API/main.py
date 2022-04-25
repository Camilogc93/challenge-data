# 1. Library imports

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
import aporia

import uuid






# Create the app
app = FastAPI()


## Model monitoring
###### Initiate Aporia ######

aporia.init(token="0f1dd74d2344ea2d2eb52f793bce3c46e1dfb88b0d497e863e0ae20845b98bb0", 
            environment="local-dev", 
            verbose=True)



# ###### Report Version Schema ######
apr_model_version = "sandbox-version"
apr_model_type = "binary"
apr_features_schema = {
    "genero": "string",
    "monto": "numeric",
    "hora": "numeric",
    "tipo_tc": "string",
    "linea_tc": "numeric",
    "interes_tc": "numeric",
    "status_txn": "string",
    "is_prime": "boolean",
    "dcto": "string",
    "cashback": "string",
    "device_score": "numeric",

    "dia": "string",

}



# Load trained Pipeline
model = load_model('app/apiligth')

# Define predict function
@app.post('/predict')
def predict(genero, monto, hora, tipo_tc, linea_tc, interes_tc, status_txn, is_prime, dcto, cashback, device_score, dia):
    data = pd.DataFrame([[genero, monto, hora, tipo_tc, linea_tc, interes_tc, status_txn, is_prime, dcto, cashback, device_score, dia]])
    data.columns = ['genero', 'monto', 'hora', 'tipo_tc', 'linea_tc', 'interes_tc', 'status_txn', 'is_prime', 'dcto', 'cashback', 'device_score', 'dia']
    predictionS = predict_model(model, data=data) 





  #  id_feature_column_name = 'pred_1337'
    apr_features_df = data
    apr_prediction_df = predictionS

    apr_model = aporia.Model(
    model_id="fraud",
    model_version=apr_model_version)
    #featureS=aporia.pandas.infer_schema_from_dataframe(data),
   # predictions=aporia.pandas.infer_schema_from_dataframe(predictionS))

    
    for features, predictions in zip(apr_features_df.to_dict("records"), apr_prediction_df.to_dict("records")):
        apr_model.log_prediction(
             id=str(uuid.uuid4()),
            features=features,
            predictions=predictions
        )

    apr_model.flush()




    return {'prediction': list(predictionS['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)