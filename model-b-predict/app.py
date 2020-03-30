import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

app = Flask(__name__)

df = pd.read_excel("data/default-of-credit-card clients.xls", skiprows=1)

bst = xgb.Booster()
bst.load_model("model/xgb.model")

@app.route('/', methods = ['GET'])
def root():
    return "To generate predictions, make POST requests at the /predict endpoint."

@app.route('/predict', methods = ['POST'])
def predict(params = "1,20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0"):
    params = request.json.get("input")
    input_data = np.fromstring(params, dtype=int, sep=",")
    input_data = input_data.reshape((1,-1))
    input_df = pd.DataFrame(input_data, columns=df.columns[:-1])
    dinput = xgb.DMatrix(input_df)
    return jsonify(prediction=str(bst.predict(dinput)[0]),
                   model="model-b")

if __name__ == '__main__':
   app.run()
