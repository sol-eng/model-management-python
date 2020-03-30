import requests
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

models = ["model-a", "model-b"]
prob_weights = [0.80, 0.20]
api_prefix = "https://colorado.rstudio.com/rsc/model-management-python/"
api_suffix = "-predict/predict"

@app.route('/', methods = ['GET'])
def root():
    return "To generate predictions, make POST requests at the /predict endpoint."

@app.route('/predict', methods = ['POST'])
def predict(params = "1,20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0"):
    params = request.json.get("input")
    data = {"input": params}
    selected_model = np.random.choice(models, size=1, p=prob_weights)
    model_endpoint = api_prefix + selected_model[0] + api_suffix
    r = requests.post(model_endpoint, json=data)
    resp = r.json()
    return resp

if __name__ == '__main__':
   app.run()
