import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    # Only read the expected feature fields (ignore submit/button fields)
    keys = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    form_values = [request.form.get(k, '').strip() for k in keys]
    missing = [k for k, v in zip(keys, form_values) if v == '']
    if missing:
        return render_template("home.html", prediction_text=f"Error: missing fields: {', '.join(missing)}")
    try:
        data = [float(v) for v in form_values]
    except ValueError:
        return render_template("home.html", prediction_text="Error: all inputs must be numeric.")
    # Validate feature count if scaler exposes it
    if hasattr(scalar, "n_features_in_") and len(data) != int(getattr(scalar, "n_features_in_")):
        return render_template("home.html", prediction_text=f"Error: expected {scalar.n_features_in_} inputs, got {len(data)}.")
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The House price prediction is {output}")




if __name__=="__main__":
    app.run(debug=True)



