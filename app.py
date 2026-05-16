import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd

app = Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    # Provide default context so template fields like `prefill` are always defined
    return render_template('home.html', prefill={}, prediction_text='')

@app.route("/predict_api",methods=['POST'])
def predict_api():
    data = request.json.get('data')
    # Accept either a list/tuple (ordered features) or a dict keyed by feature name
    keys = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    if isinstance(data, (list, tuple)):
        arr = np.array(data).reshape(1, -1)
    elif isinstance(data, dict):
        try:
            arr = np.array([float(data[k]) for k in keys]).reshape(1, -1)
        except Exception:
            # fallback: use values order
            arr = np.array(list(data.values())).reshape(1, -1)
    else:
        return jsonify({'error': 'Invalid data format'}), 400
    new_data = scalar.transform(arr)
    output = regmodel.predict(new_data)
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
    try:
        out_val = float(output)
    except Exception:
        out_val = float(np.asarray(output).item())
    # Clamp negative predictions to zero for safety and format as currency
    out_val = max(0.0, out_val)
    formatted = f"${out_val:,.2f}"
    if out_val == 0.0:
        prediction_text = f"The House price prediction is {formatted} — unrealistic value (model likely produced a non-positive estimate)."
    else:
        prediction_text = f"The House price prediction is {formatted}"
    # also return the input values so the form stays filled
    keys = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    prefill = {k: v for k, v in zip(keys, data)}
    return render_template("home.html", prediction_text=prediction_text, prefill=prefill)


@app.route('/random', methods=['GET'])
def random_sample():
    # Provide a random dataset row to pre-fill the form
    dataset = fetch_california_housing()
    idx = np.random.randint(0, len(dataset.data))
    sample = dataset.data[idx]
    keys = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    prefill = {k: float(sample[i]) for i, k in enumerate(keys)}
    return render_template('home.html', prefill=prefill)


@app.route('/random_api', methods=['GET'])
def random_api():
    dataset = fetch_california_housing()
    idx = np.random.randint(0, len(dataset.data))
    sample = dataset.data[idx]
    keys = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    prefill = {k: float(sample[i]) for i, k in enumerate(keys)}
    return jsonify(prefill)




if __name__=="__main__":
    app.run(debug=True)



