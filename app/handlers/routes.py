import this
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import os

def configure_routes(app):

    this_dir = os.path.dirname(__file__)
    predict_clf = joblib.load(os.path.join(this_dir, "predict_model.pkl"))
    grade_clf = joblib.load(os.path.join(this_dir, "grade_model.pkl"))

    @app.route('/')
    def hello():
        return "try the predict route it is great!"


    @app.route('/predict')
    def predict():
        # Receives student data and outputs 1 if they should be
        # accepted and 0 if they should not be
        failures = request.args.get('failures')
        g1 = request.args.get('G1')
        g2 = request.args.get('G2')
        school = request.args.get('school') == 'GP'
        sex = request.args.get('school') == 'male'
        query_df = pd.DataFrame({
            'failures': pd.Series(failures),
            'g1': pd.Series(g1),
            'g2': pd.Series(g2),
            'school': pd.Series(school),
            'sex': pd.Series(sex)
        })
        prediction = predict_clf.predict(query_df)
        return jsonify(np.asscalar(prediction))
    
    @app.route('/grade')
    def grade():
        # Receives student data and outputs what their expected
        # G3 grade is
        failures = request.args.get('failures')
        g1 = request.args.get('G1')
        g2 = request.args.get('G2')
        school = request.args.get('school') == 'GP'
        sex = request.args.get('school') == 'male'
        query_df = pd.DataFrame({
            'failures': pd.Series(failures),
            'g1': pd.Series(g1),
            'g2': pd.Series(g2),
            'school': pd.Series(school),
            'sex': pd.Series(sex)
        })
        prediction = grade_clf.predict(query_df)
        return jsonify(np.asscalar(prediction))

