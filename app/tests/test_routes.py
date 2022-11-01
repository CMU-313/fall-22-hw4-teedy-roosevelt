from flask import Flask
import pytest
import pandas as pd

from test_utils import *

from app.handlers.routes import configure_routes


def test_base_route():
    """Verifying that the flask app is working on the base route"""
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'
    
    
def test_predict_accuracy(student_data, acceptable_predict_accuracy):
    """Tests the accuracy of the /predict endpoint"""
    
    # Set up the Flask app
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    # Keep track of the total number of students queried and how many
    # of them were correctly accepted/rejected
    correct = 0
    total = 0
    
    # Loop over each student in the input data and make the request to
    # the /predict endpoint
    for _, row in student_data.iterrows():
        exp = 1 if row["G3"] >= 15 else 0
        url = '/predict'
        response = client.get(url, query_string=row.drop("G3").to_dict())
        
        # Verify that the response was not malformed
        assert response.status_code == 200
        total += 1
        
        # Increase the number of correct predictions if the responses
        # matches the expected value
        correct += int(response.get_data()) == exp
        
    # Verify that the correctness ratio is above the minimum requirement
    assert correct / total >= acceptable_predict_accuracy
    
    
def test_grade_accuracy(student_data, acceptable_grade_tolerance):
    """Tests the accuracy of the /grade endpoint"""
    
    # Set up the Flask app
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    # Keep track of the total number of students queried and the
    # total error of all grade predictions
    error = 0
    total = 0
    
    # Loop over each student in the input data and make the request to
    # the /grade endpoint
    for _, row in student_data.iterrows():
        exp = row["G3"]
        url = '/grade'
        response = client.get(url, query_string=row.drop("G3").to_dict())
        
        # Verify that the response was not malformed
        assert response.status_code == 200
        total += 1
        
        # Add the absolute difference between the grade prediction and
        # the actual grade to the total error
        error += abs(int(response.get_data()) - exp)
        
    # Verify that the average error is below the maximum tolerable error
    assert error / total <= acceptable_grade_tolerance
