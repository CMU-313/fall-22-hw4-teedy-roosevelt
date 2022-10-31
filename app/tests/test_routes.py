from flask import Flask
import pytest
import pandas as pd
import csv
import math
import random

from app.handlers.routes import configure_routes

#training_indeces = random.sample(range(0, data_file_len), num_training_students)

#generated only once from this code: random.sample(range(0, data_file_len), num_training_students)
training_indeces = [229, 247, 346, 288, 143, 212, 258, 265, 189, 385, 106, 284, 
136, 327, 180, 174, 190, 99, 172, 306, 41, 395, 211, 226, 209, 168, 371, 154, 
85, 91, 310, 44, 72, 195, 287, 271, 255, 75, 241, 204, 133, 243, 7, 23, 140, 113, 
185, 5, 382, 268, 364, 210, 46, 305, 367, 389, 321, 119, 353, 236, 267, 39, 257, 393, 
63, 94, 50, 338, 330, 181, 203, 55, 13, 109, 196, 70, 264, 347, 62, 324, 331, 234, 238, 
313, 381, 383, 295, 96, 157, 8, 370, 123, 291, 298, 4, 227, 394, 349, 128, 150, 225, 175, 93, 
343, 376, 359, 147, 315, 111, 240, 122, 24, 299, 166, 297, 20, 149, 362, 187, 15, 355, 156, 
216, 374, 138, 2, 28, 31, 188, 89, 283, 239, 377, 1, 354, 169, 273, 201, 259, 218, 333, 167, 
318, 342, 358, 290, 345, 214, 88, 141, 253, 73, 270, 129, 351, 199, 260, 252, 282, 329, 289, 116, 
350, 121, 339, 115, 296, 84, 131, 279, 57, 155, 266, 390, 37, 231, 248, 9, 58, 117, 263, 308, 302, 
153, 285, 369, 40, 192, 222, 215, 388, 162, 124, 120, 34, 52, 183, 173, 365, 98, 200, 49, 159, 151, 
312, 228, 207, 348, 92, 366, 235, 320, 22, 130, 178, 202, 326, 378, 11, 294, 179, 10, 380, 205, 
194, 328, 392, 25, 386, 311, 303, 87, 269, 224, 127, 363, 135, 90, 64, 256, 337, 32, 146, 145, 184, 
56, 100, 206, 97, 356, 95, 74, 317, 344, 319, 45, 340, 307, 132, 3, 81, 108, 232, 112, 148, 242, 163, 
213, 361, 38, 223, 142, 27, 83, 276, 105, 0, 245, 59, 262, 29, 198, 373, 65, 101, 221, 332, 275, 43, 251, 
191, 125, 219, 71, 250, 19, 309, 53, 182, 21, 230, 322, 341, 186, 16, 26, 208, 244, 42, 48, 
86, 134, 171, 103, 237, 379]



df = pd.read_csv("data/student-mat.csv", sep=";")
testing_indeces = list(range(len(df) + 1))
for idx in training_indeces:
    testing_indeces.remove(idx)

#so now test indices should contain all other indeces that training indeces does not

combined_indeces = list(range(len(df)))


def test_base_route():
    """Verifying that the flask app is working on the base route"""
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'

@pytest.fixture
def acceptable_predict_accuracy():
    """The minimum accuracy required for the accept/reject predictions to be
       considered sufficiently correct."""
    return 0.75

@pytest.fixture
def acceptable_grade_tolerance():
    """The maximum average error required for the grade predictions to be
       considered sufficiently correct."""
    return 2
    
# Todo: fill in actual data
@pytest.fixture(
    params=[
        pytest.param(training_indeces, id="training"),  
        pytest.param(testing_indeces, id="training"),
        pytest.param(combined_indeces, id="combined"),
    ]
)
def student_data(request):
    """The various combinations of student data used for testing"""
    return df.iloc[request.param]
    
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
