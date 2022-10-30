from flask import Flask
import pytest
import pandas as pd

from app.handlers.routes import configure_routes


def test_base_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'

@pytest.fixture
def acceptable_predict_accuracy():
    return 0.75

@pytest.fixture
def acceptable_grade_tolerance():
    return 2
    
# Todo: fill in actual data
@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame({ 'age' : pd.Series([1, 10, 100]) ,'health' : pd.Series([15, 18, 4]) ,'absences' : pd.Series([10, 0, 4]), "G3": pd.Series([5, 18, 3])}),
            id="training"
        ),
        pytest.param(
            pd.DataFrame({ 'age' : pd.Series([1, 10, 100]) ,'health' : pd.Series([15, 18, 4]) ,'absences' : pd.Series([10, 0, 4]), "G3": pd.Series([12, 13, 8])}),
            id="test"
        ),
        pytest.param(
            pd.DataFrame({ 'age' : pd.Series([1, 10, 100]) ,'health' : pd.Series([15, 18, 4]) ,'absences' : pd.Series([10, 0, 4]), "G3": pd.Series([12, 13, 8])}),
            id="combined"
        )
    ]
)
def student_data(request):
    return request.param
    
def test_predict_accuracy(student_data, acceptable_predict_accuracy):
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    inputs = pd.DataFrame({...})
    
    correct = 0
    total = 0
    
    for _, row in student_data.iterrows():
        exp = 1 if row["G3"] >= 15 else 0
        url = '/predict'
        response = client.get(url, query_string=row.drop("G3").to_dict())
        assert response.status_code == 200
        total += 1
        correct += int(response.get_data()) == exp
        
    assert correct / total >= acceptable_predict_accuracy
    
    
def test_grade_accuracy(student_data, acceptable_grade_tolerance):
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    inputs = pd.DataFrame({...})
    
    error = 0
    total = 0
    
    for _, row in student_data.iterrows():
        exp = row["G3"]
        url = '/grade'
        response = client.get(url, query_string=row.drop("G3").to_dict())
        assert response.status_code == 200
        total += 1
        error += abs(int(response.get_data()) - exp)
        
    assert error <= acceptable_grade_tolerance