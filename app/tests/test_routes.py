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

    
    
@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame({ 'age' : pd.Series([1, 10, 100]) ,'health' : pd.Series([15, 18, 4]) ,'absences' : pd.Series([10, 0, 4]), "G3": pd.Series([5, 18, 3])}),
            id="testA"
        ),
        pytest.param(
            pd.DataFrame({ 'age' : pd.Series([1, 10, 100]) ,'health' : pd.Series([15, 18, 4]) ,'absences' : pd.Series([10, 0, 4]), "G3": pd.Series([12, 13, 8])}),
            id="testB"
        )
    ]
)
def training_data(request):
    return request.param
    
@pytest.fixture
def acceptable_ratio():
    return 0.75
    
def test_predict_accuracy(training_data, acceptable_ratio):
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    inputs = pd.DataFrame({...})
    
    correct = 0
    total = 0
    
    for _, row in training_data.iterrows():
        exp = 1 if row["G3"] >= 15 else 0
        url = '/predict'
        response = client.get(url, query_string={'age': row["age"], "health": row["health"], "absences": row["absences"]})
        assert response.status_code == 200
        total += 1
        correct += int(response.get_data()) == exp
        
    assert correct / total >= acceptable_ratio