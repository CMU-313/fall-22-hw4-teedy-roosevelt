import pytest
import pandas as pd
import csv

@pytest.fixture
def training_data():
    """The DataFrame of data used for training and testing"""
    return pd.read_csv("data/student-mat.csv", sep=";")


@pytest.fixture
def training_indices():
    """The random subset of rows (20%) used for training the model"""
    return [229, 247, 346, 288, 143, 212, 258, 265, 189, 385, 106, 284, 
            136, 327, 180, 174, 190, 99, 172, 306, 41, 391, 211, 226, 209, 168, 371, 154, 
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


@pytest.fixture
def testing_indices(training_indices, training_data):
    """The random subset of rows (80%) used for testing the model"""
    return [i for i in range(len(training_data)) if i not in training_indices]


@pytest.fixture
def combined_indices(training_data):
    return list(range(len(training_data)))


@pytest.fixture(
    params=[
        pytest.param("testing"),
        pytest.param("training"),  
        pytest.param("combined"),
    ]
)
def student_data(request, training_data, training_indices, testing_indices, combined_indices):
    """The various combinations of student data used for testing"""
    indices = {
        "training": training_indices,
        "testing": testing_indices,
        "combined": combined_indices,
    }[request.param]
    return training_data.iloc[indices]


@pytest.fixture
def acceptable_predict_accuracy():
    """The minimum accuracy required for the accept/reject predictions to be
       considered sufficiently correct."""
    return 0.85


@pytest.fixture
def acceptable_grade_tolerance():
    """The maximum average error required for the grade predictions to be
       considered sufficiently correct."""
    return 2