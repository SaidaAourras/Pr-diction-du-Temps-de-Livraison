from pipeline import split_X_Y , training_model_with_Pipeline , type_columns , clean_data
from sklearn.svm import SVR
import pytest

def test_data_dimension():
    X , y = split_X_Y()
    assert X.shape[0] == len(y)

# fonction type colonnes 
def test_format_data():
    data = clean_data()
    cat_col , num_col = type_columns(data)
    for c , n in zip(cat_col , num_col):
        assert data[c].dtypes == 'object'
        assert data[n].dtypes in ['float64','int64']
    
model = SVR()
def test_max_mae():
    _ , mae = training_model_with_Pipeline(model)
    assert mae <= 7
    
