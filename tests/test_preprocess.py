import sys
import os
sys.path.append(os.path.abspath("src"))

from preprocess import preprocess_data

def test_preprocessing():
    df = preprocess_data("data/sample.csv")
    assert not df.empty
    assert df.isnull().sum().sum() == 0
