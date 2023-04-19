from sklearn.linear_model import LogisticRegression
from joblib import dump

def main(data, labels):    
    model = LogisticRegression().fit(data, labels)
    dump(model, "model/model.joblib")