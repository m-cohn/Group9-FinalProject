import numpy as np
import pandas as pd
import sksound.sounds as sounds
from sklearn.linear_model import LogisticRegression
from joblib import dump

def read_audio(df, col):
    return sounds.read_sound(df[col].to_list())

def get_labels(df, col):
    return df[col].to_list()

def load(csv):
    df = pd.read_csv(csv)
    labels = get_labels(df, "native_language")
    audio = read_audio(df, "filename")
    data = np.concatenate(audio, axis=0)
    return data, labels

def main(path):
    csv = path / "speakers_all.csv"
    data, labels = load(csv)
    model = LogisticRegression().fit(data, labels)
    dump(model, path / "model/model.joblib")