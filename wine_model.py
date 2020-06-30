from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model(X, y):
    """Train a random forest classifier on some data"""
    m = RandomForestClassifier()
    m.fit(X, y)
    return m

if __name__ == '__main__':

    d = load_wine()
    # print(d['DESCR'])
    X = pd.DataFrame(d['data'], columns=d['feature_names'])
    y = d['target']  # cultivator
    m = train_model(X, y)
    print(m.score(X, y))
