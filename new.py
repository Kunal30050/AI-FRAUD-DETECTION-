from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

X, y = make_classification(n_samples=100, n_features=8, random_state=42)
model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
