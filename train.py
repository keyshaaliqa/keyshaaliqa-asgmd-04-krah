from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle

def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    return best_model


def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)