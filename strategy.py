"""
This file is meant to be used for the creation of custom code used in the strategy.
"""
import pandas as pd
import numpy as np


class MyCustomRuleBasedStrategy:
    classes_ = [0.0, 1.0, -1.0]
    n_classes_ = 3

    def __init__(self):
        # add argument to your class here
        pass

    def predict(self, X: pd.DataFrame):
        # add your rules here
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        probabilities = np.zeros(shape=(len(predictions), len(self.classes_)))
        for i, class_value in enumerate(self.classes_):
            probabilities[:, i] = predictions == class_value
        return probabilities
