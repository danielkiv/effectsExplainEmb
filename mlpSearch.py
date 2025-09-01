from flaml.automl.model import BaseEstimator
from sklearn.neural_network import MLPRegressor
from flaml import tune

# --- Custom MLP Learner for FLAML ---
class MLP(BaseEstimator):
    '''Custom scikit-learn MLP learner for FLAML'''

    def __init__(self, **config):
        super().__init__()
        self.model = MLPRegressor(random_state=1, **config)

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    @classmethod
    def search_space(cls, data_size, task):
        """
        Define the hyperparameter search space for the MLP.
        """
        # Each hyperparameter's value must be a dictionary containing a 'domain' key.
        return {
            'hidden_layer_sizes': {
                'domain': tune.choice([[50], [100], [50, 50], [100, 50]])
            },
            'activation': {
                'domain': tune.choice(['relu', 'tanh'])
            },
            'solver': {
                'domain': tune.choice(['adam', 'sgd'])
            },
            'alpha': {
                'domain': tune.loguniform(1e-5, 1e-1)
            },
            'learning_rate_init': {
                'domain': tune.loguniform(1e-4, 1e-2)
            },
            'max_iter': {
                'domain': tune.choice([500, 1000])
            }
        }