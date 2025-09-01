
from sklearn.neural_network import MLPRegressor
from flaml.automl.model import SKLearnEstimator
from flaml import tune

# Helper class for FLAML to autotune MLPRegressor

class MLP(SKLearnEstimator):
    def __init__(self, task='regression', **config):
        super().__init__(task, **config)
        self.estimator_class = MLPRegressor

    @classmethod
    def search_space(cls, data_size, task, **kwargs):
        return {
            'hidden_layer_sizes': {
                'domain': tune.choice([(100,), (50, 50), (100, 50, 25)]),
                'init_value': (100,),
            },
            'activation': {
                'domain': tune.choice(['relu', 'tanh']),
                'init_value': 'relu',
            },
            'solver': {
                'domain': tune.choice(['adam', 'sgd']),
                'init_value': 'adam',
            },
            'alpha': {
                'domain': tune.loguniform(1e-5, 1),
                'init_value': 1e-3,
            },
            'learning_rate_init': {
                'domain': tune.loguniform(1e-4, 0.1),
                'init_value': 1e-3,
            },
        }