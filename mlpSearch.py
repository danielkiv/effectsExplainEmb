import numpy as np
from sklearn.neural_network import MLPRegressor
from flaml.automl.model import SKLearnEstimator
from flaml import tune

# Helper class for FLAML to autotune MLPRegressor
class MLP(SKLearnEstimator):
    def __init__(self, task='regression', n_jobs=None, **config):
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

    def fit(self, X_train, y_train, **kwargs):
        # Separate standard features from embedding features
        embedding_cols = [col for col in X_train.columns if isinstance(X_train[col].iloc[0], (list, np.ndarray))]
        standard_cols = [col for col in X_train.columns if col not in embedding_cols]
        
        X_standard = X_train[standard_cols].to_numpy()
        
        # If there are embedding columns, pad them and combine
        if embedding_cols:
            X_embeddings_ragged = X_train[embedding_cols].to_numpy()
            
            # Pad the ragged array
            max_len = max(len(row[0]) for row in X_embeddings_ragged)
            padded_embeddings = np.zeros((X_embeddings_ragged.shape[0], max_len))
            for i, row in enumerate(X_embeddings_ragged):
                emb = row[0]
                padded_embeddings[i, :len(emb)] = emb
            
            # Combine the standard features and the newly padded embeddings
            X_processed = np.concatenate([X_standard, padded_embeddings], axis=1)
        else:
            X_processed = X_standard

        # Now, call the original scikit-learn fit method with the clean data
        return super().fit(X_processed, y_train, **kwargs)