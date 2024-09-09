import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error


def xgboost_regressor_optuna(X: np.ndarray, y: np.ndarray, hyper_param_search_trials: Optional[int] = 0, n_splits: int = 3) -> Pipeline:
    """
    Create an XGBoost regressor pipeline with scaling and optional hyperparameter tuning using Optuna.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        hyper_param_search_trials (int, optional): Number of Optuna trials. If 0, default hyperparameters will be used. 
                                                   Defaults to 0.
        n_splits (int, optional): Number of splits for Time Series Cross-Validation. Defaults to 3.

    Returns:
        Pipeline: A pipeline with StandardScaler and XGBoost regressor.
    """
    # Check if hyperparameter tuning is requested
    if hyper_param_search_trials == 0:
        # No hyperparameter tuning; use default parameters
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor())
        ])
        model.fit(X, y)
        return model

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters to tune
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e1, log=True),
            'lambda': trial.suggest_float('lambda', 1e-5, 1e1, log=True),
        }
        # Create an XGBRegressor model with the suggested hyperparameters
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(**param))
        ])

        # Time-based cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        mae_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores.append(mae)

        return np.mean(mae_scores)

    # Create and optimize the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hyper_param_search_trials)

    # Create the final model with the best parameters and fit it
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(**study.best_params))
    ])
    best_model.fit(X, y)

    return best_model


def lasso_regressor_optuna(X: np.ndarray, y: np.ndarray, hyper_param_search_trials: Optional[int] = 0, n_splits: int = 3) -> Pipeline:
    """
    Create a Lasso regressor pipeline with scaling and optional hyperparameter tuning using Optuna.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        hyper_param_search_trials (int, optional): Number of Optuna trials. If 0, default hyperparameters will be used. 
                                                   Defaults to 0.
        n_splits (int, optional): Number of splits for Time Series Cross-Validation. Defaults to 3.

    Returns:
        Pipeline: A pipeline with StandardScaler and Lasso regressor.
    """
    # Check if hyperparameter tuning is requested
    if hyper_param_search_trials == 0:
        # No hyperparameter tuning; use default parameters
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso())
        ])
        model.fit(X, y)
        return model

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters to tune
        alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
        
        # Create a Lasso regressor with the suggested hyperparameters
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha))
        ])

        # Time-based cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        mae_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores.append(mae)

        return np.mean(mae_scores)

    # Create and optimize the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hyper_param_search_trials)

    # Create the final model with the best parameters and fit it
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=study.best_params['alpha'], max_iter=1000))
    ])
    best_model.fit(X, y)

    return best_model
