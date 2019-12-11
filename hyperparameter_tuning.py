import math
import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from submission import get_train_test_features, label_encode_categorical_features, stratified_group_k_fold, fit_model, \
    integer_encode_params, get_lgbm_classifier, get_lgbm_regressor, get_catboost_classifier, \
    categorical_features, get_catboost_regressor, get_elasticnet_regressor, fit_sklearn_model, \
    ohe_encode_categorical_features, get_xgboost_regressor, get_xgboost_classifier

np.random.seed(0)


def cv(model_fn, score_fn, fit_fn, X, y, y_accuracy_group, installation_ids, model_params,
       n_splits=10, predict_proba=False, standard_scale=False, pca_transform=False):
    scores = []

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_test = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_test = y[train_split], y[test_split]

        if standard_scale:
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

        if pca_transform:
            pca = PCA(svd_solver='full', n_components=0.99)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            print('PCA components: ', pca.n_components_)

        model = model_fn(model_params)
        fit_fn(model, X_train, y_train, X_test, y_test)
        if predict_proba:
            pred = model.predict_proba(X_test)
        else:
            pred = model.predict(X_test)

        score = score_fn(y_test, pred)
        scores.append(score)

    return np.mean(scores)


def lgb_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def lgb_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def lgb_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def catboost_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_classifier('Logloss', 'AUC', cat_features),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def catboost_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def catboost_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def elasticnet_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_elasticnet_regressor(),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_sklearn_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        standard_scale=True,
        pca_transform=True
    )

    print(f'CV score: {score}')
    return -score


def elasticnet_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_elasticnet_regressor(),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_sklearn_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        standard_scale=True,
        pca_transform=True
    )

    print(f'CV score: {score}')
    return -score


def xgboost_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_classifier('binary:logistic', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def xgboost_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def xgboost_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def bayes_opt(fn, name, params, probes=None, n_iter=40, init_points=10):
    print(f'Optimizing {name}...')
    opt = BayesianOptimization(fn, params, verbose=2)
    if os.path.exists(f'./bayes_opt_logs/{name}.json'):
        print('Loading logs...')
        load_logs(opt, logs=[f'./bayes_opt_logs/{name}.json'])

    logger = JSONLogger(path=f'./bayes_opt_logs/{name}.json')
    opt.subscribe(Events.OPTMIZATION_STEP, logger)

    # Probe with a set of know "good" params
    if probes:
        for probe in probes:
            opt.probe(params=probe, lazy=True)

    opt.maximize(n_iter=n_iter, init_points=init_points)
    print(opt.max)


def lgb_tuning():
    CV_CORRECT_ATTEMPTS = True
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    lgb_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 15),
        'max_bin': (2, 500),
        'num_leaves': (5, 500),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'reg_alpha': (1, 30),
        'reg_lambda': (1, 30),
    }

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(lgb_cv_correct_attempts, 'lgb_cv_correct_attempts', lgb_bayes_params,
                  n_iter=200, init_points=30)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(lgb_cv_accuracy_rate_regression, 'lgb_cv_accuracy_rate_regression', lgb_bayes_params,
                  n_iter=200, init_points=30)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(lgb_cv_accuracy_group_regression, 'lgb_cv_accuracy_group_regression', lgb_bayes_params,
                  n_iter=200, init_points=30)


def catboost_tuning():
    CV_CORRECT_ATTEMPTS = True
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    catboost_bayes_params = {
        'learning_rate': (0.1, 1.0),
        'max_depth': (3, 8),
        'l2_leaf_reg': (1, 30),
    }

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(catboost_cv_correct_attempts, 'catboost_cv_correct_attempts', catboost_bayes_params,
                  n_iter=30, init_points=10)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(catboost_cv_accuracy_rate_regression, 'catboost_cv_accuracy_rate_regression', catboost_bayes_params,
                  n_iter=30, init_points=10)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(catboost_cv_accuracy_group_regression, 'catboost_cv_accuracy_group_regression', catboost_bayes_params,
                  n_iter=30, init_points=10)


def elasticnet_tuning():
    elasticnet_bayes_params = {
        'alpha': (0.0001, 1.0),
        'l1_ratio': (0.05, 0.95),
        'tol': (1e-5, 1e-3)
    }

    bayes_opt(
        elasticnet_cv_accuracy_group_regression, 'elasticnet_cv_accuracy_group_regression', elasticnet_bayes_params,
        n_iter=30, init_points=10,
    )

    bayes_opt(
        elasticnet_cv_accuracy_rate_regression, 'elasticnet_cv_accuracy_rate_regression', elasticnet_bayes_params,
        n_iter=30, init_points=10,
    )


def xgboost_tuning():
    CV_CORRECT_ATTEMPTS = False
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = False

    xgboost_bayes_params = {
        'learning_rate': (0.05, 1.0),
        'max_depth': (3, 10),
        'max_bin': (2, 500),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'reg_alpha': (1, 30),
        'reg_lambda': (1, 30),
    }

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(xgboost_cv_accuracy_group_regression, 'xgboost_cv_accuracy_group_regression', xgboost_bayes_params,
                  n_iter=20, init_points=10)

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(xgboost_cv_correct_attempts, 'xgboost_cv_correct_attempts', xgboost_bayes_params,
                  n_iter=20, init_points=10)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(xgboost_cv_accuracy_rate_regression, 'xgboost_cv_accuracy_rate_regression', xgboost_bayes_params,
                  n_iter=20, init_points=10)


if __name__ == '__main__':
    # lgb_tuning()
    xgboost_tuning()
    # elasticnet_tuning()
    # catboost_tuning()
