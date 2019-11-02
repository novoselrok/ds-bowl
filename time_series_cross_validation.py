import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


def search_best_model(X, y_target, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target,
        test_size=0.20, random_state=2019, stratify=y_target
    )
    kf = StratifiedKFold(n_splits=5, random_state=2019)
    params = {
        'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.005],
        'max_depth': sp_randint(3, 12),
        'num_leaves': sp_randint(6, 50),
        'min_child_samples': sp_randint(100, 500),
        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
        'subsample': sp_uniform(loc=0.2, scale=0.8),
        'colsample_bytree': sp_uniform(loc=0.2, scale=0.8),
        'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
    }

    clf = LGBMClassifier(
        n_estimators=5000,
        objective='multiclass',
        metric='multi_error',
        random_state=2019,
        num_classes=4,
        silent=True,
        n_jobs=8
    )

    rscv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=params,
        cv=kf,
        n_iter=500,
        verbose=6,
        scoring=make_scorer(cohen_kappa_score, weights='quadratic'),
        n_jobs=2
    )

    rscv.fit(
        X_train, y_train,
        early_stopping_rounds=500,
        eval_set=[(X_test, y_test)],
        verbose=-1,
        feature_name=feature_names,
        categorical_feature=categorical_features
    )

    print(rscv.best_params_)
    print(rscv.best_score_)


def fit_model_and_output_submission(X, X_test, y_target, train_installation_ids, test_installation_ids, feature_names):
    n_splits = 10
    # kf = StratifiedKFold(n_splits=n_splits, random_state=2019)
    kf = GroupKFold(n_splits=n_splits)

    scores = []
    y_test = np.zeros((X_test.shape[0], 4))
    for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
        print(f'Starting fold {fold}...')

        x_train, x_val, y_train, y_val = X[train_split], X[test_split], y_target[train_split], y_target[test_split]
        best_params_1 = {'colsample_bytree': 0.609790382002179, 'min_child_samples': 310, 'min_child_weight': 1,
                         'num_leaves': 26, 'reg_alpha': 10, 'reg_lambda': 10, 'subsample': 0.20402842832306567}
        best_params_2 = dict(colsample_bytree=0.9610601623001905, learning_rate=0.005, max_depth=3,
                             min_child_samples=185, min_child_weight=1, num_leaves=45, reg_alpha=5, reg_lambda=5,
                             subsample=0.5108884959811426)
        best_params_3 = {'colsample_bytree': 0.20814718702813428, 'learning_rate': 0.05, 'max_depth': 11,
                         'min_child_samples': 103, 'min_child_weight': 10.0, 'num_leaves': 33, 'reg_alpha': 7,
                         'reg_lambda': 5, 'subsample': 0.3409567257294305}

        model = LGBMClassifier(
            n_estimators=100000,
            num_classes=4,
            objective='multiclass',
            metric='multi_error',
            n_jobs=-1,
            **best_params_3
        )

        model.fit(
            x_train, y_train,
            early_stopping_rounds=500,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=100,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        y_test += model.predict_proba(X_test) / n_splits

        # Diagnostic
        pred = model.predict_proba(x_val).argmax(axis=1)
        score = cohen_kappa_score(y_val, pred, weights='quadratic')
        scores.append(score)
        print(score)

        print('Fold done.')

    print(np.mean(scores))
    pd.DataFrame.from_dict({
        'installation_id': test_installation_ids,
        'accuracy_group': y_test.argmax(axis=1)
    }).to_csv('submission.csv', index=False)


categorical_features = [
    'assessment_title',
    'assessment_type',
    'assessment_world',
]

columns_to_drop = [
    'installation_id',
    'assessment_game_session',
    'assessment_start_timestamp',
]


def main():
    df_train_features = pd.read_csv('preprocessed-data/train_features.csv')
    # Make sure test columns are in the correct order
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')[df_train_features.columns.tolist()]

    df_train_labels = pd.read_csv(
        'data/train_labels.csv',
        usecols=['installation_id', 'game_session', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session', 'accuracy_group': 'target'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['installation_id', 'assessment_game_session'])

    # Extract target and installation ids
    y_target = df_train_features['target'].values
    train_installation_ids = df_train_features['installation_id']
    test_installation_ids = df_test_features['installation_id']

    # Encode categorical features
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train_features[column]
        test_column = df_test_features[column]
        lb.fit(train_column)
        df_train_features[column] = lb.transform(train_column)
        df_test_features[column] = lb.transform(test_column)

    df_train_features = df_train_features.drop(columns=columns_to_drop + ['target'])
    df_test_features = df_test_features.drop(columns=columns_to_drop)
    feature_names = df_train_features.columns.tolist()

    # Features matrix
    X = df_train_features.values
    X_test = df_test_features.values

    fit_model_and_output_submission(X, X_test, y_target, train_installation_ids, test_installation_ids, feature_names)
    # search_best_model(X, y_target, feature_names)


if __name__ == '__main__':
    main()
