import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

categorical_features = [
    'assessment_title',
    'assessment_world',
]

columns_to_drop = [
    'installation_id',
    'assessment_game_session',
    # 'assessment_start_timestamp',
]


def bayes_lgb(X, y_target, train_installation_ids, feature_names):
    def lgb_eval(**params):
        for param in cast_to_int:
            if param in params:
                params[param] = int(params[param])

        kf = GroupKFold(n_splits=params['n_splits'])
        del params['n_splits']

        scores = []
        for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
            x_train, x_val, y_train, y_val = X[train_split], X[test_split], y_target[train_split], y_target[test_split]

            model = LGBMClassifier(
                random_state=2019,
                n_estimators=5000,
                num_classes=4,
                objective='multiclass',
                metric='multi_logloss',
                n_jobs=-1,
                **params
            )

            model.fit(
                x_train, y_train,
                early_stopping_rounds=500,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                verbose=0,
                feature_name=feature_names,
                categorical_feature=categorical_features
            )

            # Diagnostic
            pred = model.predict_proba(x_val).argmax(axis=1)
            score = cohen_kappa_score(y_val, pred, weights='quadratic')
            scores.append(score)

        return np.mean(scores)

    cast_to_int = ['max_depth', 'num_leaves', 'min_child_samples', 'n_splits']

    lgb_bo = BayesianOptimization(lgb_eval, {
        'n_splits': (8, 11),
        'learning_rate': (0.005, 0.1),
        'max_depth': (5, 10),
        'num_leaves': (20, 700),
        'min_child_samples': (50, 500),
        'min_child_weight': (1e-5, 1e4),
        'subsample': (0.1, 1.0),
        'colsample_bytree': (0.1, 1.0),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    })

    lgb_bo.maximize(n_iter=200, init_points=60)
    print(lgb_bo.max)
    with open('tmp.json', 'w', encoding='utf-8') as f:
        f.write(str(lgb_bo.max))


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
        best_params_4 = {'colsample_bytree': 0.5971164870817448, 'learning_rate': 0.0777901292547735,
                         'max_depth': 3, 'min_child_samples': 66,
                         'min_child_weight': 715.5457196501912,
                         'num_leaves': 18, 'reg_alpha': 0.24033071667305395,
                         'reg_lambda': 65.40955854278762, 'subsample': 0.31442139140586617}
        best_params_5 = {'colsample_bytree': 0.3909905519188057, 'learning_rate': 0.01,
                         'max_depth': 14, 'min_child_samples': 494,
                         'min_child_weight': 644.1218422907667,
                         'num_leaves': 7, 'reg_alpha': 2.8507673240668163,
                         'reg_lambda': 1.8450857631793993, 'subsample': 0.13477625349661213}
        model = LGBMClassifier(
            random_state=2019,
            n_estimators=100000,
            num_classes=4,
            objective='multiclass',
            metric='multi_error',
            n_jobs=-1,
            **best_params_5
        )

        model.fit(
            x_train, y_train,
            early_stopping_rounds=500,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=100,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        # lgb.plot_importance(model, figsize=(12, 60), max_num_features=20)
        # plt.show()

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


def get_feature_importances(X, y_target, train_installation_ids, feature_names):
    n_splits = 10
    kf = GroupKFold(n_splits=n_splits)

    feature_importances = np.zeros(X.shape[1])
    for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
        x_train, x_val, y_train, y_val = X[train_split], X[test_split], y_target[train_split], y_target[test_split]
        model = LGBMClassifier(
            random_state=2019,
            n_estimators=100000,
            num_classes=4,
            objective='multiclass',
            metric='multi_error',
            n_jobs=-1,
        )

        model.fit(
            x_train, y_train,
            early_stopping_rounds=500,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=100,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )
        feature_importances += model.feature_importances_ / n_splits

    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': feature_importances}).sort_values('importance',
                                                                                        ascending=False)
    feature_importances.to_csv('feature_importances.csv', index=False)


def main():
    feature_importances = pd.read_csv('feature_importances.csv')
    non_zero_features = set(feature_importances[feature_importances['importance'] != 0.0]['feature'])
    df_train_features = pd.read_csv('preprocessed-data/train_features.csv')
    used_columns = ['installation_id', 'assessment_game_session'] + list(
        set(df_train_features.columns.tolist()) & non_zero_features)
    df_train_features = df_train_features[used_columns]
    # Make sure test columns are in the correct order
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')[used_columns]

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
    # bayes_lgb(X, y_target, train_installation_ids, feature_names)
    # get_feature_importances(X, y_target, train_installation_ids, feature_names)


if __name__ == '__main__':
    main()
