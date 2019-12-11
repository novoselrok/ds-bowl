import csv
import json
import math
import os
import gc
import random
from collections import defaultdict, Counter
from functools import partial

import pandas as pd
import numpy as np
import scipy.optimize
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'
INTEGER_PARAMS = ['max_depth', 'max_bin', 'num_leaves',
                  'min_child_samples', 'n_splits', 'subsample_freq']

np.random.seed(0)


def fillna0(df):
    return df.fillna(0.0)


def preprocess_events():
    def _postprocessing(game_sessions):
        for idx in range(len(game_sessions)):
            event_codes = dict(game_sessions[idx]['event_codes'])
            event_ids = dict(game_sessions[idx]['event_ids'])
            game_time = np.max(game_sessions[idx]['game_times'])
            game_times_sorted = np.array(list(sorted(game_sessions[idx]['game_times'])))

            if game_times_sorted.shape[0] <= 1:
                game_time_mean_diff = 0.0
                game_time_std_diff = 0.0
            else:
                game_times_diff = game_times_sorted[1:] - game_times_sorted[:-1]
                game_time_mean_diff = np.mean(game_times_diff)
                game_time_std_diff = np.std(game_times_diff)

            del game_sessions[idx]['game_times']
            del game_sessions[idx]['event_codes']
            del game_sessions[idx]['event_ids']

            correct, uncorrect = game_sessions[idx]['correct_attempts'], game_sessions[idx]['uncorrect_attempts']
            if correct == 0 and uncorrect == 0:
                pass
            else:
                game_sessions[idx]['accuracy_rate'] = correct / float(correct + uncorrect)

            if correct == 0 and uncorrect == 0:
                pass
            elif correct == 1 and uncorrect == 0:
                game_sessions[idx]['accuracy_group'] = 3
                game_sessions[idx]['accuracy_group_3'] = 3
            elif correct == 1 and uncorrect == 1:
                game_sessions[idx]['accuracy_group'] = 2
                game_sessions[idx]['accuracy_group_2'] = 1
            elif correct == 1 and uncorrect >= 2:
                game_sessions[idx]['accuracy_group'] = 1
                game_sessions[idx]['accuracy_group_1'] = 1
            else:
                game_sessions[idx]['accuracy_group'] = 0
                game_sessions[idx]['accuracy_group_0'] = 1

            game_sessions[idx] = {
                **game_sessions[idx],
                **event_codes,
                **event_ids,
                'game_time': game_time,
                'game_time_mean_diff': game_time_mean_diff,
                'game_time_std_diff': game_time_std_diff,
            }

        return game_sessions

    def _aggregate_game_sessions(csv_path, prefix, installation_ids_to_keep=None):
        game_sessions = {}
        with open(csv_path, encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            for idx, row in enumerate(reader):
                if installation_ids_to_keep and row['installation_id'] not in installation_ids_to_keep:
                    continue

                if row['game_session'] not in game_sessions:
                    game_sessions[row['game_session']] = {
                        'game_session': row['game_session'],
                        'timestamp': row['timestamp'],
                        'installation_id': row['installation_id'],
                        'event_count': 0,
                        'game_times': [],
                        'correct_attempts': 0,
                        'uncorrect_attempts': 0,
                        'title': row['title'],
                        'type': row['type'],
                        'world': row['world'],
                        'event_codes': defaultdict(int),
                        'event_ids': defaultdict(int),
                    }

                game_session = game_sessions[row['game_session']]

                game_session['event_count'] = max(game_session['event_count'], int(row['event_count']))
                game_session['game_times'].append(int(row['game_time']))

                event_data = json.loads(row['event_data'])
                event_code = int(row['event_code'])

                game_session['event_codes'][f'event_code_{event_code}'] += 1

                is_assessment = row['type'] == 'Assessment'
                if is_assessment:
                    is_bird_measurer_attempt = row['title'] == BIRD_MEASURER_ASSESSMENT and event_code == 4110
                    is_non_bird_measurer_attempt = row['title'] != BIRD_MEASURER_ASSESSMENT and event_code == 4100
                    is_assessment_attempt = is_bird_measurer_attempt or is_non_bird_measurer_attempt

                    if is_assessment_attempt and 'correct' in event_data:
                        if event_data['correct']:
                            game_session['correct_attempts'] += 1
                        else:
                            game_session['uncorrect_attempts'] += 1

                if row['type'] == 'Game' and 'correct' in event_data:
                    if event_data['correct']:
                        game_session['correct_attempts'] += 1
                    else:
                        game_session['uncorrect_attempts'] += 1

                event_id = row['event_id']
                game_session['event_ids'][f'event_id_{event_id}'] += 1

                if idx % 10000 == 0:
                    print(idx)

        df_data = pd.DataFrame(_postprocessing(list(game_sessions.values())))
        fillna0_columns = [column for column in df_data.columns if column not in ['accuracy_group', 'accuracy_rate']]
        df_data[fillna0_columns] = df_data[fillna0_columns].fillna(0.0)
        df_data.to_csv(f'preprocessed-data/{prefix}_game_sessions.csv', index=False)

    os.makedirs('preprocessed-data', exist_ok=True)

    if not os.path.exists(TRAIN_FEATURES_CSV):
        df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)

        # Only keep installation ids from train labels
        _aggregate_game_sessions(
            TRAIN_CSV,
            'train',
            installation_ids_to_keep=set(df_train_labels['installation_id'].unique())
        )

    _aggregate_game_sessions(
        TEST_CSV,
        'test'
    )

    pd.read_csv(TEST_CSV) \
        .groupby(['installation_id']) \
        .last() \
        .reset_index() \
        .to_csv('preprocessed-data/test_assessments_to_predict.csv', index=False)


def feature_engineering():
    aggfns = {'std': np.nanstd, 'mean': np.nanmean, 'sum': np.nansum}

    def _add_time_features(df):
        timestamp = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
        df['assessment_hour'] = timestamp.dt.hour
        df['assessment_dayofweek'] = timestamp.dt.dayofweek

    def _agg_game_sessions(values, columns, prefix=''):
        agg_columns = []
        agg = []
        for name, aggfn in aggfns.items():
            agg_columns.extend(
                [f'{prefix}{column}_{name}' for column in columns]
            )
            agg.append(
                aggfn(values, axis=0).reshape(1, -1)
            )

        return pd.DataFrame(np.concatenate(agg, axis=1), columns=agg_columns)

    def _compute_features(df_data, prefix, df_assessments):
        game_sessions = []

        event_codes_columns = [column for column in df_data.columns if column.startswith('event_code')]
        event_ids_columns = [column for column in df_data.columns if column.startswith('event_id')]

        df_assessments = df_assessments.sort_values(by='installation_id')
        installation_id_to_game_sessions = dict(tuple(df_data.groupby('installation_id')))

        _aggregate_game_sessions_columns = (aggregate_game_sessions_columns +
                                            event_codes_columns +
                                            event_ids_columns)

        total_assessments = df_assessments.shape[0]
        for idx, assessment in df_assessments.iterrows():
            installation_id = assessment['installation_id']
            start_timestamp = assessment['timestamp']

            game_sessions_for_installation_id = installation_id_to_game_sessions[installation_id]

            previous_game_sessions: pd.DataFrame = game_sessions_for_installation_id[
                game_sessions_for_installation_id['timestamp'] < start_timestamp
                ].copy(deep=True)

            assessment_info = pd.DataFrame({
                'installation_id': installation_id,
                'assessment_game_session': assessment['game_session'],
                'assessment_title': assessment['title'],
                'assessment_world': assessment['world'],
                'assessment_most_common_title_accuracy_group': assessment[
                    'assessment_most_common_title_accuracy_group'],
                'assessment_dayofweek': assessment['assessment_dayofweek'],
                'assessment_hour': assessment['assessment_hour'],
            }, index=[0])

            if previous_game_sessions.shape[0] == 0:
                game_sessions.append(assessment_info)
                continue

            # Previous user attempts for this assessment
            previous_user_attempts_at_assessment_game_sessions = previous_game_sessions[
                previous_game_sessions['title'] == assessment['title']
                ][aggregate_assessment_game_session_columns].values

            if previous_user_attempts_at_assessment_game_sessions.shape[0] > 0:
                previous_user_attempts_at_assessment_game_sessions_agg = _agg_game_sessions(
                    previous_user_attempts_at_assessment_game_sessions, aggregate_assessment_game_session_columns,
                    prefix='previous_user_attempt_at_assessment_'
                )
            else:
                previous_user_attempts_at_assessment_game_sessions_agg = pd.DataFrame()

            # Previous user attempts for games in this world
            previous_user_attempts_at_world_games = previous_game_sessions[
                (previous_game_sessions['world'] == assessment['world']) &
                (previous_game_sessions['type'] == 'Game')
                ][aggregate_game_game_session_columns].values

            if previous_user_attempts_at_world_games.shape[0] > 0:
                previous_user_attempts_at_world_games_agg = _agg_game_sessions(
                    previous_user_attempts_at_world_games, aggregate_game_game_session_columns,
                    prefix='previous_user_attempts_at_world_games_'
                )
            else:
                previous_user_attempts_at_world_games_agg = pd.DataFrame()

            # One-hot-encode categorical features
            previous_game_sessions = pd.get_dummies(
                previous_game_sessions, columns=['title', 'type', 'world'], prefix='ohe')
            ohe_columns = [column for column in previous_game_sessions.columns if column.startswith('ohe')]
            ohe_agg = pd.DataFrame(
                np.nansum(previous_game_sessions[ohe_columns].values, axis=0).reshape(1, -1), columns=ohe_columns)

            previous_game_sessions_values = previous_game_sessions[_aggregate_game_sessions_columns].values
            previous_game_sessions_agg = _agg_game_sessions(
                previous_game_sessions_values, _aggregate_game_sessions_columns)

            df_final_agg = pd.concat(
                (
                    assessment_info,
                    previous_game_sessions_agg,
                    ohe_agg,
                    previous_user_attempts_at_assessment_game_sessions_agg,
                    previous_user_attempts_at_world_games_agg
                ),
                axis=1
            )

            game_sessions.append(df_final_agg)

            # if idx == 10:
            #     break

            if idx % 100 == 0:
                print(f'Row {idx + 1}/{total_assessments} done')

        df_final = pd.concat(game_sessions, ignore_index=True, sort=False).fillna(0.0)
        df_final['user_game_time_mean'] = df_final.groupby('installation_id')['game_time_mean'].transform('mean')
        print('Writing features...')
        df_final.to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    aggregate_game_sessions_columns = [
        'game_time',
        'event_count',
        'game_time_mean_diff',
        'game_time_std_diff',
    ]

    aggregate_assessment_game_session_columns = aggregate_game_sessions_columns + [
        'correct_attempts',
        'uncorrect_attempts',
        'accuracy_rate',
        'accuracy_group',
        'accuracy_group_3',
        'accuracy_group_2',
        'accuracy_group_1',
        'accuracy_group_0',
    ]

    aggregate_game_game_session_columns = aggregate_game_sessions_columns + [
        'correct_attempts',
        'uncorrect_attempts',
        'accuracy_rate',
    ]

    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)
    # Add most common title accuracy group to assessment
    title_to_acc_group_dict = dict(
        df_train_labels.groupby('title')['accuracy_group'].agg(lambda x: x.value_counts().index[0])
    )

    if not os.path.exists(TRAIN_FEATURES_CSV):
        print('Preparing train data...')
        df_train = pd.read_csv('preprocessed-data/train_game_sessions.csv')

        title_to_world = df_train[['title', 'world']].groupby('title').agg('first').apply(list).to_dict()
        # Add assessment world
        df_train_labels['world'] = df_train_labels.apply(
            lambda assessment: title_to_world['world'][assessment['title']],
            axis=1
        )

        df_train_labels['assessment_most_common_title_accuracy_group'] = df_train_labels['title'] \
            .map(title_to_acc_group_dict)

        # Add game session start timestamp to train labels
        game_session_start_timestamps = df_train.groupby(
            ['installation_id', 'game_session']
        ).first()['timestamp'].reset_index()

        df_train_labels = df_train_labels.merge(
            game_session_start_timestamps, on=['installation_id', 'game_session'], how='left'
        )

        _add_time_features(df_train_labels)
        df_train['timestamp'] = (pd.to_datetime(df_train['timestamp']).astype(int) / 10 ** 9).astype(int)

        _compute_features(df_train, 'train', df_train_labels)

        del df_train
        gc.collect()

    print('Preparing test data...')
    df_test = pd.read_csv('preprocessed-data/test_game_sessions.csv')
    df_test_assessments_to_predict = pd.read_csv('preprocessed-data/test_assessments_to_predict.csv')
    df_test_assessments_to_predict['assessment_most_common_title_accuracy_group'] = \
        df_test_assessments_to_predict['title'].map(title_to_acc_group_dict)

    _add_time_features(df_test_assessments_to_predict)
    df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)

    _compute_features(
        df_test,
        'test',
        df_test_assessments_to_predict
    )

    del df_test
    gc.collect()


def label_encode_categorical_features(df_train, df_test=None):
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train[column]
        lb.fit(train_column)
        df_train[column] = lb.transform(train_column)
        df_train[column] = df_train[column].astype('category')

        if df_test is not None:
            test_column = df_test[column]
            df_test[column] = lb.transform(test_column)
            df_test[column] = df_test[column].astype('category')

    return df_train, df_test


def ohe_encode_categorical_features(df_train, df_test=None):
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train[column]
        lb.fit(train_column)
        df_train[column] = lb.transform(train_column)
        df_train = pd.get_dummies(df_train, columns=[column])

        if df_test is not None:
            test_column = df_test[column]
            df_test[column] = lb.transform(test_column)
            df_test = pd.get_dummies(df_test, columns=[column])

    return df_train, df_test


def get_train_test_features():
    columns_to_drop = [
        'installation_id',
        'assessment_game_session',
    ]

    df_train_features = pd.read_csv(TRAIN_FEATURES_CSV)
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')

    df_train_labels = pd.read_csv(
        TRAIN_LABELS_CSV,
        usecols=['installation_id', 'game_session', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['installation_id', 'assessment_game_session'])

    # Extract target and installation ids
    y_correct = df_train_features['num_correct'].values
    y_uncorrect = df_train_features['num_incorrect'].values
    y_accuracy_rate = df_train_features['accuracy'].values
    y_accuracy_group = df_train_features['accuracy_group'].values
    train_installation_ids = df_train_features['installation_id']
    test_installation_ids = df_test_features['installation_id']

    df_train_features = df_train_features.drop(
        columns=columns_to_drop + ['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
    )
    df_train_features_columns = set(df_train_features.columns)
    df_test_features_columns = set(df_test_features.columns)
    columns = df_train_features_columns & df_test_features_columns

    # Make sure test columns are in the correct order
    df_train_features = df_train_features[columns]
    df_test_features = df_test_features.drop(columns=columns_to_drop)[columns]

    return {
        'df_train_features': df_train_features,
        'df_test_features': df_test_features,
        'y_correct': y_correct,
        'y_uncorrect': y_uncorrect,
        'y_accuracy_rate': y_accuracy_rate,
        'y_accuracy_group': y_accuracy_group,
        'train_installation_ids': train_installation_ids,
        'test_installation_ids': test_installation_ids
    }


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_lgbm_classifier(objective, metric, n_estimators=5000):
    def inner(params):
        return LGBMClassifier(
            random_state=2019,
            n_estimators=n_estimators,
            n_jobs=-1,
            objective=objective,
            metric=metric,
            subsample_freq=1,
            subsample=1.0,
            colsample_bytree=1.0,
            **params
        )

    return inner


def get_lgbm_regressor(objective, metric, n_estimators=5000):
    def inner(params):
        return LGBMRegressor(
            random_state=2019,
            n_estimators=n_estimators,
            n_jobs=-1,
            objective=objective,
            metric=metric,
            subsample_freq=1,
            subsample=1.0,
            colsample_bytree=1.0,
            **params
        )

    return inner


def get_catboost_classifier(objective, metric, cat_features, n_estimators=5000):
    def inner(params):
        return CatBoostClassifier(
            random_state=2019,
            loss_function=objective,
            eval_metric=metric,
            n_estimators=n_estimators,
            cat_features=cat_features,
            use_best_model=True,
            colsample_bylevel=1.0,
            **params
        )

    return inner


def get_catboost_regressor(objective, metric, cat_features, n_estimators=5000):
    def inner(params):
        return CatBoostRegressor(
            random_state=2019,
            loss_function=objective,
            eval_metric=metric,
            n_estimators=n_estimators,
            cat_features=cat_features,
            use_best_model=True,
            colsample_bylevel=1.0,
            **params
        )

    return inner


def get_elasticnet_regressor():
    def inner(params):
        return ElasticNet(
            random_state=2019,
            max_iter=10000,
            selection='random',
            **params
        )

    return inner


def get_xgboost_classifier(objective, metric, n_estimators=5000):
    def inner(params):
        return XGBClassifier(
            random_state=2019,
            n_estimators=n_estimators,
            objective=objective,
            eval_metric=metric,
            **params
        )

    return inner


def get_xgboost_regressor(objective, metric, n_estimators=5000):
    def inner(params):
        return XGBRegressor(
            random_state=2019,
            n_estimators=n_estimators,
            objective=objective,
            eval_metric=metric,
            **params
        )

    return inner


def integer_encode_params(params):
    for param, value in params.items():
        if param in INTEGER_PARAMS:
            params[param] = int(round(value))
    return params


def fit_model(model, X_train, y_train, X_val, y_val, early_stopping_rounds=10, verbose=0):
    model.fit(
        X_train, y_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )


def fit_sklearn_model(model, X_train, y_train, X_val, y_val, verbose=0):
    model.fit(
        X_train, y_train
    )


def cv_with_oof_predictions(
        model_fn, score_fn, fit_fn, X, X_test, y, y_accuracy_group, installation_ids, model_params,
        n_splits=10, n_predicted_features=2, predict_proba=True, standard_scale=False
):
    scores = []
    oof_train_predictions = np.zeros((X.shape[0], n_predicted_features))
    test_predictions = np.zeros((X_test.shape[0], n_predicted_features))

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_val = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_val = y[train_split], y[test_split]

        if standard_scale:
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_val = ss.transform(X_val)
            X_test = ss.transform(X_test)

        model = model_fn(model_params)
        fit_fn(model, X_train, y_train, X_val, y_val)

        if predict_proba:
            val_prediction = model.predict_proba(X_val)
            test_prediction = model.predict_proba(X_test)
        else:
            val_prediction = model.predict(X_val).reshape((-1, 1))
            test_prediction = model.predict(X_test).reshape((-1, 1))

        oof_train_predictions[test_split, :] = val_prediction
        test_predictions += (test_prediction / n_splits)

        scores.append(score_fn(y_val, model.predict(X_val)))

    print('CV score:', np.mean(scores))

    return oof_train_predictions, test_predictions


def lgb_fit_predict_correct_attempts_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    meta_train, meta_test = cv_with_oof_predictions(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        df_test,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    return meta_train[:, 1].reshape(-1, 1), meta_test[:, 1].reshape(-1, 1)


def lgb_fit_predict_accuracy_group_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def lgb_fit_predict_accuracy_rate_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def catboost_fit_predict_correct_attempts_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    meta_train, meta_test = cv_with_oof_predictions(
        get_catboost_classifier('Logloss', 'AUC', cat_features),
        roc_auc_score,
        fit_model,
        df_train,
        df_test,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    return meta_train[:, 1].reshape(-1, 1), meta_test[:, 1].reshape(-1, 1)


def catboost_fit_predict_accuracy_group_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def catboost_fit_predict_accuracy_rate_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def elasticnet_fit_predict_accuracy_group_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = ohe_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_elasticnet_regressor(),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_sklearn_model,
        df_train,
        df_test,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1,
        standard_scale=True
    )


def elasticnet_fit_predict_accuracy_rate_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_elasticnet_regressor(),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_sklearn_model,
        df_train,
        df_test,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def xgboost_fit_predict_correct_attempts_model(params):
    features = get_train_test_features()

    df_train, df_test = ohe_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    meta_train, meta_test = cv_with_oof_predictions(
        get_xgboost_classifier('binary:logistic', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        df_test,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    return meta_train[:, 1].reshape(-1, 1), meta_test[:, 1].reshape(-1, 1)


def xgboost_fit_predict_accuracy_group_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = ohe_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def xgboost_fit_predict_accuracy_rate_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = ohe_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        df_test,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


class OptimizedRounder:
    def __init__(self, initial_coef, labels):
        self.coef_ = 0
        self.initial_coef = initial_coef
        self.labels = labels

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=self.labels)
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = scipy.optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=self.labels)
        return preds

    def coefficients(self):
        return self.coef_['x']


def output_submission():
    lgb_models = [
        {
            'name': 'lgb_correct_attempts',
            'fit_predict_fn': lgb_fit_predict_correct_attempts_model,
            'thresholds': [(0.3, 0.5), (0.5, 0.85), (0.85, 0.95)],
            'params': {'learning_rate': 0.48167614730562847,
                       'max_bin': 419.5794922742474,
                       'max_depth': 13.998338323966081,
                       'min_child_samples': 599.3629839842425,
                       'min_child_weight': 250.25027513465395,
                       'num_leaves': 6.802197073843434,
                       'reg_alpha': 3.678603438728225,
                       'reg_lambda': 1.0191385015353065}
        },
        {
            'name': 'lgb_accuracy_group_regression',
            'fit_predict_fn': lgb_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'learning_rate': 0.05212540561823331,
                       'max_bin': 204.7112162055064,
                       'max_depth': 12.01521111509996,
                       'min_child_samples': 50.62820932872383,
                       'min_child_weight': 523.3866794090738,
                       'num_leaves': 309.03594226296747,
                       'reg_alpha': 1.64211663480082,
                       'reg_lambda': 27.791046379299146}
        },
        {
            'name': 'lgb_accuracy_rate_regression',
            'fit_predict_fn': lgb_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'learning_rate': 0.016286068132386897,
                       'max_bin': 123.22441899808396,
                       'max_depth': 11.433462193261546,
                       'min_child_samples': 377.7231525336464,
                       'min_child_weight': 284.94771870804243,
                       'num_leaves': 16.060258947743566,
                       'reg_alpha': 1.945045074540653,
                       'reg_lambda': 27.18254007683529}
        },
    ]

    catboost_models = [
        {
            'name': 'catboost_correct_attempts',
            'fit_predict_fn': catboost_fit_predict_correct_attempts_model,
            'thresholds': [(0.3, 0.5), (0.5, 0.85), (0.85, 0.95)],
            'params': {'l2_leaf_reg': 1.070238795631429,
                       'learning_rate': 0.10077590426786258,
                       'max_depth': 4.9100843756637875}
        },
        {
            'name': 'catboost_accuracy_group_regression',
            'fit_predict_fn': catboost_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'l2_leaf_reg': 4.932974091609228,
                       'learning_rate': 0.10142152091274366,
                       'max_depth': 5.564929440504873}
        },
        {
            'name': 'catboost_accuracy_rate_regression',
            'fit_predict_fn': catboost_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'l2_leaf_reg': 23.11192947324505,
                       'learning_rate': 0.1012324928519919,
                       'max_depth': 7.216560644143432}
        },
    ]

    xgboost_models = [
        {
            'name': 'xgboost_correct_attempts',
            'fit_predict_fn': xgboost_fit_predict_correct_attempts_model,
            'thresholds': [(0.3, 0.5), (0.5, 0.85), (0.85, 0.95)],
            'params': {'learning_rate': 0.42355675717639446,
                       'max_bin': 201.60605657684866,
                       'max_depth': 7.253975720615041,
                       'min_child_samples': 1241.614766407504,
                       'min_child_weight': 213.82939964039417,
                       'reg_alpha': 1.9706368862497017,
                       'reg_lambda': 14.590840603304597}
        },
        {
            'name': 'xgboost_accuracy_group_regression',
            'fit_predict_fn': xgboost_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'learning_rate': 0.12759819624111682,
                       'max_bin': 372.75756638231735,
                       'max_depth': 6.966748383686637,
                       'min_child_samples': 275.205166701255,
                       'min_child_weight': 277.6947060814316,
                       'reg_alpha': 10.858373818774762,
                       'reg_lambda': 4.573895049487007}
        },
        {
            'name': 'xgboost_accuracy_rate_regression',
            'fit_predict_fn': xgboost_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'learning_rate': 0.2704548231125633,
                       'max_bin': 151.72548066572423,
                       'max_depth': 6.571678634234293,
                       'min_child_samples': 1148.0189646018578,
                       'min_child_weight': 45.53782541558896,
                       'reg_alpha': 25.432340908016528,
                       'reg_lambda': 1.7609398148851962}
        },
    ]

    models = xgboost_models + lgb_models + catboost_models

    def output_meta_features():
        base_features = [model_desc['fit_predict_fn'](model_desc['params']) for model_desc in models]

        meta_train_features = np.concatenate([_train for (_train, _) in base_features], axis=1)
        meta_test_features = np.concatenate([_test for (_, _test) in base_features], axis=1)

        train_test_features = get_train_test_features()
        y_accuracy_group = train_test_features['y_accuracy_group']
        train_installation_ids = train_test_features['train_installation_ids']
        test_installation_ids = train_test_features['test_installation_ids']

        columns = [model_desc['name'] for model_desc in models]
        df_meta_train = pd.DataFrame(meta_train_features, columns=columns)
        df_meta_train['target'] = y_accuracy_group
        df_meta_train['installation_ids'] = train_installation_ids

        df_meta_test = pd.DataFrame(meta_test_features, columns=columns)
        df_meta_test['installation_ids'] = test_installation_ids

        df_meta_train.to_csv('preprocessed-data/meta_train_features.csv', index=False)
        df_meta_test.to_csv('preprocessed-data/meta_test_features.csv', index=False)

    def find_best_thresholds(model_desc, feature, target):
        print('Thresholding: ', model_desc['name'])
        best_or = None
        best_score = 0.0

        for _ in range(30):
            thresholds = [np.random.uniform(*init_thresholds) for init_thresholds in model_desc['thresholds']]
            or_ = OptimizedRounder(thresholds, [0, 1, 2, 3])
            or_.fit(feature, target)

            prediction = or_.predict(feature, or_.coefficients())
            score = cohen_kappa_score(prediction, target, weights='quadratic')

            if score > best_score:
                best_score = score
                best_or = or_
                print('New best score: ', score)

        print('Best score: ', best_score)
        return best_or

    def output_thresholded_predictions():
        df_meta_test_features = pd.read_csv('preprocessed-data/meta_test_features.csv')
        df_meta_train_features = pd.read_csv('preprocessed-data/meta_train_features.csv')
        target = df_meta_train_features['target'].values
        test_installation_ids = df_meta_test_features['installation_ids']

        meta_train_thresholded = {}
        meta_test_thresholded = {}

        for model_desc in models:
            train_feature = df_meta_train_features[model_desc['name']]
            or_ = find_best_thresholds(model_desc, train_feature, target)

            train_prediction = or_.predict(train_feature, or_.coefficients())
            test_prediction = or_.predict(df_meta_test_features[model_desc['name']], or_.coefficients())

            meta_train_thresholded[model_desc['name']] = train_prediction.values
            meta_test_thresholded[model_desc['name']] = test_prediction.values

        train_prediction_mean = pd.DataFrame(meta_train_thresholded).values.mean(axis=1)
        test_prediction_mean = pd.DataFrame(meta_test_thresholded).values.mean(axis=1)

        or_ = find_best_thresholds(
            {'name': 'Final', 'thresholds': [(0, 1), (1, 2), (2, 3)]}, train_prediction_mean, target)

        train_prediction = or_.predict(train_prediction_mean, or_.coefficients())
        test_prediction = or_.predict(test_prediction_mean, or_.coefficients())

        score = cohen_kappa_score(train_prediction, target, weights='quadratic')
        print('Train score: ', score)

        pd.DataFrame.from_dict({
            'installation_id': test_installation_ids,
            'accuracy_group': test_prediction
        }).to_csv('submission.csv', index=False)

    output_meta_features()
    output_thresholded_predictions()


categorical_features = [
    'assessment_title',
    'assessment_world',
]

TRAIN_CSV = 'data/train.csv'
TRAIN_LABELS_CSV = 'data/train_labels.csv'
TEST_CSV = 'data/test.csv'
TRAIN_FEATURES_CSV = 'preprocessed-data/train_features.csv'
EVENT_PROPS_JSON = 'event_props.json'
CORRELATED_FEATURES_JSON = 'correlated_features.json'
FEATURE_IMPORTANCES_CSV = 'feature_importances.csv'
