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
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'
CAULDRON_FILLER_ASSESSMENT = 'Cauldron Filler (Assessment)'
ASSESSMENTS = ['Mushroom Sorter (Assessment)', 'Bird Measurer (Assessment)',
               'Cauldron Filler (Assessment)', 'Cart Balancer (Assessment)',
               'Chest Sorter (Assessment)']
INTEGER_PARAMS = ['max_depth', 'max_bin', 'num_leaves',
                  'min_child_samples', 'n_splits', 'subsample_freq']

np.random.seed(0)
# Except 2000
event_codes = [3010, 3110, 4070, 4090, 4030, 4035, 4021, 4020, 4010, 2080,
               2083, 2040, 2020, 2030, 3021, 3121, 2050, 3020, 3120, 2060, 2070,
               4031, 4025, 5000, 5010, 2081, 2025, 4022, 2035, 4040, 4100, 2010,
               4110, 4045, 4095, 4220, 2075, 4230, 4235, 4080, 4050]


def preprocess_events():
    def _postprocessing(_game_sessions):
        game_sessions = []
        for _game_session in _game_sessions:
            title = _game_session['title']
            type_ = _game_session['type']

            game_session = {
                'game_session': _game_session['game_session'],
                'timestamp': _game_session['timestamp'],
                'installation_id': _game_session['installation_id'],
                'title': _game_session['title'],
                'type': _game_session['type'],
                'world': _game_session['world'],
            }

            if type_ != 'Clip':
                game_session = {
                    **game_session,
                    **_game_session['event_codes'],
                    **dict(_game_session['event_ids']),
                    f'game_time': _game_session['game_time'],
                }

                if type_ != 'Activity':
                    has_durations = len(_game_session['durations']) > 0
                    game_session[f'mean_round_durations'] = np.mean(_game_session['durations']) \
                        if has_durations else np.nan
                    game_session[f'sum_round_durations'] = np.sum(_game_session['durations']) \
                        if has_durations else np.nan
            else:
                # Clips don't have any information beyond counting them
                game_session[f'watched_{title}'] = 1

            if type_ == 'Assessment':
                correct = _game_session['correct_attempts']
                uncorrect = _game_session['uncorrect_attempts']

                game_session[f'correct_attempts'] = correct
                game_session[f'uncorrect_attempts'] = uncorrect

                if correct == 0 and uncorrect == 0:
                    game_session[f'accuracy_rate'] = np.nan
                else:
                    game_session[f'accuracy_rate'] = correct / float(correct + uncorrect)

                if correct == 0 and uncorrect == 0:
                    game_session[f'accuracy_group'] = np.nan
                elif correct == 1 and uncorrect == 0:
                    game_session[f'accuracy_group'] = 3
                elif correct == 1 and uncorrect == 1:
                    game_session[f'accuracy_group'] = 2
                elif correct == 1 and uncorrect >= 2:
                    game_session[f'accuracy_group'] = 1
                else:
                    game_session[f'accuracy_group'] = 0

                game_session[f'did_not_finish'] = 1 if correct == 0 and uncorrect == 0 else 0

                if title == BIRD_MEASURER_ASSESSMENT:
                    has_stage_numbers = len(_game_session['stage_numbers']) > 0
                    game_session[f'max_stage_number'] = np.max(_game_session['stage_numbers']) \
                        if has_stage_numbers else 0

                if title == CAULDRON_FILLER_ASSESSMENT:
                    has_round_numbers = len(_game_session['round_numbers']) > 0
                    game_session[f'max_round_number'] = np.max(_game_session['round_numbers']) \
                        if has_round_numbers else 0

            elif type_ == 'Game':
                has_misses = len(_game_session['misses']) > 0
                game_session[f'mean_round_misses'] = np.mean(_game_session['misses']) if has_misses else np.nan
                game_session[f'sum_round_misses'] = np.sum(_game_session['misses']) if has_misses else np.nan

                has_rounds = len(_game_session['rounds']) > 0
                game_session[f'max_round'] = np.max(_game_session['rounds']) if has_rounds else 0

            game_sessions.append(game_session)

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
                        'title': row['title'],
                        'type': row['type'],
                        'world': row['world'],

                        'game_time': 0,
                        'correct_attempts': 0,
                        'uncorrect_attempts': 0,
                        'durations': [],
                        'misses': [],
                        'rounds': [],
                        'stage_numbers': [],
                        'round_numbers': [],
                        'event_codes': {f'event_code_{event_code}': 0 for event_code in event_codes},
                        'event_ids': defaultdict(int)
                    }

                game_session = game_sessions[row['game_session']]
                game_session['game_time'] = max(int(row['game_time']), game_session['game_time'])

                event_data = json.loads(row['event_data'])
                event_code = int(row['event_code'])
                event_id = row['event_id']

                if event_code in event_codes:
                    game_session['event_codes'][f'event_code_{event_code}'] += 1

                game_session['event_ids'][f'event_id_{event_id}'] += 1

                if row['type'] == 'Assessment':
                    is_bird_measurer_attempt = row['title'] == BIRD_MEASURER_ASSESSMENT and event_code == 4110
                    is_non_bird_measurer_attempt = row['title'] != BIRD_MEASURER_ASSESSMENT and event_code == 4100
                    is_assessment_attempt = is_bird_measurer_attempt or is_non_bird_measurer_attempt

                    if is_assessment_attempt and 'correct' in event_data:
                        if event_data['correct']:
                            game_session['correct_attempts'] += 1
                        else:
                            game_session['uncorrect_attempts'] += 1

                # End-of-round event
                if 'misses' in event_data and event_code == 2030:
                    game_session['durations'].append(event_data['duration'])

                    if game_session['type'] == 'Game':
                        game_session['misses'].append(event_data['misses'])
                        game_session['rounds'].append(event_data['round'])

                    elif game_session['type'] == 'Assessment':
                        if 'stage_number' in event_data:
                            game_session['stage_numbers'].append(event_data['stage_number'])

                        if 'round_number' in event_data:
                            game_session['round_numbers'].append(event_data['round_number'])

                    else:
                        raise Exception(f'Invalid game session type: {game_session["type"]}')

                if idx % 10000 == 0:
                    print(idx)

        df_data = pd.DataFrame(_postprocessing(list(game_sessions.values())))
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
    def _add_time_features(df):
        timestamp = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
        df['assessment_hour'] = timestamp.dt.hour
        df['assessment_dayofweek'] = timestamp.dt.dayofweek

    def _agg_game_sessions(values, columns, aggfns, prefix=''):
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

        df_assessments = df_assessments.sort_values(by='installation_id')
        installation_id_to_game_sessions = dict(tuple(df_data.groupby('installation_id')))

        clip_columns = [column for column in df_data.columns if column.startswith('watched_')]
        assessment_columns = ['game_time', 'mean_round_durations', 'sum_round_durations', 'correct_attempts',
                              'uncorrect_attempts', 'accuracy_rate', 'accuracy_group', 'did_not_finish',
                              'max_stage_number', 'max_round_number']
        game_columns = ['game_time', 'mean_round_durations', 'sum_round_durations', 'mean_round_misses',
                        'sum_round_misses', 'max_round']

        event_code_columns = [column for column in df_data.columns if column.startswith('event_code_')]
        event_id_columns = [column for column in df_data.columns if column.startswith('event_id_')]

        total_assessments = df_assessments.shape[0]
        for idx, assessment in df_assessments.iterrows():
            installation_id = assessment['installation_id']
            start_timestamp = assessment['timestamp']

            game_sessions_for_installation_id = installation_id_to_game_sessions[installation_id]

            previous_game_sessions: pd.DataFrame = game_sessions_for_installation_id[
                game_sessions_for_installation_id['timestamp'] < start_timestamp].copy(deep=True)

            assessment_info = pd.DataFrame({
                'installation_id': installation_id,
                'assessment_game_session': assessment['game_session'],
                'assessment_title': assessment['title'],
                'assessment_world': assessment['world'],
                'assessment_dayofweek': assessment['assessment_dayofweek'],
                'assessment_hour': assessment['assessment_hour'],
            }, index=[0])

            if previous_game_sessions.shape[0] == 0:
                game_sessions.append(assessment_info)
                continue

            df_clips_agg = _agg_game_sessions(
                previous_game_sessions[clip_columns].values, clip_columns, {'sum': np.nansum})

            previous_assessment_attempts = previous_game_sessions[
                previous_game_sessions['title'] == assessment['title']]

            if previous_assessment_attempts.shape[0] > 0:
                df_previous_assessment_agg = _agg_game_sessions(
                    previous_assessment_attempts[assessment_columns].values,
                    assessment_columns,
                    {'std': np.nanstd, 'mean': np.nanmean, 'median': np.nanmedian, 'max': np.nanmax, 'sum': np.nansum},
                    prefix='previous_assessment_attempt_'
                )
            else:
                df_previous_assessment_agg = pd.DataFrame()

            previous_relevant_game_attempts = previous_game_sessions[
                (previous_game_sessions['world'] == assessment['world']) &
                (previous_game_sessions['type'] == 'Game')]

            if previous_relevant_game_attempts.shape[0] > 0:
                df_previous_relevant_game_agg = _agg_game_sessions(
                    previous_relevant_game_attempts[game_columns].values,
                    game_columns,
                    {'std': np.nanstd, 'mean': np.nanmean, 'median': np.nanmedian, 'max': np.nanmax, 'sum': np.nansum},
                    prefix='previous_relevant_games_attempt_'
                )
            else:
                df_previous_relevant_game_agg = pd.DataFrame()

            df_event_codes_agg = _agg_game_sessions(
                previous_game_sessions[event_code_columns].values,
                event_code_columns,
                {'std': np.nanstd, 'mean': np.nanmean, 'sum': np.nansum},
            )

            df_event_ids_agg = _agg_game_sessions(
                previous_game_sessions[event_id_columns].values,
                event_id_columns,
                {'std': np.nanstd, 'mean': np.nanmean, 'sum': np.nansum},
            )

            df_final_agg = pd.concat(
                (
                    assessment_info,
                    df_clips_agg,
                    df_previous_assessment_agg,
                    df_previous_relevant_game_agg,
                    df_event_codes_agg,
                    df_event_ids_agg
                ),
                axis=1
            )

            game_sessions.append(df_final_agg)

            # if idx == 10:
            #     break

            if idx % 100 == 0:
                print(f'Row {idx + 1}/{total_assessments} done')

        df_final = pd.concat(game_sessions, ignore_index=True, sort=False)
        print('Writing features...')
        df_final.to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)

    if not os.path.exists(TRAIN_FEATURES_CSV):
        print('Preparing train data...')
        df_train = pd.read_csv('preprocessed-data/train_game_sessions.csv')

        title_to_world = df_train[['title', 'world']].groupby('title').agg('first').apply(list).to_dict()
        # Add assessment world to train_labels
        df_train_labels['world'] = df_train_labels.apply(
            lambda assessment: title_to_world['world'][assessment['title']],
            axis=1
        )

        df_train_labels = df_train_labels.merge(
            df_train[['installation_id', 'game_session', 'timestamp']],
            on=['installation_id', 'game_session'], how='left'
        )

        _add_time_features(df_train_labels)
        df_train['timestamp'] = (pd.to_datetime(df_train['timestamp']).astype(int) / 10 ** 9).astype(int)

        _compute_features(df_train, 'train', df_train_labels)

        del df_train
        gc.collect()

    print('Preparing test data...')
    df_test = pd.read_csv('preprocessed-data/test_game_sessions.csv')
    df_test_assessments_to_predict = pd.read_csv('preprocessed-data/test_assessments_to_predict.csv')

    _add_time_features(df_test_assessments_to_predict)
    df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)

    _compute_features(
        df_test,
        'test',
        df_test_assessments_to_predict
    )

    del df_test
    gc.collect()


def identify_zero_importance_features():
    features = get_train_test_features(feature_removal=False)

    X, _ = label_encode_categorical_features(features['df_train_features'])

    y_accuracy_group = y = features['y_correct']
    installation_ids = features['train_installation_ids']

    feature_importances = np.zeros(X.shape[1])
    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, 10, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_val = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_val = y[train_split], y[test_split]

        model = get_lgbm_regressor('regression', 'rmse')({})
        fit_model(model, X_train, y_train, X_val, y_val)

        feature_importances += model.feature_importances_ / 10

    feature_importances = pd.DataFrame({'feature': list(X.columns),
                                        'importance': feature_importances})

    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

    return zero_features


def feature_selection():
    df_train_features = pd.read_csv(TRAIN_FEATURES_CSV)

    df_nan_columns = (df_train_features.isna().sum() / df_train_features.shape[0])
    nan_columns = list(df_nan_columns[df_nan_columns > 0.9].index)

    print('NaN columns: ', len(nan_columns))

    non_numerical_columns = ['installation_id', 'assessment_game_session', 'assessment_title', 'assessment_world']
    numerical_columns = [column for column in df_train_features.columns if
                         column not in non_numerical_columns]

    # Threshold for removing correlated variables
    threshold = 0.9
    # Absolute value correlation matrix
    corr_matrix = df_train_features[numerical_columns].corr().abs()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Select columns with correlations above threshold
    correlated_columns = [column for column in upper.columns if any(upper[column] > threshold)]

    print('Correlated columns: ', len(correlated_columns))

    zero_importance_columns = identify_zero_importance_features()

    print('Zero importance columns: ', len(zero_importance_columns))

    to_remove_columns = list(set(nan_columns).union(set(correlated_columns).union(set(zero_importance_columns))))

    print('To remove columns: ', len(to_remove_columns))

    pd.DataFrame({'to_remove': to_remove_columns}).to_csv('to_remove_columns.csv', index=False)


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


def get_train_test_features(feature_removal=True):
    columns_to_drop = [
        'installation_id',
        'assessment_game_session',
    ]

    df_train_features = pd.read_csv(TRAIN_FEATURES_CSV)
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')

    if feature_removal:
        to_remove_columns = list(pd.read_csv(TO_REMOVE_COLUMNS_CSV)['to_remove'])
        df_train_features = df_train_features.drop(columns=to_remove_columns)

    df_previous_assessment_counts = pd.read_csv(
        PREVIOUS_ASSESSMENT_COUNTS_CSV).rename({'game_session': 'assessment_game_session'}, axis=1)

    df_train_labels = pd.read_csv(
        TRAIN_LABELS_CSV,
        usecols=['installation_id', 'game_session', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['assessment_game_session', 'installation_id'])
    df_train_features = df_train_features.merge(df_previous_assessment_counts,
                                                on=['assessment_game_session', 'installation_id'])

    # Extract target and installation ids
    y_correct = df_train_features['num_correct'].values
    y_uncorrect = df_train_features['num_incorrect'].values
    y_accuracy_rate = df_train_features['accuracy'].values
    y_accuracy_group = df_train_features['accuracy_group'].values
    train_installation_ids = df_train_features['installation_id']
    previous_assessment_counts = df_train_features['cnt']
    test_installation_ids = df_test_features['installation_id']

    df_train_features = df_train_features.drop(
        columns=columns_to_drop + ['cnt', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
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
        'test_installation_ids': test_installation_ids,
        'previous_assessment_counts': previous_assessment_counts
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


def get_xgboost_classifier(objective, metric, n_estimators=5000):
    def inner(params):
        return XGBClassifier(
            random_state=2019,
            n_estimators=n_estimators,
            objective=objective,
            eval_metric=metric,
            n_jobs=-1,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
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
            n_jobs=-1,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            **params
        )

    return inner


def integer_encode_params(params):
    for param, value in params.items():
        if param in INTEGER_PARAMS:
            params[param] = int(round(value))
    return params


def fit_model(model, X_train, y_train, X_val, y_val, early_stopping_rounds=100, verbose=0):
    model.fit(
        X_train, y_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
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
            'name': 'lgb_accuracy_group_regression',
            'fit_predict_fn': lgb_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'learning_rate': 0.046295660978342965,
                       'max_bin': 364.7659132601798,
                       'max_depth': 8.086012620294996,
                       'min_child_samples': 155.7444857253162,
                       'min_child_weight': 21.471762086956247,
                       'num_leaves': 28.34715424520298,
                       'reg_alpha': 3.7494861513748754,
                       'reg_lambda': 2.927302840794126}
        },
        {
            'name': 'lgb_accuracy_rate_regression',
            'fit_predict_fn': lgb_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'learning_rate': 0.08161491228629601,
                       'max_bin': 678.1459459195172,
                       'max_depth': 3.0877218697323094,
                       'min_child_samples': 272.53308048359884,
                       'min_child_weight': 108.02435073923876,
                       'num_leaves': 474.87329601720654,
                       'reg_alpha': 1.4509589486220889,
                       'reg_lambda': 14.281772877520885}
        },
    ]

    catboost_models = [
        {
            'name': 'catboost_accuracy_group_regression',
            'fit_predict_fn': catboost_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'l2_leaf_reg': 12.338879231664864,
                       'learning_rate': 0.10145245820476756,
                       'max_depth': 6.348050329120875}
        },
        {
            'name': 'catboost_accuracy_rate_regression',
            'fit_predict_fn': catboost_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'l2_leaf_reg': 99.83613832920992,
                       'learning_rate': 0.8130625751595671,
                       'max_depth': 5.7693045135515435}
        },
    ]

    xgboost_models = [
        {
            'name': 'xgboost_accuracy_group_regression',
            'fit_predict_fn': xgboost_fit_predict_accuracy_group_regression_model,
            'thresholds': [(0, 1), (1, 2), (2, 3)],
            'params': {'learning_rate': 0.3981549175973284,
                       'max_bin': 77.84261933504594,
                       'max_depth': 6.062177045768829,
                       'min_child_samples': 1274.2428221907223,
                       'min_child_weight': 76.05499660171316,
                       'reg_alpha': 83.07716602429112,
                       'reg_lambda': 26.251291204740134}
        },
        {
            'name': 'xgboost_accuracy_rate_regression',
            'fit_predict_fn': xgboost_fit_predict_accuracy_rate_regression_model,
            'thresholds': [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)],
            'params': {'learning_rate': 0.25467290260085323,
                       'max_bin': 498.42457220114846,
                       'max_depth': 8.094539153478776,
                       'min_child_samples': 990.9274333833525,
                       'min_child_weight': 957.6336743590531,
                       'reg_alpha': 14.634083423029397,
                       'reg_lambda': 98.67041984693017}
        },
    ]

    models = lgb_models + catboost_models + xgboost_models

    def output_meta_features():
        base_features = [model_desc['fit_predict_fn'](model_desc['params']) for model_desc in models]

        meta_train_features = np.concatenate([_train for (_train, _) in base_features], axis=1)
        meta_test_features = np.concatenate([_test for (_, _test) in base_features], axis=1)

        train_test_features = get_train_test_features()
        y_accuracy_group = train_test_features['y_accuracy_group']
        train_installation_ids = train_test_features['train_installation_ids']
        test_installation_ids = train_test_features['test_installation_ids']
        previous_assessment_counts = train_test_features['previous_assessment_counts']

        columns = [model_desc['name'] for model_desc in models]
        df_meta_train = pd.DataFrame(meta_train_features, columns=columns)
        df_meta_train['target'] = y_accuracy_group
        df_meta_train['installation_ids'] = train_installation_ids
        df_meta_train['previous_assessment_counts'] = previous_assessment_counts

        df_meta_test = pd.DataFrame(meta_test_features, columns=columns)
        df_meta_test['installation_ids'] = test_installation_ids

        df_meta_train.to_csv('preprocessed-data/meta_train_features.csv', index=False)
        df_meta_test.to_csv('preprocessed-data/meta_test_features.csv', index=False)

    def classify(x, bound):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    def output_thresholded_predictions():
        df_meta_test_features = pd.read_csv('preprocessed-data/meta_test_features.csv')
        df_meta_train_features = pd.read_csv('preprocessed-data/meta_train_features.csv')
        target = df_meta_train_features['target'].values
        test_installation_ids = df_meta_test_features['installation_ids']

        meta_train_thresholded = {}
        meta_test_thresholded = {}

        dist = Counter(target)
        for k in dist:
            dist[k] /= len(target)

        for model_desc in models:
            train_feature = df_meta_train_features[model_desc['name']]

            acc = 0
            bounds = {}
            for i in range(3):
                acc += dist[i]
                bounds[i] = np.percentile(train_feature, acc * 100)

            meta_train_thresholded[model_desc['name']] = np.array(
                list(map(lambda x: classify(x, bounds), train_feature)))
            meta_test_thresholded[model_desc['name']] = np.array(
                list(map(lambda x: classify(x, bounds), df_meta_test_features[model_desc['name']])))

            print(model_desc['name'],
                  cohen_kappa_score(meta_train_thresholded[model_desc['name']], target, weights='quadratic'))

        train_prediction_mean = pd.DataFrame(meta_train_thresholded).values.mean(axis=1)
        test_prediction_mean = pd.DataFrame(meta_test_thresholded).values.mean(axis=1)

        acc = 0
        bounds = {}
        for i in range(3):
            acc += dist[i]
            bounds[i] = np.percentile(train_prediction_mean, acc * 100)

        train_prediction = np.array(
            list(map(lambda x: classify(x, bounds), train_prediction_mean)))
        test_prediction = np.array(
            list(map(lambda x: classify(x, bounds), test_prediction_mean)))

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
TO_REMOVE_COLUMNS_CSV = 'to_remove_columns.csv'
PREVIOUS_ASSESSMENT_COUNTS_CSV = 'previous_assessment_counts.csv'
