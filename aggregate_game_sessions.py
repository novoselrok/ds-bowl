import csv
import json
from collections import defaultdict

import pandas as pd

from submission import EVENT_PROPS_JSON, TRAIN_LABELS_CSV, TEST_CSV, TRAIN_CSV

with open(EVENT_PROPS_JSON, encoding='utf-8') as f:
    event_ids_to_props = json.load(f)

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'


def postprocessing(game_sessions):
    for idx in range(len(game_sessions)):
        event_codes = game_sessions[idx]['event_codes']
        event_data_props = game_sessions[idx]['event_data_props']

        del game_sessions[idx]['event_codes']
        del game_sessions[idx]['event_data_props']

        correct, uncorrect = game_sessions[idx]['correct_attempts'], game_sessions[idx]['uncorrect_attempts']
        if correct == 0 and uncorrect == 0:
            game_sessions[idx]['accuracy_rate'] = 0.0
        else:
            game_sessions[idx]['accuracy_rate'] = correct / float(correct + uncorrect)

        if correct == 0 and uncorrect == 0:
            pass
        elif correct == 1 and uncorrect == 0:
            game_sessions[idx]['accuracy_group_3'] = 1
        elif correct == 1 and uncorrect == 1:
            game_sessions[idx]['accuracy_group_2'] = 1
        elif correct == 1 and uncorrect >= 2:
            game_sessions[idx]['accuracy_group_1'] = 1
        else:
            game_sessions[idx]['accuracy_group_0'] = 1

        game_sessions[idx] = {
            **game_sessions[idx],
            **dict(event_codes),
            **dict(event_data_props)
        }

    return game_sessions


def aggregate_game_sessions(csv_path, prefix, installation_ids_to_keep=None):
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
                    'game_time': 0,
                    'correct_attempts': 0,
                    'uncorrect_attempts': 0,
                    'title': row['title'],
                    'type': row['type'],
                    'world': row['world'],
                    'event_codes': defaultdict(int),
                    'event_data_props': defaultdict(int)
                }

            game_session = game_sessions[row['game_session']]

            game_session['event_count'] = max(game_session['event_count'], int(row['event_count']))
            game_session['game_time'] = max(game_session['game_time'], int(row['game_time']))

            event_data = json.loads(row['event_data'])
            event_code = int(row['event_code'])

            game_session['event_codes'][f'cnt_event_code_{event_code}'] += 1

            is_assessment = row['type'] == 'Assessment'
            if is_assessment:
                is_bird_measurer_attempt = row['title'] == BIRD_MEASURER_ASSESSMENT and event_code == 4110
                is_non_bird_measurer_attempt = row['title'] != BIRD_MEASURER_ASSESSMENT and event_code == 4100
                is_assessment_attempt = is_bird_measurer_attempt or is_non_bird_measurer_attempt

                if is_assessment_attempt:
                    is_correct_attempt = 'correct' in event_data and event_data['correct']
                    is_uncorrect_attempt = 'correct' in event_data and not event_data['correct']

                    if is_correct_attempt and is_assessment_attempt:
                        game_session['correct_attempts'] += 1

                    if is_uncorrect_attempt and is_assessment_attempt:
                        game_session['uncorrect_attempts'] += 1

            event_id = row['event_id']
            if event_id in event_ids_to_props:
                for prop in event_ids_to_props[event_id]:
                    game_session['event_data_props'][f'event_data_prop_{event_id}_{prop}'] += event_data[prop]

            if idx % 10000 == 0:
                print(idx)

    df_data = pd.DataFrame(postprocessing(list(game_sessions.values())))
    df_data.fillna(0.0, inplace=True)
    df_data.to_csv(f'preprocessed-data/{prefix}_game_sessions.csv', index=False)


df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)

# Only keep installation ids from train labels
aggregate_game_sessions(
    TRAIN_CSV,
    'train',
    installation_ids_to_keep=set(df_train_labels['installation_id'].unique())
)

aggregate_game_sessions(
    TEST_CSV,
    'test'
)

pd.read_csv(TEST_CSV) \
    .groupby(['installation_id']) \
    .last() \
    .reset_index() \
    .to_csv('preprocessed-data/test_assessments_to_predict.csv', index=False)
