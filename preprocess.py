import os

import pandas as pd


def preprocess_data(df_data, output_path, drop_columns=None):
    is_assessment = df_data['type'] == 'Assessment'

    is_bird_measurer_attempt = (df_data['title'] == 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4110)
    is_non_bird_measurer_attempt = (df_data['title'] != 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4100)
    is_assessment_attempt = is_assessment & (is_bird_measurer_attempt | is_non_bird_measurer_attempt)

    is_correct_attempt = df_data['event_data'].str.contains('"correct":true')
    is_uncorrect_attempt = df_data['event_data'].str.contains('"correct":false')

    df_data['is_correct_attempt'] = (is_assessment_attempt & is_correct_attempt).astype(int)
    df_data['is_uncorrect_attempt'] = (is_assessment_attempt & is_uncorrect_attempt).astype(int)

    if drop_columns:
        df_data = df_data.drop(columns=drop_columns)

    df_data.to_csv(output_path, index=False)


os.makedirs('preprocessed-data', exist_ok=True)

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_train_labels = pd.read_csv('data/train_labels.csv')

# Only keep installation ids from train labels
installation_ids = df_train_labels['installation_id'].unique()
df_train = df_train[df_train['installation_id'].isin(installation_ids)]

print('Preprocessing train...')
preprocess_data(df_train, 'preprocessed-data/train.csv', ['event_data'])
print('Preprocessing test...')
preprocess_data(df_test, 'preprocessed-data/test.csv', ['event_data'])
