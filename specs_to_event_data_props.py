import json

import pandas as pd

df_specs = pd.read_csv('data/specs.csv')

args_to_extract = {'duration', 'total_duration', 'misses'}
event_ids_to_props = {}
for _, spec in df_specs.iterrows():
    event_id = spec['event_id']

    info = spec['info']
    args_json = json.loads(spec['args'])
    args_names = set([arg['name'] for arg in args_json])

    present_args = args_to_extract & args_names
    event_ids_to_props[event_id] = list(present_args)

with open('event_props.json', 'w', encoding='utf-8') as f:
    json.dump(event_ids_to_props, f)
