import json

import pandas as pd

df_specs = pd.read_csv('data/specs.csv')

args_to_extract = {'duration', 'total_duration', 'misses'}
event_ids = {'2dcad279', 'a8a78786', '67439901', '28520915', 'ca11f653', '84538528', 'a1e4395d', '36fa3ebe', 'c74f40cd',
             '4a09ace1', 'e694a35b', '828e68f9', '022b4259', '5c2f29ca', 'fcfdffb6', '1bb5fbdb', 'b2dba42b', 'beb0a7b9',
             'd45ed6a1', '5f0eb72c', 'd185d3ea', 'bdf49a58', '923afab1', 'c58186bf', '30614231', '3bf1cf26', 'e9c52111',
             '49ed92e9', '562cec5f'}
event_ids_to_props = {}
for _, spec in df_specs.iterrows():
    event_id = spec['event_id']

    if event_id not in event_ids:
        event_ids_to_props[event_id] = []
        continue

    info = spec['info']
    args_json = json.loads(spec['args'])
    args_names = set([arg['name'] for arg in args_json])

    present_args = args_to_extract & args_names
    event_ids_to_props[event_id] = list(present_args)

with open('event_props.json', 'w', encoding='utf-8') as f:
    json.dump(event_ids_to_props, f)
