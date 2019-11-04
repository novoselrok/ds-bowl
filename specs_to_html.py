import pandas as pd
import json

df_specs = pd.read_csv('data/specs.csv')

template = """<html>
<head>
  <style>
    table {{
      font-family: arial, sans-serif;
      border-collapse: collapse;
      width: 100%;
    }}
    
    td, th {{
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }}
    
    tr:nth-child(even) {{
      background-color: #cccccc;
    }}
  </style>
</head>
<body>
 <table style="width:100%">
  <tr>
    <th>ID</th>
    <th>Info</th>
    <th>Args</th>
  </tr>
  {}
</table> 
</body>
</html>
"""

# df_specs['info'] = df_specs['info'].str.lower().replace('[^a-zA-Z0-9]', '', regex=True)
#
# df_specs = df_specs.groupby('info')

args_to_extract = {'duration', 'total_duration', 'dwell_time', 'misses'}
event_ids_to_props = {}
rows = []
for _, spec in df_specs.iterrows():
    event_id = spec['event_id']
    info = spec['info']
    args_json = json.loads(spec['args'])
    args_names = set([arg['name'] for arg in args_json])

    present_args = args_to_extract & args_names
    print(present_args)
    if len(present_args) > 0:
        event_ids_to_props[event_id] = list(present_args)

    skip = ['game_time', 'event_count', 'event_code']
    args_html = ''.join([
        f"<li>Name: <strong>{arg['name']}</strong>, {arg['info']}</li>"
        for arg in args_json
        if arg['name'] not in skip
    ])

    rows.append(f"""
        <tr>
            <td>{event_id}</td>
            <td>{info}</td>
            <td>
                <ul>{args_html}</ul>
            </td>
        </tr>
    """)

with open('event_props.json', 'w', encoding='utf-8') as f:
    json.dump(event_ids_to_props, f)


with open('specs.html', 'w', encoding='utf-8') as f:
    f.write(template.format('\n'.join(rows)))
