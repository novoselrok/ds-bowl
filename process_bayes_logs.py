import json


def process_bayes_logs(path):
    with open(path, encoding='utf-8') as f:
        logs = [json.loads(line) for line in f.readlines()]

    logs = list(sorted(logs, key=lambda log: log['target'], reverse=True))
    print(logs)

    return logs
