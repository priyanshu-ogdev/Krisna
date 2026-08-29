import urllib.request
import json

datasets = ['conceptual_12m', 'bevaya/RICO-Screen2Words']
for ds in datasets:
    try:
        url = f'https://datasets-server.huggingface.co/info?dataset={ds}'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            print(f'=== {ds} ===')
            for config, info in data.get('dataset_info', {}).items():
                features = list(info.get('features', {}).keys())
                print(f'Config: {config}, Features: {features}')
    except Exception as e:
        print(f'Error fetching {ds}: {e}')
