import os
import json
import glob
import numpy as np
import pandas as pd

def load_data():
    records = []
    
    # 1. Main experiments/results/*.json
    files = glob.glob('experiments/results/*.json')
    for f in files:
        if 'summary_' in os.path.basename(f):
            continue
        with open(f, 'r') as fp:
            data = json.load(fp)
            data['dataset_source'] = 'experiments_results'
            records.append(data)

    # 2. experiments/results/opt1_cqi/*.json
    opt1_files = glob.glob('experiments/results/opt1_cqi/*.json')
    for f in opt1_files:
        if os.path.basename(f) in ['aggregate.json', 'all_results.json', 'a5_results.json']:
            continue
        with open(f, 'r') as fp:
            data = json.load(fp)
            data['dataset_source'] = 'opt1_cqi'
            records.append(data)
            
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    df = load_data()
    print(f"Loaded total {len(df)} run records.")
    print("Columns:", list(df.columns))
    print("\nCounts by dataset source & config:")
    print(df.groupby(['dataset_source', 'config', 'scenario', 'profile'])['seed'].count())
    
    # Check numeric columns summary
    num_cols = ['success_rate', 'tokens', 'api_calls', 'computation_s', 'cfr', 'tfr', 
                'switch_count', 'peer_messages', 'broadcast_count', 'consensus_rounds', 
                'cloud_planning_calls', 'device_planning_calls', 'avg_planning_latency', 'memory_mb']
    
    summary = df.groupby(['config', 'scenario', 'profile'])[num_cols].agg(['mean', 'std'])
    print("\n--- Summary Statistics ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary)
