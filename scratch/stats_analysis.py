import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

def analyze_statistics():
    files = glob.glob('experiments/results/*.json')
    records = []
    for f in files:
        if 'summary_' in os.path.basename(f):
            continue
        with open(f, 'r') as fp:
            records.append(json.load(fp))
    df = pd.DataFrame(records)
    
    metrics = ['success_rate', 'tokens', 'api_calls', 'computation_s', 'cfr', 'switch_count', 'peer_messages', 'device_planning_calls', 'cloud_planning_calls', 'avg_planning_latency']
    
    print("=== STATISTICAL ANALYSIS (Mean ± Std, 95% CI) ===")
    for (scenario, profile), group in df.groupby(['scenario', 'profile']):
        print(f"\n--- Scenario: {scenario.upper()} | Profile: {profile.upper()} ---")
        a5 = group[group['config'] == 'A5']
        b1 = group[group['config'] == 'B1']
        
        for m in metrics:
            a5_vals = a5[m].values
            b1_vals = b1[m].values
            
            a5_mean, a5_std = np.mean(a5_vals), np.std(a5_vals, ddof=1)
            b1_mean, b1_std = np.mean(b1_vals), np.std(b1_vals, ddof=1)
            
            # 95% CI (using t-distribution)
            n_a5, n_b1 = len(a5_vals), len(b1_vals)
            a5_ci = stats.t.interval(0.95, df=n_a5-1, loc=a5_mean, scale=stats.sem(a5_vals)) if n_a5 > 1 and a5_std > 0 else (a5_mean, a5_mean)
            b1_ci = stats.t.interval(0.95, df=n_b1-1, loc=b1_mean, scale=stats.sem(b1_vals)) if n_b1 > 1 and b1_std > 0 else (b1_mean, b1_mean)
            
            # Mann-Whitney U test / t-test
            try:
                stat_mwu, p_mwu = stats.mannwhitneyu(a5_vals, b1_vals)
            except Exception:
                p_mwu = 1.0
                
            try:
                stat_tt, p_tt = stats.ttest_ind(a5_vals, b1_vals)
            except Exception:
                p_tt = 1.0
                
            print(f"Metric: {m:22s} | A5: {a5_mean:8.2f} ± {a5_std:6.2f} (95% CI: [{a5_ci[0]:.2f}, {a5_ci[1]:.2f}]) | B1: {b1_mean:8.2f} ± {b1_std:6.2f} (95% CI: [{b1_ci[0]:.2f}, {b1_ci[1]:.2f}]) | MWU p: {p_mwu:.4f} | t-test p: {p_tt:.4f}")

if __name__ == '__main__':
    import os
    analyze_statistics()
