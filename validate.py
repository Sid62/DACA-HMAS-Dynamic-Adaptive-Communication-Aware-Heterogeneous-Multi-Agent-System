import sys,os,io,json,contextlib
sys.path.insert(0,'.')
from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
K=['success_rate','cloud_planning_calls','cloud_network_calls','cloud_total_tokens',
   'computation_s','total_wall_clock_s','paper_communication_steps','communication_steps',
   'switch_count','replanning_count','cache_hits','steps','planning_time_s','consensus_time_s']
out={}
for sc in ['logistics','inspection','search_rescue']:
    for seed in [1,2,3,4,5]:
        if os.path.exists('experience_store.json'):
            try: os.remove('experience_store.json')
            except Exception: pass
        o=DACAOrchestrator(scenario=sc,network_profile='oscillatory',seed=seed,
                           config=CONFIGS['A5'],max_steps=200)
        with contextlib.redirect_stdout(io.StringIO()): m=o.run()
        d=m.to_dict(); out[f'{sc}_s{seed}']={k:d.get(k) for k in K}
        print(f"{sc[:9]:9s} s{seed} succ={d['success_rate']:6.2f} cloud={d['cloud_planning_calls']:3d}",flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
