import subprocess,pathlib,json,csv,statistics,hashlib
h=pathlib.Path('/tmp/ds4-splash-20260926');out=h/'final-validation';source=h/'hc-coherent.metal';env='DS4_METAL_DSV4_HC_SOURCE='+str(source)
(out/'hc-coherent-source.json').write_text(json.dumps({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
def run(name,cmd,envs=(),cwd=h/'final'):
 meta=out/(name+'.json')
 if meta.exists():
  assert json.loads(meta.read_text()).get('returncode')==0,name
  return
 args=['python3','speed-bench/ds41f-m3ultra-perf/run.py','--out',str(out),'--name',name]
 for e in envs:args+=['--env',e]
 subprocess.run([*args,'--',*cmd],cwd=cwd,check=True)
run('hc-coherent-oracle',['./tests/test_glm53_kda'],[env,'MTL_DEBUG_LAYER=1'])
rates=[]
for i,cwd in enumerate([h/'baseline',h/'final',h/'final',h/'baseline']):
 name='glm-hc-coherent-cost-'+str(i+1)
 run(name,['./ds4-bench','-m','/Users/jw/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf','--prompt-file','speed-bench/promessi_sposi.txt','--ctx-start','2048','--ctx-max','2048','--ctx-alloc','32768','--gen-tokens','1024','--csv',str(out/(name+'.csv'))],[env] if i in [1,2] else [],cwd)
 rates.append(float(next(csv.DictReader((out/(name+'.csv')).open()))['gen_tps']))
b=statistics.harmonic_mean([rates[0],rates[3]]);c=statistics.harmonic_mean(rates[1:3]);cost=100*(b/c-1)
result={'baseline_tps':b,'candidate_tps':c,'cost_percent':cost,'runs':rates,'passes_cost_budget':cost<.3}
(out/'glm-hc-coherent-cost-result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if cost>=.3:raise SystemExit('coherent HC cost over budget')
