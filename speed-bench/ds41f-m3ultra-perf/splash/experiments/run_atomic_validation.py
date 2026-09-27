"""Historical atomic-candidate driver. Its HC toggle comparison was invalid
because the feature switch is cached; see invalid-reference-comparisons.json.
Use run_coherent_final.py for the retained implementation.
"""
import subprocess,pathlib,sys,json,hashlib,os
h=pathlib.Path('/tmp/ds4-splash-20260926');root=h/'final';base=h/'baseline';out=h/'final-validation';out.mkdir(exist_ok=True)
model='/Users/jw/ds4/gguf/DeepSeek-V4.1-Flash-Q4.gguf';glm='/Users/jw/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf';prompt='speed-bench/promessi_sposi.txt'
def run(name,cmd,env=(),cwd=root):
 record=out/(name+'.json')
 if record.exists() and json.loads(record.read_text()).get('returncode')==0:print('SKIP completed',name,flush=True);return
 assert not record.exists(),name+' needs investigation/new name'
 print('START',name,flush=True);args=['python3','speed-bench/ds41f-m3ultra-perf/run.py','--out',str(out),'--name',name]
 for v in env:args+=['--env',v]
 r=subprocess.run([*args,'--',*cmd],cwd=cwd);print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(name+' failed')
def compare_captures(tag):
 b=h/('base-'+tag);c=h/('final-'+tag)
 p=subprocess.run(['python3','speed-bench/ds41f-m3ultra-perf/compare_capture.py',str(b/'logits.bin'),str(c/'logits.bin')],cwd=root,check=True,text=True,capture_output=True)
 (out/(tag+'-logit-comparison.json')).write_text(p.stdout)
 assert sorted(f.name for f in (b/'snap').glob('*.bin'))==sorted(f.name for f in (c/'snap').glob('*.bin'))
 rows=[]
 for f in (b/'snap').glob('*.bin'):
  other=c/'snap'/f.name
  subprocess.run(['cmp',str(f),str(other)],check=True)
  rows.append({'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'exact':True})
 (out/(tag+'-snapshot-comparison.json')).write_text(json.dumps(rows,indent=2))
phase=sys.argv[1]
if phase=='correctness':
 for exe in ['tests/test_deepseek41_attention','tests/test_deepseek41_fusions','tests/test_deepseek41_topk','tests/test_deepseek41_q4_tail','tests/test_deepseek41_engram_admission','tests/test_glm53_kda']:
  run(pathlib.Path(exe).name,['./'+exe],['MTL_DEBUG_LAYER=1'])
 run('live-edges-fixed',['./tests/test_deepseek41_live_edges',model,prompt])
 run('verify-parity',['./tests/test_deepseek41_dspark',model,'--verify-parity',prompt])
 run('verify-stage-timing',['./tests/test_deepseek41_dspark',model,'--stage-timing',prompt])
 for tag,m,start,maximum,step,gen in [('v41',model,512,32703,4,64),('glm',glm,2048,16384,2,128)]:
  dst=h/('final-'+tag);(dst/'snap').mkdir(parents=True,exist_ok=True)
  run('capture-'+tag,[str(h/'final-capture'/'capture-bench'),'-m',m,'--prompt-file',prompt,'--ctx-start',str(start),'--ctx-max',str(maximum),'--step-mul',str(step),'--ctx-alloc','32768','--gen-tokens',str(gen)],['DS4_BENCH_FORCE_SNAPSHOT=1','DS4_BENCH_TRACE_LOGITS='+str(dst/'logits.bin'),'DS4_BENCH_SNAPSHOT_DIR='+str(dst/'snap')])
  compare_captures(tag)
 run('glm-hc-4096-soak',['./speed-bench/metal_decode_schedule_bench','-m',glm,'--prompt-file',prompt,'--candidate-env','DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE','--prefix-tokens','8192','--ctx','32768','--warmup','16','--tokens','4096','--include-selection'])
elif phase=='perf':
 for tag,m,start,maximum,step,gen in [('v41',model,512,32703,4,64),('glm',glm,2048,16384,2,128)]:
  if tag=='v41':subprocess.run(['python3',str(h/'warm_engram.py'),m],check=True,stdout=subprocess.DEVNULL)
  for i,cwd in enumerate([base,root,root,base]):
   name=tag+'-combined-'+str(i+1)
   run(name,['./ds4-bench','-m',m,'--prompt-file',prompt,'--ctx-start',str(start),'--ctx-max',str(maximum),'--step-mul',str(step),'--ctx-alloc','32768','--gen-tokens',str(gen),'--csv',str(out/(name+'.csv'))],cwd=cwd)
elif phase=='greedy':
 for tag,m,gen in [('v41',model,400),('glm',glm,128)]:
  for size in ['short','long']:
   args=['-m',m,'--prompt-file',str(h/('prompts-'+tag)/(size+'.txt')),'--temp','0','-n',str(gen),'-c','32768']+(['--raw-prompt'] if tag=='glm' else ['--nothink'])
   files=[]
   for variant,cwd in [('base',base),('final',root)]:
    name=f'{tag}-greedy-{size}-{variant}';dst=out/(name+'.txt');err=out/(name+'.stderr');files.append(dst)
    if not dst.exists():
     env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','MTL_'))}
     with dst.open('wb') as fp,err.open('wb') as ep:subprocess.run(['./ds4',*args],cwd=cwd,env=env,stdout=fp,stderr=ep,check=True)
   subprocess.run(['cmp',*[str(f) for f in files]],check=True)
   (out/f'{tag}-greedy-{size}-comparison.json').write_text(json.dumps({'exact':True,'sha256':hashlib.sha256(files[0].read_bytes()).hexdigest(),'argv':['./ds4',*args]},indent=2))
elif phase=='ssd':
 subprocess.run(['python3',str(h/'run_final.py'),'hc-cost'],check=True)
 run('ssd-session-v41',['./tests/test_deepseek41_graph',model,'--session-fixture'],['MTL_DEBUG_LAYER=1'])
 for tag,m in [('v41',model),('glm',glm)]:run('ssd-kv-'+tag,['./speed-bench/kvstore_write_bench',m,prompt,'ssd'])
elif phase=='hc-cost':
 import csv,statistics
 rates=[]
 for i,cwd in enumerate([base,root,root,base]):
  name='glm-hc-cost-long-'+str(i+1)
  run(name,['./ds4-bench','-m',glm,'--prompt-file',prompt,'--ctx-start','2048','--ctx-max','2048','--ctx-alloc','32768','--gen-tokens','1024','--csv',str(out/(name+'.csv'))],cwd=cwd)
  rates.append(float(next(csv.DictReader((out/(name+'.csv')).open()))['gen_tps']))
 b=statistics.harmonic_mean([rates[0],rates[3]]);c=statistics.harmonic_mean(rates[1:3]);cost=100*(b/c-1)
 result={'baseline_tps':b,'candidate_tps':c,'cost_percent':cost,'runs':rates,'passes_cost_budget':cost<.3}
 (out/'glm-hc-cost-long-result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result,flush=True)
 if cost>=.3:raise SystemExit('HC long-run cost needs investigation before proceeding')
elif phase=='http':
 for tag,m in [('v41',model),('glm',glm)]:
  if tag=='v41':subprocess.run(['python3',str(h/'warm_engram.py'),m],check=True,stdout=subprocess.DEVNULL)
  for i,staged in enumerate([True,False,False,True]):
   name=f'http-{tag}-{i+1}'
   run(name,['python3',str(h/'http_bench.py'),'--checkout',str(root),'--model',m,'--prompts',str(h/('prompts-'+tag)),'--out',str(out/name)]+(['--staged'] if staged else []))
else:raise SystemExit('unknown phase')
