import subprocess,pathlib,os,json,hashlib,sys
h=pathlib.Path('/tmp/ds4-splash-20260926');root=h/'final';out=h/'final-validation';prompt='speed-bench/promessi_sposi.txt'
models={'v41':'/Users/jw/ds4/gguf/DeepSeek-V4.1-Flash-Q4.gguf','glm':'/Users/jw/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf'}
def run(name,cmd,env=(),cwd=root):
 p=out/(name+'.json')
 if p.exists():
  assert json.loads(p.read_text()).get('returncode')==0,name+' failed previously'
  print('SKIP',name,flush=True);return
 print('START',name,flush=True);args=['python3','speed-bench/ds41f-m3ultra-perf/run.py','--out',str(out),'--name',name]
 for e in env:args+=['--env',e]
 subprocess.run([*args,'--',*cmd],cwd=cwd,check=True)
phase=sys.argv[1]
if phase=='correctness':
 for exe in ['tests/test_deepseek41_attention','tests/test_deepseek41_fusions','tests/test_deepseek41_topk','tests/test_deepseek41_q4_tail','tests/test_deepseek41_engram_admission','tests/test_glm53_kda']:
  run('coherent-'+pathlib.Path(exe).name,['./'+exe],['MTL_DEBUG_LAYER=1'])
 for tag,start,maximum,step,gen in [('v41',512,32703,4,64),('glm',2048,16384,2,128)]:
  dst=h/('final-coherent-'+tag);(dst/'snap').mkdir(parents=True,exist_ok=True)
  run('capture-coherent-'+tag,[str(h/'coherent-capture/capture-bench'),'-m',models[tag],'--prompt-file',prompt,'--ctx-start',str(start),'--ctx-max',str(maximum),'--step-mul',str(step),'--ctx-alloc','32768','--gen-tokens',str(gen)],['DS4_BENCH_FORCE_SNAPSHOT=1','DS4_BENCH_TRACE_LOGITS='+str(dst/'logits.bin'),'DS4_BENCH_SNAPSHOT_DIR='+str(dst/'snap')])
  b=h/('base-'+tag)
  r=subprocess.run(['python3','speed-bench/ds41f-m3ultra-perf/compare_capture.py',str(b/'logits.bin'),str(dst/'logits.bin')],cwd=root,check=True,capture_output=True,text=True)
  (out/(tag+'-coherent-logit-comparison.json')).write_text(r.stdout)
  assert sorted(f.name for f in (b/'snap').glob('*.bin'))==sorted(f.name for f in (dst/'snap').glob('*.bin'))
  snaps=[]
  for p in (b/'snap').glob('*.bin'):
   subprocess.run(['cmp',str(p),str(dst/'snap'/p.name)],check=True)
   snaps.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exact':True})
  (out/(tag+'-coherent-snapshot-comparison.json')).write_text(json.dumps(snaps,indent=2)+'\n')
 for size in ['short','long']:
  name='glm-greedy-'+size+'-coherent';dst=out/(name+'.txt');marker=out/(name+'-comparison.json')
  if marker.exists():continue
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','MTL_'))}
  args=['./ds4','-m',models['glm'],'--prompt-file',str(h/'prompts-glm'/(size+'.txt')),'--raw-prompt','--temp','0','-n','128','-c','32768']
  with dst.open('wb') as fp,(out/(name+'.stderr')).open('wb') as ep:subprocess.run(args,cwd=root,env=env,stdout=fp,stderr=ep,check=True)
  subprocess.run(['cmp',str(dst),str(out/('glm-greedy-'+size+'-base.txt'))],check=True)
  marker.write_text(json.dumps({'exact':True,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'argv':args},indent=2)+'\n')
 run('glm-hc-coherent-4096-soak',[str(h/'hc-compare/hc-compare'),'-m',models['glm'],'--prompt-file',prompt,'--candidate-env','DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE','--prefix-tokens','8192','--ctx','32768','--warmup','16','--tokens','4096','--include-selection'])
elif phase=='perf':
 for i,cwd in enumerate([h/'baseline',root,root,h/'baseline']):
  name='glm-coherent-combined-'+str(i+1)
  run(name,['./ds4-bench','-m',models['glm'],'--prompt-file',prompt,'--ctx-start','2048','--ctx-max','16384','--step-mul','2','--ctx-alloc','32768','--gen-tokens','128','--csv',str(out/(name+'.csv'))],cwd=cwd)
elif phase=='ssd':
 run('ssd-session-v41',['./tests/test_deepseek41_graph',models['v41'],'--session-fixture'],['MTL_DEBUG_LAYER=1'])
 for tag,m in models.items():run('ssd-kv-'+tag,['./speed-bench/kvstore_write_bench',m,prompt,'ssd'])
elif phase=='http':
 for tag,m in models.items():
  if tag=='v41':subprocess.run(['python3',str(h/'warm_engram.py'),m],check=True,stdout=subprocess.DEVNULL)
  for i,staged in enumerate([True,False,False,True]):
   name=f'http-{tag}-{i+1}'
   run(name,['python3',str(h/'http_bench.py'),'--checkout',str(root),'--model',m,'--prompts',str(h/('prompts-'+tag)),'--out',str(out/name)]+(['--staged'] if staged else []))
else:raise SystemExit('unknown phase')
