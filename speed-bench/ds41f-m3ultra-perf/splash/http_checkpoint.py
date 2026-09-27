import subprocess,pathlib,json,hashlib
h=pathlib.Path('/tmp/ds4-splash-20260926');root=h/'final';out=h/'final-validation';corpus=(root/'speed-bench/promessi_sposi.txt').read_text();manifest={}
models={'v41':'/Users/jw/ds4/gguf/DeepSeek-V4.1-Flash-Q4.gguf','glm':'/Users/jw/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf'}
# The default server interval rounds 10,000 up to 10,240. Both original HTTP
# fixtures confirmed API counts CLI-6 and a 10-token continuation turn:
# 10,172 CLI -> 10,166 API + 64 sampled tokens + 10 new tokens = 10,240.
for tag,model in models.items():
 prompts=h/('checkpoint-prompts-'+tag);prompts.mkdir(exist_ok=True);target=prompts/'http-10240.txt'
 if not target.exists():
  work=prompts/'working.txt'
  def count(n):
   text='Benchmark checkpoint passage.\n'+corpus[:n]+'\nSummarize this passage in a short paragraph.';work.write_text(text)
   r=subprocess.run(['./ds4','--dump-tokens','--nothink','--ctx','32768','-m',model,'--prompt-file',str(work)],cwd=root,capture_output=True,check=True)
   return len(json.loads(r.stdout.splitlines()[0])),text
  lo,hi=1,80000
  while lo+1<hi:
   n=(lo+hi)//2;tokens,text=count(n)
   if tokens<=10172:lo=n
   else:hi=n
  tokens,text=count(lo);assert tokens==10172,(tag,tokens);target.write_text(text);work.unlink()
 manifest[tag]={'cli_chat_tokens':10172,'expected_cold_api_tokens':10166,'expected_continued_api_tokens':10240,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(h/'checkpoint-prompt-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for tag,model in models.items():
 if tag=='v41':subprocess.run(['python3',str(h/'warm_engram.py'),model],check=True,stdout=subprocess.DEVNULL)
 for i,staged in enumerate([True,False,False,True]):
  name=f'http-checkpoint-{tag}-{i+1}';record=out/(name+'.json')
  if record.exists():
   assert json.loads(record.read_text()).get('returncode')==0,name+' failed previously'
   continue
  cmd=['python3',str(h/'http_bench.py'),'--checkout',str(root),'--model',model,'--prompts',str(h/('checkpoint-prompts-'+tag)),'--out',str(out/name),'--expect-continued-frontier','10240']+(['--staged'] if staged else [])
  subprocess.run(['python3','speed-bench/ds41f-m3ultra-perf/run.py','--out',str(out),'--name',name,'--',*cmd],cwd=root,check=True)
