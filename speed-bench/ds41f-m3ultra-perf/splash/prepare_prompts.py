import subprocess,pathlib,json,sys,hashlib
root=pathlib.Path('/Users/jw/ds4');home=pathlib.Path('/tmp/ds4-splash-20260926');corpus=(root/'speed-bench/promessi_sposi.txt').read_text();records={}
for tag,model in [('v41','DeepSeek-V4.1-Flash-Q4.gguf'),('glm','GLM-5.3-Flash-Q4_K.gguf')]:
 out=home/('prompts-'+tag);out.mkdir(exist_ok=True)
 for size in [8192,32000]:
  work=out/'working.txt'
  def count(n):
   text=f'Benchmark passage {size}.\n'+corpus[:n]+'\nSummarize this passage in a short paragraph.';work.write_text(text)
   p=subprocess.run([str(root/'ds4'),'--dump-tokens','--nothink','--ctx','32768','-m',str(root/'gguf'/model),'--prompt-file',str(work)],capture_output=True,check=True)
   return len(json.loads(p.stdout.splitlines()[0])),text
  lo,hi=1,min(len(corpus),size*6)
  while lo+1<hi:
   n=(lo+hi)//2;tokens,text=count(n)
   if tokens<=size:lo=n
   else:hi=n
  tokens,text=count(lo);target=out/f'http-{size:05d}.txt';target.write_text(text)
  records[str(target)]={'chat_tokens':tokens,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 work.unlink()
 (out/'short.txt').write_text('Write a portable C function that implements an open-addressing hash table lookup. Explain how it handles collisions and missing keys.\n')
 (out/'long.txt').write_text((out/'http-08192.txt').read_text())
(home/'prompt-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
