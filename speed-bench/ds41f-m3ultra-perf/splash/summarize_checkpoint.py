import pathlib,json,statistics,re
h=pathlib.Path('/tmp/ds4-splash-20260926');p=h/'final-validation';result={}
for tag in ['v41','glm']:
 dirs=[p/f'http-checkpoint-{tag}-{i}' for i in range(1,5)]
 if not all((d/'images.json').exists() for d in dirs):continue
 records=[json.loads((d/'results.json').read_text()) for d in dirs];images=[json.loads((d/'images.json').read_text()) for d in dirs]
 assert all(images[0]==x for x in images[1:]),(tag,'image mismatch')
 results=[]
 for i,r in enumerate(records[0]):
  others=[x[i] for x in records]
  def output(x):return {field:''.join(part.get(field,'') for part in x['parts']) for field in ['content','reasoning_content']}
  assert all(output(others[0])==output(x) for x in others[1:]),(tag,r['name'],'output mismatch')
  assert all(x['usage']==others[0]['usage'] for x in others[1:]),(tag,r['name'],'usage mismatch')
  values=[x['ttft_ms'] for x in others];base=statistics.mean([values[0],values[3]]);cand=statistics.mean(values[1:3]);row={'name':r['name'],'staged_ttft_ms':base,'direct_ttft_ms':cand,'saved_ms':base-cand,'runs_ttft_ms':values,'output_exact':True,'usage':r['usage']}
  for arm,ids in [('staged',[0,3]),('direct',[1,2])]:
   times=sorted(t for idx in ids for t in others[idx]['itl_ms']);row[arm+'_itl_p99_ms']=times[min(len(times)-1,int(.99*len(times)))]
  if r['name'].endswith('continued'):
   assert r['usage']['prompt_tokens']==10240
   stores=[]
   for d in dirs:
    lines=[l for l in (d/'server.log').read_text().splitlines() if 'kv cache stored tokens=10240 ' in l and 'reason=continued' in l]
    assert len(lines)==1,(tag,d,lines)
    stores.append(float(re.search(r'save=([\d.]+) ms',lines[0])[1]))
   row['runs_store_ms']=stores;row['staged_store_ms']=statistics.mean([stores[0],stores[3]]);row['direct_store_ms']=statistics.mean(stores[1:3])
  results.append(row)
 result[tag]={'results':results,'images_exact':True,'images':images[0]}
out=pathlib.Path('/Users/jw/ds4/speed-bench/ds41f-m3ultra-perf/splash/checkpoint-http-results.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
