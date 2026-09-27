import pathlib,csv,json,statistics,hashlib
h=pathlib.Path('/tmp/ds4-splash-20260926');p=h/'final-validation';out=pathlib.Path('/Users/jw/ds4/speed-bench/ds41f-m3ultra-perf/splash')
combined={}
for tag in ['v41','glm']:
 prefix='glm-coherent-combined' if tag=='glm' else 'v41-combined'
 files=[p/f'{prefix}-{i}.csv' for i in range(1,5)]
 if not all(f.exists() for f in files):continue
 data=[list(csv.DictReader(f.open())) for f in files];rows=[]
 assert all(len(x)==len(data[0]) for x in data)
 for n,row in enumerate(data[0]):
  r={'ctx':int(row['ctx_tokens']),'prefill_tokens':int(row['prefill_tokens'])}
  for metric in ['prefill_tps','gen_tps','gen_steady_tps']:
   values=[float(d[n][metric]) for d in data];base=statistics.harmonic_mean([values[0],values[3]]);cand=statistics.harmonic_mean(values[1:3]);r[metric]={'base':base,'candidate':cand,'gain_percent':100*(cand/base-1),'runs':values}
  rows.append(r)
 combined[tag]=rows
(out/'combined-results.json').write_text(json.dumps(combined,indent=2)+'\n')
http={}
for tag in ['v41','glm']:
 dirs=[p/f'http-{tag}-{i}' for i in range(1,5)]
 if not all((d/'images.json').exists() and (d/'results.json').exists() for d in dirs):continue
 records=[json.loads((d/'results.json').read_text()) for d in dirs];images=[json.loads((d/'images.json').read_text()) for d in dirs]
 assert all(images[0]==x for x in images[1:]),(tag,'checkpoint image mismatch')
 results=[]
 for i,r in enumerate(records[0]):
  others=[x[i] for x in records]
  def output(x):return {field:''.join(part.get(field,'') for part in x['parts']) for field in ['content','reasoning_content']}
  assert all(output(others[0])==output(x) for x in others[1:]),(tag,r['name'],'output mismatch')
  assert all(x['usage']==others[0]['usage'] for x in others[1:]),(tag,r['name'],'usage mismatch')
  values=[x['ttft_ms'] for x in others];base=statistics.mean([values[0],values[3]]);cand=statistics.mean(values[1:3]);rows={'name':r['name'],'staged_ttft_ms':base,'direct_ttft_ms':cand,'saved_ms':base-cand,'runs_ttft_ms':values,'output_exact':True,'usage':r['usage']}
  for arm,ids in [('staged',[0,3]),('direct',[1,2])]:
   timings=sorted(t for idx in ids for t in others[idx]['itl_ms']);rows[arm+'_itl_p99_ms']=timings[min(len(timings)-1,int(.99*len(timings)))]
  results.append(rows)
 http[tag]={'results':results,'images_exact':True,'images':images[0]}
(out/'http-results.json').write_text(json.dumps(http,indent=2)+'\n')
print(json.dumps({'combined':combined,'http':http},indent=2))
