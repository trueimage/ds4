from pathlib import Path
import subprocess,hashlib,json
root=Path('/tmp/ds4-splash-20260926/final');out=Path('/tmp/ds4-splash-20260926/hc-compare');out.mkdir(exist_ok=True)
s=(root/'ds4.c').read_text()
a=s.index('static bool glm53_flash_feature_enabled(');b=s.index('\n/* Rows per split block',a)
part=s[a:b];needle='if (state[feature] == 0) {';assert part.count(needle)==1
part=part.replace(needle,'if (state[feature] == 0 || feature == GLM53_FLASH_HC_PRODUCER_FUSE) {');s=s[:a]+part+s[b:]
a=s.index('static bool glm53_graph_hc_pre(');b=s.index('\nstatic ',a+10)
part=s[a:b];assert part.count('if (fused > 0) {')==1
part=part.replace('if (fused > 0) {','if (fused > 0) {\n            hc_compare_fused++;')
needle='#endif\n    const uint32_t passes =';assert part.count(needle)==1
part=part.replace(needle,'#endif\n    hc_compare_separate++;\n    const uint32_t passes =')
helper='''static uint64_t hc_compare_fused, hc_compare_separate;
static void hc_compare_report(void) {
    fprintf(stderr,"HC executed: fused=%llu separate=%llu\\n",(unsigned long long)hc_compare_fused,(unsigned long long)hc_compare_separate);
}
__attribute__((constructor)) static void hc_compare_register(void) { atexit(hc_compare_report); }
'''
s=s[:a]+helper+part+s[b:];(out/'ds4.c').write_text(s)
objects=['ds4_help.o','ds4_gpu_args.o','ds4_image.o','ds4_distributed.o','ds4_tp.o','ds4_ssd.o','ds4_metal.o','ds4_layer_pack.o','ds4_engram.o']
flags=['cc','-O3','-ffast-math','-g','-mcpu=native','-Wall','-Wextra','-std=c99','-I'+str(root)]
subprocess.run([*flags,'-c',str(out/'ds4.c'),'-o',str(out/'ds4.o')],cwd=root,check=True)
benchmark=(root/'speed-bench/metal_decode_schedule_bench.c').read_text()
# Only this object uncaches HC and reports both paths; the ordinary harness
# must reject cached GLM switches instead of pretending to compare them.
benchmark=benchmark.replace('#include "variant_env.h"\n','').replace('    bench_reject_cached_glm_env(cfg.candidate_env);\n','')
(out/'benchmark.c').write_text(benchmark)
subprocess.run([*flags,str(out/'benchmark.c'),str(out/'ds4.o'),*[str(root/x) for x in objects],'-o',str(out/'hc-compare'),'-lm','-pthread','-framework','Foundation','-framework','Metal'],cwd=root,check=True)
(out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'production_source_sha256':hashlib.sha256((root/'ds4.c').read_bytes()).hexdigest(),'purpose':'Uncache only the HC rollback in this measurement object; report actual fused and separate executions. Production source and other objects unchanged.'},indent=2))
