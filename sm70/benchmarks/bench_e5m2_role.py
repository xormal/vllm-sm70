import math, os, sys, subprocess, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
D=256
def foreign():
    me=str(os.getpid())
    uu=subprocess.run(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader","-i","0,1"],capture_output=True,text=True).stdout.split()
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],capture_output=True,text=True).stdout
    return sum(1 for l in apps.strip().splitlines() if l.strip() and l.split(",")[0].strip() in uu and l.split(",")[1].strip()!=me)
p=fa2_sm70._ext.prefill_ext(); torch.manual_seed(0)
H,Hkv,Sq=16,2,512; sc=1/math.sqrt(D)
print(f"# чужих на картах 0-1: {foreign()}")
print("# +128 LOP3 x 8 СЧИТАЮЩИХ варпов = 1024 выдачи;  +64 LOP3 x 4 ВЕЗУЩИХ = 256 выдач (вчетверо меньше)")
print(f"{'Sk':>8} {'e5m2':>9} {'счит(+1024)':>12} {'везу(+256)':>11} | {'цена счит':>10} {'цена везу':>10} | {'на 1000 выдач: счит':>20} {'везу':>8} {'во сколько':>11}")
for Sk in [16384,65536,131072]:
    q=(torch.randn(1,Sq,H,D,device='cuda',dtype=torch.float16)*0.3).contiguous()
    kf=(torch.randn(1,Hkv,Sk,D,device='cuda',dtype=torch.float16)*0.4).contiguous()
    vf=(torch.randn(1,Hkv,Sk,D,device='cuda',dtype=torch.float16)*0.4).contiguous()
    kb=kf.to(torch.float8_e5m2).view(torch.uint8).contiguous(); vb=vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
    def mk(v):
        def g():
            os.environ['FA2SM70_FAKEXOR']=v; return p.attn_fwd_volta_e5m2(q,kb,vb,sc,True)
        return g
    fns={'base':mk('0'),'cons':mk('1'),'prod':mk('2')}
    for f in fns.values():
        for _ in range(3): f()
    torch.cuda.synchronize(); acc={k:[] for k in fns}
    for _ in range(11):
        for k,f in fns.items():
            s,e=torch.cuda.Event(True),torch.cuda.Event(True)
            s.record(); f(); e.record(); torch.cuda.synchronize(); acc[k].append(s.elapsed_time(e))
    m={k:sorted(v)[len(v)//2] for k,v in acc.items()}
    dc=m['cons']-m['base']; dp=m['prod']-m['base']
    pc=100*dc/m['base']/1024*1000; pp=100*dp/m['base']/256*1000
    print(f"{Sk:>8} {m['base']:>9.3f} {m['cons']:>12.3f} {m['prod']:>11.3f} | {dc:>10.3f} {dp:>10.3f} | {pc:>20.4f}% {pp:>7.4f}% {pp/pc if pc else 0:>10.2f}x")
    del q,kf,vf,kb,vb; torch.cuda.empty_cache()
print(f"# чужих на картах 0-1 в конце: {foreign()}")
