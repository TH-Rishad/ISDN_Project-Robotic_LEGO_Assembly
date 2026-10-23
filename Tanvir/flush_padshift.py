"""Scripted FLUSH/CORNER/SLOT sockets with and without the pad shift (WRIGGLE_FIT_RL.md §6.10).
    python flush_padshift.py FLUSH       # 6 train + 6 validation contexts, levels 0 and 2, no brace, one seed
"""
import sys, multiprocessing as mp
sys.path.insert(0,'/home/rishadroxx/ISDN_Project-Robotic_LEGO_Assembly/Tanvir')
import numpy as np
def job(a):
    import socket_env as SE, socket_family as SF
    cls,seed,lvl,ps=a
    ctx=SF.family(seed,cls)
    env=SE.SocketInsertionEnv(mode="scripted",seed=10000,pad_shift=ps)
    i=SE.run_episode(env,None,context=ctx,level=lvl,brace="none")
    tb=[b for b in ctx["bricks"] if b["id"]=="target"][0]
    return (cls,seed,lvl,ps,i["success"],i["reason"],i["steps"],tb["type"],ctx["faces"],round(i["peak_force_N"]))
if __name__=="__main__":
    cls=sys.argv[1]
    jobs=[(cls,off+s,l,ps) for off in (0,100000) for s in range(6) for l in (0,2) for ps in (False,True)]
    with mp.get_context("spawn").Pool(7) as p: rows=p.map(job,jobs)
    import collections
    for ps in (False,True):
        r=[x for x in rows if x[3]==ps]
        print(cls,"pad_shift",ps,"seated",sum(x[4] for x in r),"/",len(r),collections.Counter(x[5] for x in r))
        for l in (0,2): print("   level",l,sum(x[4] for x in r if x[2]==l),"/",sum(1 for x in r if x[2]==l))
    # paired details
    for a,b in zip([x for x in rows if not x[3]],[x for x in rows if x[3]]):
        if a[4]!=b[4]: print("  diff",a[:3],a[7],a[8],"old",a[5],"new",b[5])
