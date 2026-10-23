"""Per-tick trace of scripted (theta = 0) episodes on FLUSH family sockets: phase, brick height above its seat,
brick lateral error, hand force, neighbour-joint utilisation and the contact pairs (WRIGGLE_FIT_RL.md §6.10).
    python flush_trace.py 100001 100002      # family seeds (100000 + k = validation split, k < 100000 = train)
"""
import sys, collections
sys.path.insert(0,'/home/rishadroxx/ISDN_Project-Robotic_LEGO_Assembly/Tanvir')
import numpy as np, mujoco, socket_env as SE, socket_family as SF
def classify(env):
    m,d=env.m,env.d
    c=collections.Counter()
    for k in range(d.ncon):
        con=d.contact[k]
        g1,g2=con.geom1,con.geom2
        b1,b2=m.body(m.geom_bodyid[g1]).name,m.body(m.geom_bodyid[g2]).name
        n1,n2=m.geom(g1).name,m.geom(g2).name
        def kind(b,n):
            if b=="hand": return "hand/"+n.split("/")[-1]
            return b+("/stud" if "stud" in n else "")
        a,b=sorted([kind(b1,n1),kind(b2,n2)])
        if a==b: continue
        c[(a,b)]+=1
    return c
which=sys.argv[1:]
for off,s in [(100000,0),(100000,1),(100000,2),(0,0),(0,1),(0,2)]:
    ctx=SF.family(off+s,"FLUSH")
    if which and str(off+s) not in which: continue
    env=SE.SocketInsertionEnv(mode="scripted",seed=10000)
    env.reset(context=ctx,level=0,brace="none")
    seat_z=env.true_tgt[0][2]-0.012
    print("==",ctx["id"],"faces",ctx["faces"],"axis",env.axis,"aim",round(env.aim_err_mm,2),"budget",round(env.budget,1))
    done=False;t=0
    while not done:
        o,r,done,info=env.step(np.zeros(9)); t+=1
        if env.base.phase in("search","press") or done:
            if t%3==0 or done:
                z=(env.d.xpos[env.bid][2]-seat_z)*1000
                cc=classify(env)
                top=", ".join("%s|%s:%d"%(a,b,n) for (a,b),n in cc.most_common(4))
                print(" t%3d %-6s z+%5.1fmm brickerr %.2fmm F %5.1f u %.2f  %s"%(t,env.base.phase,z,env._brick_err()*1000,np.linalg.norm(env.ft_true[:3]),env._u_struct(),top))
    print("  END",info["reason"],info["peak_force_N"],"broken",info["broken"])
