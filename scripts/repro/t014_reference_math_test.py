"""Closed-form CPU semantic checks, zero environment interactions."""
import json,math,random
from pathlib import Path
from t014_reference_math import soft_q_return,paired_direction_summary,perturbed_actions

def main():
    # Hand-computed:1+.5*(2-.2*4)+.25*(3-.2*5)=2.1.
    a=soft_q_return([1,2,3],[100,4,5],.5,.2)
    b=soft_q_return([1,2,3],[-1000,4,5],.5,.2)
    assert abs(a-2.1)<1e-14 and a==b
    assert soft_q_return([3],[999])==3
    # Known finite-H expectation: sum gamma^t*(2+t)*tanh(z), plus shared zero-mean noise.
    z=.2;delta=.025;H=7;gamma=.8
    plus,minus=perturbed_actions([z],[1.],delta)
    gp=math.fsum(gamma**t*(2+t) for t in range(H))
    expected=gp*(plus[0]-minus[0])/(2*delta)
    p=[];m=[]
    for rep in range(4):
        noises=[random.Random(140014+rep*100+t).gauss(0,1) for t in range(H)]
        logp=[-1-t*.1 for t in range(H)]
        p.append(soft_q_return([(2+t)*plus[0]+noises[t] for t in range(H)],logp,gamma,.2))
        m.append(soft_q_return([(2+t)*minus[0]+noises[t] for t in range(H)],logp,gamma,.2))
    stats=paired_direction_summary(p,m,delta)
    assert abs(stats['mean']-expected)<1e-12
    assert stats['mean_standard_error']<1e-12
    true_derivative=gp*(1-math.tanh(z)**2)
    assert abs(stats['mean']-true_derivative)/abs(true_derivative)<.001
    report={'status':'passed_CPU_math_only','first_entropy_excluded':True,'hand_computed_soft_return':a,'paired_toy_expected_direction':expected,'paired_summary':stats,'relative_finite_delta_error':abs(stats['mean']-true_derivative)/abs(true_derivative),'GPU_jobs':0,'environment_interactions':0,'optimizer_updates':0,'not_verified':['MuJoCo full-state restoration','stochastic policy logprob mapping','real finite-H/delta uncertainty','production rollout code']}
    out=Path('_so2_work/t014/reference-math-cpu.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
