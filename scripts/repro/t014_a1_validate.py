"""Finite native-update regression, paired-mask tests and full-learner timing."""
import argparse
import gc
import math
import statistics
from unittest.mock import patch
from t014_a1_common import *
from t014_a1_mask import Pair, Context


def native_gate(device):
    x=torch.linspace(-2,2,2*512*256,device=device).reshape(2,512,256)
    state=rng_state()
    expected=torch.nn.functional.dropout(x,p=.005,training=True); end=rng_state()
    restore_rng(state)
    actual,mask=torch.ops.aten.native_dropout(x,.005,True)
    assert torch.equal(expected,actual); equivalent(end,rng_state())
    pair=Pair(True); current=Context(pair,False); target=Context(pair,True)
    for layer in range(3): current.apply(x,layer,.005,True)
    for layer in range(3):
        with torch.no_grad(): got=target.apply(x,layer,.005,True)
        expected=x*pair.masks[layer]*(1/.995)
        assert torch.equal(got,expected)
    # Detect accidental sharing, not calibrated epistemic uncertainty.
    masks=[~pair.masks[i] for i in range(3)]
    marginals=[]
    for m in masks:
        rate=m.float().mean().item(); n=m.numel()
        assert abs(rate-.005)<6*math.sqrt(.005*.995/n)+1e-5
        marginals.append(rate)
    comparisons=[(masks[0][0],masks[0][1]),(masks[0][0],masks[1][0]),
                 (masks[0][:,:-1],masks[0][:,1:])]
    for a,b in comparisons:
        joint=(a&b).float().mean().item(); n=a.numel(); p=.005**2
        assert abs(joint-p)<6*math.sqrt(p*(1-p)/n)+1e-5
    return {'native_outputs_exact':True,'native_RNG_exact':True,'drop_marginals':marginals}


def gradients(policy):
    return [None if p.grad is None else p.grad.detach().clone() for p in policy._model.parameters()]


def actor_grad(policy,device):
    obs=torch.linspace(-1,1,32*17,device=device).reshape(32,17)
    eps=torch.linspace(-.5,.5,32*6,device=device).reshape(32,6)
    mu,sigma=policy._model(obs,mode='compute_actor')['logit']
    z=mu+sigma*eps; action=z.tanh()
    logp=torch.distributions.Normal(mu,sigma).log_prob(z).sum(-1)-torch.log(1-action.square()+1e-6).sum(-1)
    q=policy._model({'obs':obs,'action':action},mode='compute_critic')['q_value'].mean(0)
    loss=(.2*logp-q).mean()
    return torch.autograd.grad(loss,tuple(policy._model.actor.parameters())+(action,))


def correctness(device,weights):
    batch=fixed_batch(32)
    reference,_=make_policy(device); actual,_=make_policy(device)
    load(reference,weights); load(actual,weights); bind_original(reference)
    assert not any('_dropout_pair' in k for k in actual._model.state_dict())
    for p in (reference,actual): p._forward_learn_cnt=9
    state=rng_state(); expected=reference.learn_mode.forward(batch); after=rng_state()
    restore_rng(state); got=actual.learn_mode.forward(batch)
    equivalent(expected,got);equivalent(reference.learn_mode.state_dict(),actual.learn_mode.state_dict())
    equivalent(gradients(reference),gradients(actual));equivalent(after,rng_state())
    assert actual._forward_learn_cnt==10
    del reference,actual
    independent,_=make_policy(device); matched,_=make_policy(device,mode='matched')
    load(independent,weights);load(matched,weights)
    for p in (independent,matched):p._forward_learn_cnt=9
    matched._a1_debug=True
    calls=[]
    handles=[]
    for name,model in [('current',matched._model),('target',matched._target_model)]:
        handles.append(model.critic.register_forward_pre_hook(
            lambda m,i,name=name: calls.append((name,getattr(m,'_dropout_pair_context',None) is not None,torch.is_grad_enabled()))))
    state=rng_state(); independent.learn_mode.forward(batch); after=rng_state()
    restore_rng(state)
    with patch.object(matched._optimizer_q,'step',wraps=matched._optimizer_q.step) as qstep, \
         patch.object(matched._optimizer_policy,'step',wraps=matched._optimizer_policy.step) as astep:
        matched.learn_mode.forward(batch)
        assert qstep.call_count==astep.call_count==1
    equivalent(after,rng_state())
    assert calls==[('current',True,True),('target',True,False),('current',False,True)],calls
    for h in handles:h.remove()
    assert not any(p.grad is not None for p in matched._target_model.parameters())
    assert not any(matched._a1_last_pair.target_grad_enabled)
    first=[m.clone() for m in matched._a1_last_pair.masks.values()]
    matched.learn_mode.forward(batch)
    assert all(not torch.equal(a,b) for a,b in zip(first,matched._a1_last_pair.masks.values()))
    assert matched._a1_paired_updates==2 and matched._forward_learn_cnt==11
    state=rng_state(); a=actor_grad(matched,device)
    bind_original(matched);restore_rng(state);b=actor_grad(matched,device)
    equivalent(a,b)
    assert torch.isfinite(a[-1]).all() and torch.count_nonzero(a[-1])>0
    del independent,matched
    gc.collect()
    return dict(default_loss_grad_update_RNG_exact=True,parent_strict=True,
                target_no_grad=True,actor_mask_independent_and_action_gradient_exact=True,
                no_extra_critic_forwards=True,cross_update_refresh=True,
                diagnostic_critic_updates=5,diagnostic_actor_updates=4,environment_interactions=0)


def benchmark(weights):
    from ding.worker import BaseLearner
    policy,cfg=make_policy('cuda')
    init=WORK/'benchmark-init.pth'
    if not init.exists(): torch.save(weights,init)
    cfg.policy.learn.learner.load_path=str(init)
    cfg.policy.learn.learner.hook.load_ckpt_before_run=str(init)
    learner=BaseLearner(cfg.policy.learn.learner,policy.learn_mode,exp_name=str(WORK/'benchmark'))
    learner.call_hook('before_run')
    assert learner.train_iter==policy._forward_learn_cnt==0
    assert tensor_hash(policy._model.state_dict())==tensor_hash(weights['model'])
    batch=fixed_batch(2560)
    times={'independent':[],'matched':[]};peaks={k:[] for k in times};reserved={k:[] for k in times}
    try:
        for mode in ('independent','matched'):
            policy._a1_mode=mode
            for _ in range(20): learner.train(batch,envstep=0)
        for pair in range(10):
            order=('independent','matched') if pair%2==0 else ('matched','independent')
            for mode in order:
                policy._a1_mode=mode
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                for _ in range(10): learner.train(batch,envstep=0)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                times[mode].append(elapsed)
                peaks[mode].append(torch.cuda.max_memory_allocated());reserved[mode].append(torch.cuda.max_memory_reserved())
        assert learner.train_iter==policy._forward_learn_cnt==240
        assert policy._a1_paired_updates==120
        ratio=statistics.median(times['matched'])/statistics.median(times['independent'])
        ledger=json.loads((ROOT/'_so2_work/t014/budget.json').read_text())
        protocol=json.loads(PROTO.read_text())
        measured=statistics.median(times['matched'])/10*950000
        estimated=1.30*max(11068.843548100005*ratio,
                           measured+protocol['GPU_preflight']['historical_nonlearner_seconds'])
        remaining=protocol['stage_budget']['absolute_T014_job_ledger_limit_for_A1']-ledger['used_seconds']-60
        result=dict(times_seconds=times,median_ratio=ratio,cost_gate=ratio<=1.10,
                    estimated_formal_seconds_with30percent=estimated,remaining_stage_seconds=remaining,
                    budget_gate=estimated<=remaining,peak_allocated_bytes=peaks,peak_reserved_bytes=reserved,
                    complete_learner_updates=240,actor_updates=24,alpha_updates=0,batch_size=2560,
                    timing_scope='Full BaseLearner.train including native mask generation/store/discard, gradients/optimizers/target update/hooks and actor1/10; one evolving policy, paired order.',
                    historical_wall_not_paired_speed_claim=True,environment_interactions=0)
        result['formal_admitted']=result['cost_gate'] and result['budget_gate']
        return result
    finally:
        learner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--tag',default='')
    args=parser.parse_args();start=time.monotonic();torch.set_num_threads(2)
    seed_all(35035,args.device=='cuda')
    weights,manifest=checked_weights()
    result={'status':'running','device':args.device,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':source_hashes(),'protocol_sha256':digest(PROTO),'parent_sha256':PARENT_SHA}
    result['native']=native_gate(args.device)
    result['correctness']=correctness(args.device,weights)
    if args.device=='cuda':
        gc.collect();torch.cuda.empty_cache()
        result['benchmark']=benchmark(weights)
    else:
        assert not torch.cuda.is_initialized()
    result.update(status='passed',wall_seconds=time.monotonic()-start)
    suffix='-'+args.tag if args.tag else ''
    atomic_json(WORK/f'validation-{args.device}{suffix}.json',result)
    print(json.dumps(result))
