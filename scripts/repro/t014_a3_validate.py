"""Focused A3 regression and one preregistered complete-learner benchmark."""
import argparse
import copy
import gc
import statistics
from t014_a1_common import *
from t014_a1_validate import gradients
import t014_a3 as stage
import t014_a3_mask as a3


def policy(device, weights, off):
    result, cfg = make_policy(device,mode='matched')
    result._a3_actor_off = off
    load(result,weights)
    return result,cfg


def capture(p):
    p._a1_debug = True
    samples=[];count=[0]
    def before(module,inputs):
        count[0]+=1
        if getattr(module,'_dropout_pair_context',None) is None:
            samples.append(inputs[0].detach().clone())
    handle=p._model.critic.register_forward_pre_hook(before)
    return samples,handle


def correctness(device,weights):
    batch=fixed_batch(32)
    # Construct reference before patch installation: learn_mode holds A1 method.
    reference,_=make_policy(device,mode='matched');load(reference,weights)
    a3.install(False)
    actual,_=policy(device,weights,False)
    for p in (reference,actual):p._forward_learn_cnt=9
    for _ in range(2):
        state=rng_state();expected=reference.learn_mode.forward(batch);after=rng_state()
        restore_rng(state);got=actual.learn_mode.forward(batch)
        equivalent(expected,got);equivalent(reference.learn_mode.state_dict(),actual.learn_mode.state_dict())
        equivalent(gradients(reference),gradients(actual));equivalent(after,rng_state())
    del reference,actual
    on,_=policy(device,weights,False);off,_=policy(device,weights,True)
    for p in (on,off):p._forward_learn_cnt=9
    # Registered before A3 temporary prehook, thus actor call initially has None.
    on_actions,h1=capture(on);off_actions,h2=capture(off)
    for i in range(3):
        state=rng_state();a=on.learn_mode.forward(batch);after=rng_state()
        masks=[m.clone() for m in on._a1_last_pair.masks.values()]
        restore_rng(state);b=off.learn_mode.forward(batch)
        equivalent(after,rng_state())
        equivalent(masks,list(off._a1_last_pair.masks.values()))
        if i==0:
            assert a['critic_loss']==b['critic_loss']
            # Same input action means same reparameterized actor entropy sample;
            # only Q evaluation differs before actor optimizer step.
            equivalent(on_actions,off_actions)
        for p in (on,off):
            assert p._model.critic.training and p._target_model.critic.training
            assert p._model.critic._dropout_pair_context is p._target_model.critic._dropout_pair_context is None
            assert not any(x.grad is not None for x in p._target_model.parameters())
    h1.remove();h2.remove()
    assert off._a3_actor_queries==1 and off._a1_paired_updates==3
    # Explicit differentiable action + complete entropy term, actual critic path.
    obs=torch.linspace(-1,1,32*17,device=device).reshape(32,17)
    eps=torch.linspace(-.5,.5,32*6,device=device).reshape(32,6)
    mu,sigma=off._model(obs,mode='compute_actor')['logit']
    z=mu+sigma*eps;action=z.tanh()
    entropy=.2*(torch.distributions.Normal(mu,sigma).log_prob(z).sum(-1)-torch.log(1-action.square()+1e-6).sum(-1))
    off._model.critic._dropout_pair_context=a3.ActorContext()
    q=off._model({'obs':obs,'action':action},mode='compute_critic')['q_value'].mean(0)
    grads=torch.autograd.grad((entropy-q).mean(),tuple(off._model.actor.parameters())+(action,))
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.count_nonzero(grads[-1])>0
    # Same weights: dropout-off forward exactly equals critic eval-mode values,
    # while training flags remain True in the actual path. Entropy is untouched.
    saved=rng_state();off._model.critic.eval()
    expected=off._model({'obs':obs,'action':action.detach()},mode='compute_critic')['q_value'].mean(0)
    off._model.critic.train();restore_rng(saved)
    assert torch.equal(q.detach(),expected.detach())
    # Checkpoint serialization/reload preserves function, opt-in identity explicit.
    checkpoint=stage.WORK/f'validation-reload-{device}.pth'
    atomic_torch(checkpoint,off.learn_mode.state_dict())
    fresh,_=policy(device,weights,True)
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False)
    fresh.learn_mode.load_state_dict(restored)
    # Native frozen loader intentionally restores only network weights. These
    # test-only explicit optimizer/counter copies do not add production resume.
    fresh._optimizer_q.load_state_dict(restored['optimizer_q'])
    fresh._optimizer_policy.load_state_dict(restored['optimizer_policy'])
    off._forward_learn_cnt=fresh._forward_learn_cnt=19
    equivalent(off.learn_mode.state_dict(),fresh.learn_mode.state_dict())
    state=rng_state();before=off.learn_mode.forward(batch);after=rng_state()
    restore_rng(state);reloaded=fresh.learn_mode.forward(batch)
    equivalent(before,reloaded);equivalent(off.learn_mode.state_dict(),fresh.learn_mode.state_dict());equivalent(after,rng_state())
    # Exception after actor context setup must remove hook/context and retain modes.
    bad,_=policy(device,weights,True);bad._forward_learn_cnt=9
    original=a3.ActorContext.apply
    def fail(*args,**kwargs):raise RuntimeError('intentional context-cleanup check')
    a3.ActorContext.apply=fail
    try:
        try:bad.learn_mode.forward(batch)
        except RuntimeError as exc:assert 'intentional' in str(exc)
        else:raise AssertionError('failure injection did not execute')
    finally:a3.ActorContext.apply=original
    assert not bad._model.critic._forward_pre_hooks
    assert bad._model.critic._dropout_pair_context is bad._target_model.critic._dropout_pair_context is None
    assert bad._model.critic.training and bad._target_model.critic.training
    return dict(default_A1_loss_grad_model_optimizer_RNG_bitwise=True,
                matched_current_target_masks_and_three_update_RNG_exact=True,
                same_first_actor_action_and_entropy_sample=True,actor_action_gradient_nonzero=True,
                actor_off_equals_eval_critic_function=True,train_target_modes_unchanged=True,
                reload_network_function_and_testonly_explicit_optimizer_counter_continuity=True,production_resume_supported=False,exception_cleanup=True,
                environment_interactions=0,diagnostic_critic_updates=13,diagnostic_actor_updates=6)


def benchmark(weights):
    from ding.worker import BaseLearner
    p,cfg=policy('cuda',weights,False)
    init=stage.WORK/'benchmark-init.pth';atomic_torch(init,weights)
    cfg.policy.learn.learner.load_path=str(init)
    cfg.policy.learn.learner.hook.load_ckpt_before_run=str(init)
    learner=BaseLearner(cfg.policy.learn.learner,p.learn_mode,exp_name=str(stage.WORK/'benchmark'))
    learner.call_hook('before_run')
    assert p._forward_learn_cnt==learner.train_iter==0
    assert tensor_hash(p._model.state_dict())==tensor_hash(weights['model'])
    batch=fixed_batch(2560)
    times={'A1':[],'A3':[]};peaks={k:[] for k in times};reserved={k:[] for k in times}
    try:
        for name in ('A1','A3'):
            p._a3_actor_off=name=='A3'
            for _ in range(20):learner.train(batch,envstep=0)
        for pair in range(10):
            for name in (('A1','A3') if pair%2==0 else ('A3','A1')):
                p._a3_actor_off=name=='A3'
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                for _ in range(10):learner.train(batch,envstep=0)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                times[name].append(elapsed);peaks[name].append(torch.cuda.max_memory_allocated());reserved[name].append(torch.cuda.max_memory_reserved())
        assert learner.train_iter==p._forward_learn_cnt==p._a1_paired_updates==240
        assert p._a3_actor_queries==12
        ratio=statistics.median(times['A3'])/statistics.median(times['A1'])
        protocol=stage.read(stage.PROTO);b=protocol['budget']
        estimate=1.30*max(b['formal_reference_seconds']*ratio,950000*statistics.median(times['A3'])/10+protocol['benchmark']['historical_A1_nonlearner_seconds'])
        state=stage.read(ROOT/'_so2_work/t014/budget.json')
        available=min(b['stage_endpoint_seconds']-state['used_seconds'],86400-state['used_seconds'],b['job_inner_seconds'],b['job_outer_seconds'])-60
        return dict(times_seconds=times,median_ratio=ratio,cost_gate=ratio<=1.10,
                    estimated_formal_seconds_with30percent=estimate,available_seconds=available,
                    budget_gate=estimate<=available,formal_admitted=ratio<=1.10 and estimate<=available,
                    peak_allocated_bytes=peaks,peak_reserved_bytes=reserved,
                    complete_learner_updates=240,actor_updates=24,alpha_updates=0,batch_size=2560,
                    environment_interactions=0,timing_scope='BaseLearner.train complete240updates with20warm/mode,10pairs alternating10updates/block,oneevolvingpolicy,actor1in10; no retiming')
    finally:learner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True);args=parser.parse_args()
    torch.set_num_threads(2);seed_all(36036,args.device=='cuda');start=time.monotonic()
    stage.identity();weights,manifest=checked_weights()
    result=dict(device=args.device,source_sha256=stage.source_hashes(),protocol_sha256=stage.digest(stage.PROTO),parent_sha256=PARENT_SHA)
    result['correctness']=correctness(args.device,weights)
    if args.device=='cuda':
        gc.collect();torch.cuda.empty_cache();result['benchmark']=benchmark(weights)
    else:assert not torch.cuda.is_initialized()
    result.update(status='passed',wall_seconds=time.monotonic()-start)
    stage.save(stage.WORK/f'validation-{args.device}.json',result);print(json.dumps(result))
