"""Finite-H soft-Q reference semantics; no simulator, policy or training imports."""
import math
import statistics


def soft_q_return(rewards, log_probs, gamma=.99, alpha=.2):
    """r0 + sum_{t>=1} gamma**t (r_t-alpha logpi_t); no learned tail."""
    assert len(rewards) == len(log_probs) and len(rewards) > 0
    assert 0 <= gamma <= 1 and alpha >= 0
    assert all(math.isfinite(x) for x in list(rewards)+list(log_probs))
    terms = [rewards[0]]
    terms.extend(gamma**t*(rewards[t]-alpha*log_probs[t]) for t in range(1,len(rewards)))
    return math.fsum(terms)


def paired_direction_summary(plus, minus, delta):
    assert len(plus) == len(minus) and len(plus) >= 2 and delta > 0
    samples = [(p-m)/(2*delta) for p,m in zip(plus,minus)]
    assert all(math.isfinite(x) for x in samples)
    variance = statistics.variance(samples)
    return {'mean':statistics.mean(samples),'sample_variance':variance,
            'mean_standard_error':math.sqrt(variance/len(samples)),
            'paired_samples':samples,'replicates':len(samples),
            'scope':'finite-H finite-difference proxy; not exact environment gradient'}


def perturbed_actions(z, direction, delta):
    assert len(z) == len(direction) and delta > 0
    assert abs(math.fsum(x*x for x in direction)-1) < 1e-10
    return ([math.tanh(x+delta*u) for x,u in zip(z,direction)],
            [math.tanh(x-delta*u) for x,u in zip(z,direction)])
