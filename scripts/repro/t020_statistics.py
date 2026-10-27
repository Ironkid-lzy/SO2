"""Unvalidated fixed-panel T020 analysis; execute only inside its science budget.

All coordinates must use the same actor parameter order. Repeated rows describe
conditional mask Monte Carlo noise, not independent training seeds or Q error.
This module has no experiment launcher, environment or optimizer entry point.
"""
import math
import torch


REPEATS = 16
RELATIVE_NORM_FLOOR = 1e-12


def bank_summary(replicates):
    """Return the full-coordinate mean and L2 standard error of that mean."""
    x = torch.as_tensor(replicates, dtype=torch.float64, device="cpu").detach()
    assert x.ndim == 2 and x.shape[0] == REPEATS and x.shape[1] > 0
    assert torch.isfinite(x).all()
    mean = x.mean(0)
    sem = float(((x-mean).square().sum() / (REPEATS*(REPEATS-1))).sqrt())
    return {"mean": mean, "sem": sem, "norm": float(mean.norm())}


def cosine(a, b):
    denominator = float(a.norm()) * float(b.norm())
    if denominator == 0:
        return None
    return max(-1.0, min(1.0, float(torch.dot(a, b))/denominator))


def public_summary(bank):
    return {"mean_norm": bank["norm"], "L2_MC_SEM": bank["sem"],
            "mask_repeats": REPEATS}


def calibrate(native_banks, unit_auxiliary_banks, strength_cap):
    """Calibrate once at seed0/online0, without amplifying weak mean signal.

    The caller must freeze this receipt before using any other panel. Saturation
    is reported, rather than silently claiming the requested 10% norm match.
    """
    assert len(native_banks) == len(unit_auxiliary_banks) == 2
    assert strength_cap in (1, 10)
    base = [bank_summary(x) for x in native_banks]
    aux = [bank_summary(x) for x in unit_auxiliary_banks]
    native_mean = (base[0]["mean"]+base[1]["mean"])/2
    auxiliary_mean = (aux[0]["mean"]+aux[1]["mean"])/2
    native_norm, auxiliary_norm = float(native_mean.norm()), float(auxiliary_mean.norm())
    alignment = cosine(aux[0]["mean"], aux[1]["mean"])
    stable = (native_norm > 0 and auxiliary_norm >= .01*native_norm
              and all(x["norm"] > 2*x["sem"] for x in aux)
              and alignment is not None and alignment > .5)
    receipt = {"eligible": stable, "auxiliary_banks": list(map(public_summary, aux)),
               "bank_cosine": alignment, "native_mean_norm": native_norm,
               "unit_auxiliary_mean_norm": auxiliary_norm, "target_ratio": .1,
               "strength_cap": strength_cap, "strength": None}
    if not stable:
        receipt["status"] = "weak_or_unstable_mean_no_amplification"
        return receipt
    requested = .1*native_norm/auxiliary_norm
    strength = min(requested, strength_cap)
    assert math.isfinite(strength) and strength > 0
    receipt.update(status="calibrated", requested_strength=requested,
                   strength=strength, saturated=requested > strength_cap,
                   achieved_ratio=strength*auxiliary_norm/native_norm)
    return receipt


def cross_bank_residual(candidate, control, reference_norm):
    """Fit each bank's scalar, apply it to the other bank, retain both signs."""
    assert len(candidate) == len(control) == 2 and reference_norm >= 0
    floor = RELATIVE_NORM_FLOOR*reference_norm
    if any(x["norm"] <= floor for x in control):
        return {"status": "numerically_zero_control_inconclusive",
                "repeatably_distinct": False, "norm_floor": floor}
    coefficients = [float(torch.dot(a["mean"], c["mean"])) / c["norm"]**2
                    for a, c in zip(candidate, control)]
    residuals, noise, records = [], [], []
    for index in range(2):
        applied = coefficients[1-index]
        residual = candidate[index]["mean"]-applied*control[index]["mean"]
        threshold = 2*(candidate[index]["sem"]+abs(applied)*control[index]["sem"])
        residuals.append(residual)
        noise.append(threshold)
        records.append({"fitted_here_scalar": coefficients[index],
                        "nonnegative_fitted_here_scalar": max(0.0, coefficients[index]),
                        "applied_other_bank_scalar": applied,
                        "residual_norm": float(residual.norm()),
                        "heuristic_noise_reference": threshold})
    norms = [float(x.norm()) for x in residuals]
    alignment = cosine(*residuals)
    difference = float((residuals[0]-residuals[1]).norm())
    distinct = (all(n > threshold for n, threshold in zip(norms, noise))
                and alignment is not None and alignment > .5
                and difference < min(norms))
    return {"status": "complete", "banks": records, "residual_cosine": alignment,
            "residual_difference_norm": difference, "norm_floor": floor,
            "repeatably_distinct": distinct, "is_p_value_or_Q_error": False}


def vector_comparison(a, b, reference_norm):
    """Descriptive full-gradient scaling, also used for min/BC directions."""
    na, nb = float(a.norm()), float(b.norm())
    result = {"candidate_norm": na, "reference_norm": nb, "cosine": cosine(a, b)}
    if nb <= RELATIVE_NORM_FLOOR*reference_norm:
        result.update(status="numerically_zero_reference", optimal_scalar=None)
    else:
        scalar = float(torch.dot(a, b))/nb**2
        result.update(status="complete", optimal_scalar=scalar,
                      nonnegative_optimal_scalar=max(0.0, scalar),
                      residual_norm=float((a-scalar*b).norm()))
    return result


def panel_report(total_gradient_banks, min_banks, bc_banks):
    """Analyze one fixed panel after coefficients and input hashes are frozen.

    Banks are method -> two [16, P] arrays of total actor gradients. H uses the
    paired native gradient from the same mask/actor-noise row, never an unrelated
    native estimate. This function does not combine seeds or infer return value.
    """
    names = ("native", "square", "k2", "known", "cross")
    assert set(total_gradient_banks) == set(names)
    tensors, means, auxiliary = {}, {}, {}
    for name in names:
        assert len(total_gradient_banks[name]) == 2
        tensors[name] = [torch.as_tensor(x, dtype=torch.float64, device="cpu").detach()
                         for x in total_gradient_banks[name]]
        means[name] = [bank_summary(x) for x in tensors[name]]
        for index in range(2):
            assert tensors[name][index].shape == tensors["native"][index].shape
        if name != "native":
            auxiliary[name] = [bank_summary(x-y) for x, y in
                               zip(tensors[name], tensors["native"])]
    native = (means["native"][0]["mean"]+means["native"][1]["mean"])/2
    full = {name: (value[0]["mean"]+value[1]["mean"])/2
            for name, value in means.items()}
    scale = max(float(native.norm()), *(x["norm"] for x in auxiliary["cross"]))
    controls = {name: cross_bank_residual(auxiliary["cross"], auxiliary[name], scale)
                for name in ("square", "k2", "known")}
    extra = {"min": min_banks, "BC": bc_banks}
    for name, banks in extra.items():
        assert len(banks) == 2
        summarized = [bank_summary(x) for x in banks]
        full[name] = (summarized[0]["mean"]+summarized[1]["mean"])/2
        assert full[name].shape == native.shape
    return {"status": "complete", "controls": controls,
            "panel_direction_gate": all(x["repeatably_distinct"] for x in controls.values()),
            "auxiliary_banks": {name: list(map(public_summary, banks))
                                for name, banks in auxiliary.items()},
            "full_gradient_comparisons": {
                name: vector_comparison(full["cross"], vector, scale)
                for name, vector in full.items() if name != "cross"},
            "scope": "Conditional fixed-panel MC diagnostic, not significance or usefulness"}
