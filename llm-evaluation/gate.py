"""Paired regression decision for baseline and candidate answers."""

from math import comb


def regression_test(baseline, candidate, alpha=0.05):
    """One-sided exact McNemar test on per-question correctness.

    Only questions where the two versions disagree carry information. Under the
    null hypothesis that the candidate is no worse, each disagreement is equally
    likely to favour either version. The candidate fails when it loses
    significantly more questions than it gains.
    """
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Baseline and candidate need the same non-empty set of questions")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    lost = sum(b and not c for b, c in zip(baseline, candidate, strict=True))
    gained = sum(c and not b for b, c in zip(baseline, candidate, strict=True))
    discordant = lost + gained
    p_value = sum(comb(discordant, k) for k in range(lost, discordant + 1)) / 2**discordant
    return {
        "questions": len(baseline),
        "baseline_accuracy": sum(baseline) / len(baseline),
        "candidate_accuracy": sum(candidate) / len(candidate),
        "lost": lost,
        "gained": gained,
        "p_value": p_value,
        "alpha": alpha,
        "regression": p_value < alpha,
    }
