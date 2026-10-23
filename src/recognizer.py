from src.dtw import average_template, dtw
from src.gaussian import fit_gaussian, negative_log_likelihood


def group_by_vowel(items):
    groups = {}
    for item in items:
        groups.setdefault(item["vowel"], []).append(item["features"])
    return groups


def build_dtw_templates(items):
    return {vowel: average_template(seqs) for vowel, seqs in group_by_vowel(items).items()}


def build_gaussian_models(items):
    return {vowel: fit_gaussian(seqs) for vowel, seqs in group_by_vowel(items).items()}


def recognize_dtw(features, templates):
    costs = {vowel: dtw(template, features)[0] for vowel, template in templates.items()}
    return min(costs, key=costs.get), costs


def recognize_gaussian(features, models):
    costs = {vowel: negative_log_likelihood(features, mean, cov) for vowel, (mean, cov) in models.items()}
    return min(costs, key=costs.get), costs
