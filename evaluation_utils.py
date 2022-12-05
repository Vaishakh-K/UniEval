import json
from scipy.stats.stats import spearmanr, pearsonr, kendalltau


def evaluate(source_file, scores, aspects=None):
    if aspects is None:
        aspects = ['consistency', 'fluency', 'relevance', 'coherence']
    test_data = json.load(open(source_file))
    correlations_file = f'{source_file.split(".")[0]}_correlation.json'
    correlations = {}
    for aspect in aspects:
        print("Aspect: ", aspect)
        expert_score_key = "expert_{}_mean".format(aspect)
        turker_score_key = "turker_{}_mean".format(aspect)

        annotations = {"expert": [example[expert_score_key] for example in test_data],
                       "turker": [example[turker_score_key] for example in test_data]}
        predicted_scores = [score[aspect] for score in scores['scores']]

        print("expert", annotations["expert"])
        print("turker", annotations["turker"])
        print("predictions", predicted_scores)

        expert_correlations = {"pearson": pearsonr(predicted_scores, annotations["expert"])[0],
                               "spearman": spearmanr(predicted_scores, annotations["expert"])[0],
                               "kendall": kendalltau(predicted_scores, annotations["expert"])[0]}

        turker_correlations = {"pearson": pearsonr(predicted_scores, annotations["turker"])[0],
                               "spearman": spearmanr(predicted_scores, annotations["turker"])[0],
                               "kendall": kendalltau(predicted_scores, annotations["turker"])[0]}

        correlations[aspect] = {"expert": expert_correlations, "turker": turker_correlations}

    with open(correlations_file, mode="w") as f_correlations:
        json.dump(correlations, f_correlations, indent=1)

