from utils import convert_to_json
from metric.evaluator import get_evaluator

from evaluation_utils import evaluate
import torch

import json

import nltk
nltk.download('punkt')

class UniEvalEvaluator:
    def __init__(self, source_file, task='summarization'):
        self.task = task
        self.source_file = source_file


    def run_eval(self):
        # a list of source documents
        src_list = []
        # a list of human-annotated reference summaries
        ref_list = []

        # a list of model outputs to be evaluataed
        output_list = []

        evaluation_data = json.load(open(self.source_file))
        for example in evaluation_data:
            output_list.append(example["decoded"])
            src_list.append(example["text"])
            ref_list.append(example["references"][0])

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=output_list,
                               src_list=src_list, ref_list=ref_list)
        # Initialize evaluator for a specific task
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = get_evaluator(self.task, device=device)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=False)

        uni_eval_score = {
            "scores": eval_scores
        }
        results_file = f'{self.source_file.split(".")[0]}_results.json'
        with open(results_file, mode="w") as f:
            json.dump(uni_eval_score, f, indent=1)

        evaluate(self.source_file, uni_eval_score)


source_file = "data/sample_2.json"
unieval_evaluator = UniEvalEvaluator(source_file)
unieval_evaluator.run_eval()