from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers import ExtendSentenceTransformer, LoggingHandler, InputExample
import logging
import json
from preprocess import generate_ref_edge

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

########### Load the model and evaluate on test set
test_sts_dataset_path = '../data/src_tgt.json'
model_path = "output/ct-wiki-bert"

model = ExtendSentenceTransformer(model_path)
tokenizer = model.tokenizer
test_samples = []
with open(test_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score'])  # / 5.0  # Normalize score to range 0 ... 1
        max_seq_length = model.max_seq_length
        edge_index, edge_type, pos_ids = generate_ref_edge(line, tokenizer, max_seq_length)
        inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                   label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        test_samples.append(inp_example)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test', batch_size=128)
test_evaluator.main_similarity = SimilarityFunction.COSINE
cos_score = test_evaluator(model)
# print(cos_score)  # spearson
