from sentence_transformers import ExtendSentenceTransformer, InputExample
import json
from preprocess import generate_ref_edge
from sklearn.metrics.pairwise import paired_cosine_distances
import warnings

warnings.filterwarnings('ignore')

########### Load the model and evaluate on test set
test_sts_dataset_path = '../data/src_tgt.json'
model_path = "output/ct-wiki-bert"


def return_simscore(model, examples):
    scores = []
    model.training = False
    for example in examples:
        sentences1 = example.texts[0]
        sentences2 = example.texts[1]
        ref1_graphs_index = example.edge_index[0]
        ref1_graphs_type = example.edge_type[0]
        ref1_pos_ids = example.pos_ids[0]
        embeddings1 = model.encode([sentences1], graph_index=[ref1_graphs_index],
                                   graph_type=[ref1_graphs_type], batch_size=1,
                                   convert_to_numpy=True,
                                   pos_ids=[ref1_pos_ids])

        ref2_graphs_index = example.edge_index[1]
        ref2_graphs_type = example.edge_type[1]
        ref2_pos_ids = example.pos_ids[1]
        embeddings2 = model.encode([sentences2], graph_index=[ref2_graphs_index],
                                   graph_type=[ref2_graphs_type], batch_size=1,
                                   convert_to_numpy=True,
                                   pos_ids=[ref2_pos_ids])
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        # print(cosine_scores[0])
        print(cosine_scores[0])
        scores.append(cosine_scores[0])
    return scores


model = ExtendSentenceTransformer(model_path)
tokenizer = model.tokenizer
test_samples = []
with open(test_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        max_seq_length = model.max_seq_length
        edge_index, edge_type, pos_ids = generate_ref_edge(line, tokenizer, max_seq_length)
        inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                   edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        # print(inp_example.pos_ids)
        test_samples.append(inp_example)
test_result = return_simscore(model, test_samples)
print(test_result)
