import json
import logging
import torch
from tqdm import tqdm

from sentence_transformers import ExtendSentenceTransformer, LoggingHandler, models, InputExample
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from preprocess import generate_ref_edge, generate_wiki_edge

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

## Training parameters
model_name = 'bert-base-uncased'
model_type = model_name.split('/')[-1]
batch_size = 16
pos_neg_ratio = 4  # batch_size must be devisible by pos_neg_ratio
epochs = 1
max_seq_length = 128

gnn = 'GraphConv'
gnn_layer = 4
adapter_size = 128
learning_rate = 1e-5
add_graph = True
# Save path to store our model
model_save_path = 'output/ct-amr-bert'
################# Train sentences #################
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = '../data/wiki_train_data.json'
# train_sentences are simply your list of sentences
train_samples = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    lines = fIn.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        if add_graph:
            graph_triples = line["aligned_triples"]
            if graph_triples == []: continue
            edge_index, edge_type, pos_ids = generate_wiki_edge(graph_triples, max_seq_length)
            if edge_index[0] is None:
                continue
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']],
                                       edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
            train_samples.append(inp_example)
        else:
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']])
            train_samples.append(inp_example)

# For ContrastiveTension we need a special data loader to construct batches with the desired properties
train_dataloader = losses.ContrastiveTensionExampleDataLoader(train_samples, batch_size=batch_size,
                                                              pos_neg_ratio=pos_neg_ratio)

################# Intialize an SBERT model #################
word_embedding_model = models.ExtendTransformer(model_name, max_seq_length=max_seq_length, adapter_size=adapter_size,
                                                gnn=gnn, gnn_layer=gnn_layer, add_gnn=add_graph)
tokenizer = word_embedding_model.tokenizer

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = ExtendSentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Download and load STSb #################
dev_sts_dataset_path = '../data/dev-sense.json'
dev_samples = []

with open(dev_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score']) / 5.0  # Normalize score to range 0 ... 1
        if add_graph:
            edge_index, edge_type, pos_ids = generate_ref_edge(line, tokenizer, max_seq_length)
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        else:
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score)
        dev_samples.append(inp_example)
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

################# Train an SBERT model #################
# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=1,
    evaluation_steps=1000,
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={'lr': learning_rate},
    output_path=model_save_path,
    use_amp=False  # Set to True, if your GPU has optimized FP16 cores
)

########### Load the model and evaluate on test set
test_sts_dataset_path = '../data/src_tgt.json'
test_samples = []
with open(test_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score']) / 5.0  # Normalize score to range 0 ... 1
        if add_graph:
            edge_index, edge_type, pos_ids = generate_ref_edge(line, tokenizer, max_seq_length)
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        else:
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score)
        test_samples.append(inp_example)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
model = ExtendSentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
