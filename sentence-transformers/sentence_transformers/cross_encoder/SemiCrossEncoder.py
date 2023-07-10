from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
import logging
import os
from typing import Dict, Type, Callable, List
import torch
import math
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .. import SentenceTransformer, util
from ..evaluation import SentenceEvaluator
from ..models import ExtendTransformer
from torch_geometric.data import Data, Batch
from bert_extend import BertExtendModel

logger = logging.getLogger(__name__)

EDGES_AMR = ["have-rel-role", "have-org-role", "have-degree", "all-over", "distance-quantity", "date-entity", ":ARG0",
             ":ARG0-of",
             ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":ARG3", ":ARG3-of", ":ARG4",
             ":ARG4-of", ":ARG5", ":ARG5-of", ":ARG6", ":ARG6-of", ":ARG7", ":accompanier", ":accompanier-of",
             ":age", ":age-of", ":beneficiary", ":beneficiary-of", ":century", ":concession", ":concession-of",
             ":condition", ":condition-of", ":conj-as-if", ":consist", ":consist-of", ":day", ":dayperiod",
             ":dayperiod-of", ":decade", ":degree", ":degree-of", ":destination", ":destination-of", ":direction",
             ":direction-of", ":domain", ":domain-of", ":duration", ":duration-of", ":era", ":example", ":example-of",
             ":extent", ":extent-of", ":frequency", ":frequency-of", ":instrument", ":instrument-of", ":li",
             ":location",
             ":location-of", ":manner", ":manner-of", ":medium", ":medium-of", ":mod", ":mod-of", ":mode", ":month",
             ":name", ":op1", ":op1-of", ":op10", ":op11", ":op12", ":op12_<lit>", ":op13", ":op14", ":op14_<lit>_:",
             ":op15", ":op16", ":op17", ":op18", ":op19", ":op19_<lit>_:", ":op1_<lit>", ":op2", ":op2-of", ":op20",
             ":op21", ":op22", ":op23", ":op24", ":op25", ":op25_<lit>_:", ":op26", ":op27", ":op27_<lit>_.", ":op28",
             ":op29", ":op3", ":op3-of", ":op30", ":op31", ":op32", ":op33", ":op34", ":op35", ":op36", ":op37",
             ":op38", ":op39", ":op4", ":op40", ":op41", ":op5", ":op6", ":op7", ":op8", ":op9", ":ord", ":ord-of",
             ":part", ":part-of", ":path", ":path-of", ":polarity", ":polarity-of", ":polite", ":poss", ":poss-of",
             ":prep-a", ":prep-about", ":prep-after", ":prep-against", ":prep-against-of", ":prep-along-to",
             ":prep-along-with", ":prep-amid", ":prep-among", ":prep-around", ":prep-as", ":prep-at", ":prep-back",
             ":prep-between", ":prep-by", ":prep-down", ":prep-for", ":prep-from", ":prep-in", ":prep-in-addition-to",
             ":prep-into", ":prep-of", ":prep-off", ":prep-on", ":prep-on-behalf", ":prep-on-behalf-of", ":prep-on-of",
             ":prep-on-side-of", ":prep-out-of", ":prep-over", ":prep-past", ":prep-per", ":prep-through", ":prep-to",
             ":prep-toward", ":prep-under", ":prep-up", ":prep-upon", ":prep-with", ":prep-without", ":purpose",
             ":purpose-of",
             ":quant", ":quant-of", ":quant101", ":quant102", ":quant104", ":quant113", ":quant114", ":quant115",
             ":quant118",
             ":quant119", ":quant128", ":quant141", ":quant143", ":quant146", ":quant148", ":quant164", ":quant165",
             ":quant166",
             ":quant179", ":quant184", ":quant189", ":quant194", ":quant197", ":quant208", ":quant214", ":quant217",
             ":quant228",
             ":quant246", ":quant248", ":quant274", ":quant281", ":quant305", ":quant306", ":quant308", ":quant309",
             ":quant312",
             ":quant317", ":quant324", ":quant329", ":quant346", ":quant359", ":quant384", ":quant396", ":quant398",
             ":quant408",
             ":quant411", ":quant423", ":quant426", ":quant427", ":quant429", ":quant469", ":quant506", ":quant562",
             ":quant597",
             ":quant64", ":quant66", ":quant673", ":quant675", ":quant677", ":quant74", ":quant754", ":quant773",
             ":quant785", ":quant787",
             ":quant79", ":quant797", ":quant801", ":quant804", ":quant86", ":quant870", ":quarter", ":range", ":scale",
             ":season",
             ":snt1", ":snt12", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":source", ":source-of",
             ":subevent",
             ":subevent-of", ":time", ":time-of", ":timezone", ":timezone-of", ":topic", ":topic-of", ":unit", ":value",
             ":weekday",
             ":weekday-of", ":year", ":year2"]


class SemiCrossEncoder():
    def __init__(self, model_name: str, num_labels: int = None, max_length: int = None, device: str = None,
                 tokenizer_args: Dict = {},
                 automodel_args: Dict = {}, default_activation_function=None, cross_attention_heads=2,
                 adapter_size=128, gnn='RGCN', gnn_layer=3, gnn_head_num=4
                 ):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param automodel_args: Arguments passed to AutoModelForSequenceClassification
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.config.adapter_size = adapter_size
        self.config.gnn_layer = gnn_layer
        # config.num_hidden_layers = 1
        self.config.gnn_head_num = gnn_head_num
        self.config.gnn = gnn

        self.model = BertExtendModel.from_pretrained(model_name, config=self.config,
                                                     **automodel_args)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        # add amr edges to tokenizer
        new_tokens_vocab = {"additional_special_tokens": []}
        # sort by edge labels
        tokens_amr = sorted(EDGES_AMR, reverse=True)
        # add edges labels to model embeddings matrix
        for t in tokens_amr:
            new_tokens_vocab["additional_special_tokens"].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        print(num_added_toks, "tokens added.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_length = max_length

        self.cross_attention = MulitAttentionLayer(cross_attention_heads, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)))
        elif hasattr(self.config,
                     'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(
                self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        edge_indexs = [[] for _ in range(num_texts)]
        edge_types = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            if example.edge_index:
                for idx, (text, edge_index, edge_type) in enumerate(
                        zip(example.texts, example.edge_index, example.edge_type)):
                    texts[idx].append(text)
                    edge_indexs[idx].append(edge_index)
                    edge_types[idx].append(edge_type)
            else:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text)
                    edge_indexs[idx].append([])
                    edge_types[idx].append([])

            labels.append(example.label)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenizer(texts[idx], padding=True, truncation='longest_first', return_tensors="pt",
                                       max_length=self.max_length)
            tokenized['edge_index'] = edge_indexs[idx]
            tokenized['edge_type'] = edge_types[idx]

            sentence_features.append(tokenized)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device)

        for tokenized_line in sentence_features:
            for name in tokenized_line:
                try:
                    tokenized_line[name] = tokenized_line[name].to(self._target_device)
                except:
                    pass

        return sentence_features, labels

    def smart_batching_collate_text_only(self, batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        edge_indexs = [[] for _ in range(num_texts)]
        edge_types = [[] for _ in range(num_texts)]

        for example in batch:
            if example.edge_index:
                for idx, (text, edge_index, edge_type) in enumerate(
                        zip(example.texts, example.edge_index, example.edge_type)):
                    texts[idx].append(text)
                    edge_indexs[idx].append(edge_index)
                    edge_types[idx].append(edge_type)
            else:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text)
                    edge_indexs[idx].append([])
                    edge_types[idx].append([])

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenizer(texts[idx], padding=True, truncation='longest_first', return_tensors="pt",
                                       max_length=self.max_length)
            tokenized['edge_index'] = edge_indexs[idx]
            tokenized['edge_type'] = edge_types[idx]
            sentence_features.append(tokenized)

        for tokenized_line in sentence_features:
            for name in tokenized_line:
                try:
                    tokenized_line[name] = tokenized_line[name].to(self._target_device)
                except:
                    pass

        return sentence_features

    def transform_graph_geometric(self, embeddings, edge_index, edge_type):
        list_geometric_data = [Data(x=emb, edge_index=torch.tensor(edge_index[idx], dtype=torch.long),
                                    y=torch.tensor(edge_type[idx], dtype=torch.long)) for idx, emb in
                               enumerate(embeddings)]
        bdl = Batch.from_data_list(list_geometric_data)
        bdl = bdl.to("cuda:" + str(torch.cuda.current_device()))

        return bdl

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)
        self.cross_attention.to(self._target_device)
        self.classifier.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        # param_optimizer = list(self.model.named_parameters())

        trained_params = ['adapter_graph', 'embeddings']
        params = []
        params_name = []

        for n, p in self.model.named_parameters():
            for trained_param in trained_params:
                if trained_param in n:
                    params.append(p)
                    params_name.append(n)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in zip(params_name, params) if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in zip(params_name, params) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                           t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.MSELoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            # features are tokenized_pair
            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05,
                                         disable=not show_progress_bar):
                feature_a, feature_b = features
                edge_index_a = feature_a.pop('edge_index')
                edge_type_a = feature_a.pop('edge_type')
                edge_index_b = feature_b.pop('edge_index')
                edge_type_b = feature_b.pop('edge_type')
                if edge_index_a:
                    graph_batch_a = self.transform_graph_geometric(feature_a['input_ids'], edge_index_a, edge_type_a)
                    feature_a['graph_batch'] = graph_batch_a
                    graph_batch_b = self.transform_graph_geometric(feature_b['input_ids'], edge_index_b, edge_type_b)
                    feature_b['graph_batch'] = graph_batch_b

                if use_amp:
                    with autocast():
                        model_encoded_a = self.model(**feature_a, return_dict=True)[0]
                        model_encoded_b = self.model(**feature_b, return_dict=True)[0]

                        biatt_encoded_a, ab_att = self.cross_attention(model_encoded_a, model_encoded_b,
                                                                       feature_a['attention_mask'])  # b is query
                        biatt_encoded_b, ba_att = self.cross_attention(model_encoded_b, model_encoded_a,
                                                                       feature_b['attention_mask'])  # a is query
                        # output_cross_ab = torch.mean(
                        #     torch.stack([torch.mean(biatt_encoded_a, 1), torch.mean(biatt_encoded_b, 1)]),
                        #     dim=0)
                        biatt_encoded_a, biatt_encoded_b = biatt_encoded_a[:, 0, :], biatt_encoded_b[:, 0, :]
                        logits = activation_fct(torch.cosine_similarity(biatt_encoded_a, biatt_encoded_b))

                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_encoded_a = self.model(**feature_a, return_dict=True)[0]
                    model_encoded_b = self.model(**feature_b, return_dict=True)[0]
                    biatt_encoded_a, ab_att = self.cross_attention(model_encoded_a, model_encoded_b,
                                                                   feature_a['attention_mask'])  # b is query
                    biatt_encoded_b, ba_att = self.cross_attention(model_encoded_b, model_encoded_a,
                                                                   feature_b['attention_mask'])  # a is query
                    # output_cross_ab = torch.mean(
                    #     torch.stack([torch.mean(biatt_encoded_a, 1), torch.mean(biatt_encoded_b, 1)]),
                    #     dim=0)
                    biatt_encoded_a, biatt_encoded_b = biatt_encoded_a[:, 0, :], biatt_encoded_b[:, 0, :]
                    logits = activation_fct(torch.cosine_similarity(biatt_encoded_a, biatt_encoded_b))

                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def predict(self,
                example_pairs,
                batch_size: int = 32,
                show_progress_bar: bool = None,
                num_workers: int = 0,
                activation_fct=None,
                apply_softmax=False,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False
                ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        inp_dataloader = DataLoader(example_pairs, batch_size=batch_size,
                                    collate_fn=self.smart_batching_collate_text_only,
                                    num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        self.cross_attention.to(self._target_device)
        self.classifier.to(self._target_device)

        with torch.no_grad():
            for features in iterator:
                feature_a, feature_b = features
                edge_index_a = feature_a.pop('edge_index')
                edge_type_a = feature_a.pop('edge_type')
                edge_index_b = feature_b.pop('edge_index')
                edge_type_b = feature_b.pop('edge_type')
                if edge_index_a:
                    graph_batch_a = self.transform_graph_geometric(feature_a['input_ids'], edge_index_a, edge_type_a)
                    feature_a['graph_batch'] = graph_batch_a
                    graph_batch_b = self.transform_graph_geometric(feature_b['input_ids'], edge_index_b, edge_type_b)
                    feature_b['graph_batch'] = graph_batch_b
                model_encoded_a = self.model(**feature_a, return_dict=True)[0]
                model_encoded_b = self.model(**feature_b, return_dict=True)[0]
                biatt_encoded_a, ab_att = self.cross_attention(model_encoded_a, model_encoded_b,
                                                               feature_a['attention_mask'])  # b is query
                biatt_encoded_b, ba_att = self.cross_attention(model_encoded_b, model_encoded_a,
                                                               feature_b['attention_mask'])  # a is query
                # output_cross_ab = torch.mean(
                #     torch.stack([torch.mean(biatt_encoded_a, 1), torch.mean(biatt_encoded_b, 1)]),
                #     dim=0)
                # print(ab_att)
                # print(ba_att)

                biatt_encoded_a, biatt_encoded_b = biatt_encoded_a[:, 0, :], biatt_encoded_b[:, 0, :]
                logits = activation_fct(torch.cosine_similarity(biatt_encoded_a, biatt_encoded_b))

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score.item() for score in pred_scores]

        # if convert_to_tensor:
        #     pred_scores = torch.stack(pred_scores)
        # elif convert_to_numpy:
        #     pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)


class MulitAttentionLayer(nn.Module):
    """attention layer"""

    def __init__(self, n_head, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % n_head == 0  # make sure the output dimension equals to hidden_size
        self.n_head = n_head
        self.attention_head_size = hidden_size // n_head
        self.all_head_size = n_head * self.attention_head_size

        self.w_qs = nn.Linear(hidden_size, self.all_head_size)
        self.w_ks = nn.Linear(hidden_size, self.all_head_size)
        self.w_vs = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask):
        mixed_query_layer = self.w_qs(query_states)  # [b, seq, hs]

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask  # [b, 1, 1, seq]

        mixed_key_layer = self.w_ks(context_states)
        mixed_value_layer = self.w_vs(context_states)  # [b, seq, hs]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [b, n_head, seq, head_hs]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [b, n_head, seq, seq]

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [b, n_head, seq, seq]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [b, n_head, seq, head_hs]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [b, seq, n_head, head_hs]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [b, seq, hs]

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer

        return outputs, attention_probs
