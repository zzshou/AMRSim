from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict, Optional, Union, Tuple
import os

# EDGES_AMR = ["have-rel-role", "have-degree", "all-over", "distance-quantity", "date-entity", ":ARG0", ":ARG0-of",
#              ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":ARG3", ":ARG3-of", ":ARG4",
#              ":ARG4-of", ":ARG5", ":ARG5-of", ":ARG6", ":ARG6-of", ":ARG7", ":accompanier", ":accompanier-of",
#              ":age", ":age-of", ":beneficiary", ":beneficiary-of", ":century", ":concession", ":concession-of",
#              ":condition", ":condition-of", ":conj-as-if", ":consist", ":consist-of", ":day", ":dayperiod",
#              ":dayperiod-of", ":decade", ":degree", ":degree-of", ":destination", ":destination-of", ":direction",
#              ":direction-of", ":domain", ":domain-of", ":duration", ":duration-of", ":era", ":example", ":example-of",
#              ":extent", ":extent-of", ":frequency", ":frequency-of", ":instrument", ":instrument-of", ":li",
#              ":location",
#              ":location-of", ":manner", ":manner-of", ":medium", ":medium-of", ":mod", ":mod-of", ":mode", ":month",
#              ":name", ":op1", ":op1-of", ":op10", ":op11", ":op12", ":op12_<lit>", ":op13", ":op14", ":op14_<lit>_:",
#              ":op15", ":op16", ":op17", ":op18", ":op19", ":op19_<lit>_:", ":op1_<lit>", ":op2", ":op2-of", ":op20",
#              ":op21", ":op22", ":op23", ":op24", ":op25", ":op25_<lit>_:", ":op26", ":op27", ":op27_<lit>_.", ":op28",
#              ":op29", ":op3", ":op3-of", ":op30", ":op31", ":op32", ":op33", ":op34", ":op35", ":op36", ":op37",
#              ":op38", ":op39", ":op4", ":op40", ":op41", ":op5", ":op6", ":op7", ":op8", ":op9", ":ord", ":ord-of",
#              ":part", ":part-of", ":path", ":path-of", ":polarity", ":polarity-of", ":polite", ":poss", ":poss-of",
#              ":prep-a", ":prep-about", ":prep-after", ":prep-against", ":prep-against-of", ":prep-along-to",
#              ":prep-along-with", ":prep-amid", ":prep-among", ":prep-around", ":prep-as", ":prep-at", ":prep-back",
#              ":prep-between", ":prep-by", ":prep-down", ":prep-for", ":prep-from", ":prep-in", ":prep-in-addition-to",
#              ":prep-into", ":prep-of", ":prep-off", ":prep-on", ":prep-on-behalf", ":prep-on-behalf-of", ":prep-on-of",
#              ":prep-on-side-of", ":prep-out-of", ":prep-over", ":prep-past", ":prep-per", ":prep-through", ":prep-to",
#              ":prep-toward", ":prep-under", ":prep-up", ":prep-upon", ":prep-with", ":prep-without", ":purpose",
#              ":purpose-of",
#              ":quant", ":quant-of", ":quant101", ":quant102", ":quant104", ":quant113", ":quant114", ":quant115",
#              ":quant118",
#              ":quant119", ":quant128", ":quant141", ":quant143", ":quant146", ":quant148", ":quant164", ":quant165",
#              ":quant166",
#              ":quant179", ":quant184", ":quant189", ":quant194", ":quant197", ":quant208", ":quant214", ":quant217",
#              ":quant228",
#              ":quant246", ":quant248", ":quant274", ":quant281", ":quant305", ":quant306", ":quant308", ":quant309",
#              ":quant312",
#              ":quant317", ":quant324", ":quant329", ":quant346", ":quant359", ":quant384", ":quant396", ":quant398",
#              ":quant408",
#              ":quant411", ":quant423", ":quant426", ":quant427", ":quant429", ":quant469", ":quant506", ":quant562",
#              ":quant597",
#              ":quant64", ":quant66", ":quant673", ":quant675", ":quant677", ":quant74", ":quant754", ":quant773",
#              ":quant785", ":quant787",
#              ":quant79", ":quant797", ":quant801", ":quant804", ":quant86", ":quant870", ":quarter", ":range", ":scale",
#              ":season",
#              ":snt1", ":snt12", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":source", ":source-of",
#              ":subevent",
#              ":subevent-of", ":time", ":time-of", ":timezone", ":timezone-of", ":topic", ":topic-of", ":unit", ":value",
#              ":weekday",
#              ":weekday-of", ":year", ":year2"]


class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """

    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path: str = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model(model_name_or_path, config, cache_dir, **model_args)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir,
            **tokenizer_args)

        # # add amr edges to tokenizer
        # new_tokens_vocab = {"additional_special_tokens": []}
        # # sort by edge labels
        # tokens_amr = sorted(EDGES_AMR, reverse=True)
        # # add edges labels to model embeddings matrix
        # for t in tokens_amr:
        #     new_tokens_vocab["additional_special_tokens"].append(t)
        # num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        # print(num_added_toks, "tokens added.")
        # self.auto_model.resize_token_embeddings(len(self.tokenizer))

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config,
                                                              "max_position_embeddings") and hasattr(self.tokenizer,
                                                                                                     "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir,
                                                        **model_args)

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir,
                                                         **model_args)

    def _load_mt5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir,
                                                          **model_args)

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(),
                                                                    self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt",
                                     max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json',
                            'sentence_distilbert_config.json', 'sentence_camembert_config.json',
                            'sentence_albert_config.json', 'sentence_xlm-roberta_config.json',
                            'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)
