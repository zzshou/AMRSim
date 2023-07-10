from tqdm import tqdm
from amr_utils.amr_readers import AMR_Reader
import json
from utils import simplify_amr_nopar

reader = AMR_Reader()


def get_amrs_file(file):
    files = [file]
    data = []
    for f in tqdm(files):
        amrs = reader.load(f, remove_wiki=True)
        data.extend(amrs)
    return data


def save_data(data, output_file):
    with open(output_file, 'w', encoding="utf-8") as fd:
        for example in data:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def combine(src_file, tgt_file, save_file, score_file=None):
    src_data = get_amrs_file(src_file)
    tgt_data = get_amrs_file(tgt_file)
    try:
        with open(score_file, 'r') as f:
            scores = f.readlines()
        assert len(scores) == len(src_data) == len(tgt_data)
    except:
        # scores = [0, 1] * int(len(src_data) / 2)
        scores = ['-1'] * len(src_data)

    error_log = 0
    error_log_claim = 0
    error_log_g = 0

    json_data = []
    for src, tgt, y in tqdm(zip(src_data, tgt_data, scores)):
        d = {}
        d['score'] = y.strip()
        ref1_sen = ' '.join(src.tokens)
        d['ref1'] = ref1_sen
        ref1_graph = src.graph_string()

        graph_simple, triples = simplify_amr_nopar(ref1_graph)

        d['graph_ref1'] = {}
        graph_simple = ' '.join(graph_simple)
        d['graph_ref1']['amr_simple'] = graph_simple
        d['graph_ref1']['triples'] = json.dumps(triples)

        try:
            ref2_sen = ' '.join(tgt.tokens)
            d['ref2'] = ref2_sen
            ref2_graph = tgt.graph_string()

            graph_simple, triples = simplify_amr_nopar(ref2_graph)

            d['graph_ref2'] = {}
            graph_simple = ' '.join(graph_simple)
            d['graph_ref2']['amr_simple'] = graph_simple
            d['graph_ref2']['triples'] = json.dumps(triples)

        except:
            error_log_claim += 1
            print("skip graph claim", error_log_claim)
            d['graph_ref2'] = {}
            d['graph_ref2']['amr_simple'] = ''
            d['graph_ref2']['triples'] = ''
        json_data.append(d)

    print("skipped graph sents", error_log_g)
    print("skipped sents", error_log)
    print("skipped graph claim", error_log_claim)

    save_data(json_data, save_file)


if __name__ == '__main__':
    src_file = '../data/src.amr'
    tgt_file = '../data/tgt.amr'
    save_file = '../data/src_tgt.json'
    combine(src_file, tgt_file, save_file)
