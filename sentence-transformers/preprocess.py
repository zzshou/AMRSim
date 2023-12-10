import unidecode
import torch
import json
import networkx as nx
from tqdm import tqdm


def align_triples(graph_string_tok, graph_string, triples, size_claim_graph=None, map=None):
    graph_string = graph_string.split()
    if map:
        map = map
    else:
        map = {}
        idx_graph_string = 0
        map[idx_graph_string] = []

        try:
            tokenized_node = ""
            plain_tokenized_node = ""
            for idx, tok in enumerate(graph_string_tok):
                tokenized_node += unidecode.unidecode(tok.replace("##", "").lower())
                plain_tokenized_node += tok.replace("##", "").lower()
                unaccented_string = unidecode.unidecode(graph_string[idx_graph_string].lower())

                if not unaccented_string.startswith(tokenized_node) and not unaccented_string.startswith(
                        plain_tokenized_node):
                    idx_graph_string += 1
                    map[idx_graph_string] = []
                    tokenized_node = ""
                    plain_tokenized_node = ""
                    plain_tokenized_node += tok.replace("##", "").lower()
                    tokenized_node += unidecode.unidecode(tok.replace("##", "").lower())

                map[idx_graph_string].append(idx)
        except:
            print(graph_string_tok, graph_string)
            raise Exception("Error when converting graph to tokenized version.")

    assert len(map) == len(graph_string), graph_string

    # update triples with tokenized graph
    updated_triples = []
    for t in triples:
        if size_claim_graph:
            heads = map[t[0] + size_claim_graph]
            tails = map[t[1] + size_claim_graph]
        else:
            heads = map[t[0]]
            tails = map[t[1]]

        relation = t[2]

        for head in heads:
            for tail in tails:
                updated_triples.append((head, tail, relation))

    return updated_triples


def update_triples(all_triples, triples, graph_string):
    size_string = len(graph_string.split())

    updated_triples = []
    for t in triples:
        updated_triples.append((t[0] + size_string, t[1] + size_string, t[2]))

    return all_triples + updated_triples


def generate_edge_tensors(triples, max_seq_length_graph):
    set_edges = {"d": 0, "r": 1}

    edge_index_head = []
    edge_index_tail = []
    edge_types = []
    max_node = 0

    G = nx.DiGraph()  # 1 is source
    for t in triples:
        head = t[0]
        tail = t[1]
        if head > max_node:
            max_node = head
        if tail > max_node:
            max_node = tail

        relation = t[2]
        if relation == 'd':
            G.add_edge(head, tail)

        if head >= max_seq_length_graph or tail >= max_seq_length_graph:
            continue

        edge_index_head.append(head)
        edge_index_tail.append(tail)
        edge_types.append(set_edges[relation])

    edge_index = torch.tensor([edge_index_head, edge_index_tail], dtype=torch.long)
    edge_types = torch.tensor(edge_types, dtype=torch.long)
    try:
        length = nx.single_source_dijkstra_path_length(G, 1)
    except:
        print(triples)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_types = torch.tensor([0], dtype=torch.long)
        position_ids = [0, 1, 2]
        return edge_index, edge_types, position_ids
    position_ids = [0]

    for i in range(1, max_node + 1):
        if i in length:
            position_ids.append(length[i] + 1)
        else:
            position_ids.append(1)
    position_ids.append(len(position_ids))

    assert len(position_ids) == max_node + 2  # cls and seq
    position_ids = position_ids
    return edge_index, edge_types, position_ids


def generate_ref_edge(line, tokenizer, max_seq_length):
    graph_ref1 = line['graph_ref1']
    graph_ref2 = line['graph_ref2']
    ref1_graph_string_tok = tokenizer.tokenize("[CLS] " + graph_ref1["amr_simple"] + " [SEP]")
    ref2_graph_string_tok = tokenizer.tokenize("[CLS] " + graph_ref2["amr_simple"] + " [SEP]")
    ref1_triples_graph = align_triples(ref1_graph_string_tok,
                                       "[CLS] " + graph_ref1["amr_simple"] + " [SEP]",
                                       json.loads(graph_ref1["triples"]), 1)
    ref2_triples_graph = align_triples(ref2_graph_string_tok,
                                       "[CLS] " + graph_ref2["amr_simple"] + " [SEP]",
                                       json.loads(graph_ref2["triples"]), 1)

    ref1_edge_index, ref1_edge_types, ref1_position_ids = generate_edge_tensors(ref1_triples_graph, max_seq_length)
    ref2_edge_index, ref2_edge_types, ref2_position_ids = generate_edge_tensors(ref2_triples_graph, max_seq_length)
    graph_edge_index = [ref1_edge_index, ref2_edge_index]
    graph_edge_type = [ref1_edge_types, ref2_edge_types]

    return graph_edge_index, graph_edge_type, [ref1_position_ids, ref2_position_ids]


def generate_camr_edge(line, tokenizer, max_seq_length):
    graph_string_tok = tokenizer.tokenize("[CLS] " + line["amr_simple"] + " [SEP]")
    try:
        triples_graph = align_triples(graph_string_tok,
                                      "[CLS] " + line["amr_simple"] + " [SEP]",
                                      json.loads(line["triples"]), 1)
        edge_index, edge_types, position_ids = generate_edge_tensors(triples_graph, max_seq_length)
        graph_edge_index = [edge_index, edge_index]
        graph_edge_type = [edge_types, edge_types]
        graph_pos_ids = [position_ids, position_ids]
    except:
        return [], [], []

    return graph_edge_index, graph_edge_type, graph_pos_ids


def generate_wiki_edge(graph_triples, max_seq_length):
    edge_index, edge_types, position_ids = generate_edge_tensors(graph_triples, max_seq_length)
    return [edge_index, edge_index], [edge_types, edge_types], [position_ids, position_ids]


def generate_wiki_aug_edge(graph_triples, positive_graph_triples, max_seq_length):
    edge_index, edge_types, position_ids = generate_edge_tensors(graph_triples, max_seq_length)
    pos_edge_index, pos_edge_types, pos_position_ids = generate_edge_tensors(positive_graph_triples, max_seq_length)
    return [edge_index, pos_edge_index], [edge_types, pos_edge_types], [position_ids, pos_position_ids]


def align_wiki_triples(line, tokenizer):
    linearized_graph = line['amr_simple']
    graph_string_tok = tokenizer.tokenize("[CLS] " + linearized_graph + " [SEP]")
    tok_map = {}
    count = 0
    graph_string = "[CLS] " + linearized_graph + " [SEP]"
    for idx, item in enumerate(graph_string.split()):
        sub_len = len(tokenizer.tokenize(item))
        sub_list = list(range(count, count + sub_len))
        tok_map[idx] = sub_list
        count += sub_len
    graph_triples = align_triples(graph_string_tok, graph_string, json.loads(line["triples"]), 1, map=tok_map)
    return graph_triples


def align_wiki_aug_triples(linearized_graph, triples, tokenizer):
    graph_string_tok = tokenizer.tokenize("[CLS] " + linearized_graph + " [SEP]")
    tok_map = {}
    count = 0
    graph_string = "[CLS] " + linearized_graph + " [SEP]"
    for idx, item in enumerate(graph_string.split()):
        sub_len = len(tokenizer.tokenize(item))
        sub_list = list(range(count, count + sub_len))
        tok_map[idx] = sub_list
        count += sub_len
    graph_triples = align_triples(graph_string_tok, graph_string, triples, 1, map=tok_map)
    return graph_triples


def aligh_wiki_amr(wikipedia_dataset_path, tokenizer):
    new_line = []
    with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
        for line in tqdm(fIn):
            line = json.loads(line)
            linearized_graph = line['amr_simple']
            triples = json.loads(line["triples"])
            graph_triples = align_wiki_aug_triples(linearized_graph, triples, tokenizer)
            line["aligned_triples"] = graph_triples

            positive_linearized_graph = line['positive_amr']
            positive_triples = json.loads(line['positive_triples'])
            positive_graph_triples = align_wiki_aug_triples(positive_linearized_graph, positive_triples, tokenizer)
            line["positive_aligned_triples"] = positive_graph_triples
            new_line.append(line)

    if len(new_line) > 0:
        output_file = wikipedia_dataset_path.replace('.json', '_align.json')
        print(output_file)
        with open(output_file, 'w', encoding="utf-8") as fd:
            for example in new_line:
                fd.write(json.dumps(example, ensure_ascii=False) + "\n")
