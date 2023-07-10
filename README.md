# AMRSim: Evaluate AMR Graph Similarity via Self-supervised Learning

This repository contains the code for our ACL-2023 paper: [Evaluate AMR Graph Similarity via Self-supervised Learning](https://aclanthology.org/2023.acl-long.892/)


## Requirements
```
pip install -r requirements.txt
```

## Calculate similarity of graphs
### Preprocess
Transfer amr graphs to linearized string
```
python amr2json.py
```

input "src.amr" and "tgt.amr"
output "src_tgt.json"

### Predicting
```
cd sentence_transformers
python test_amrsim.py
```

### Training
wiki data


code citation:
fact-graph
sentence-transformer