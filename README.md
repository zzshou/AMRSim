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
Download model from the [link](https://drive.google.com/file/d/1klTrvv3hpIPxaCoMbRI7IJDme-Vq3UPS/view?usp=share_link)
unzip to the output directory
```
cd sentence_transformers
python test_amrsim.py
```

### Training
Download wiki data from the [link]()
```
python train_stsb_ct_amr.py
```

Thanks:
fact-graph
sentence-transformer