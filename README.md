# deepsrl
deepsrl is an open source python-based package for semantic roles extraction, which is built on [Deep Semantic Role Labeling: What works and what's next](https://kentonl.com/pub/hllz-acl.2017.pdf).

### Installation
To pip install `deepsrl` from github:
```
pip install git+ssh://git@github.com/YoussefYN/deepsrl.git
```

### Minimal example
`deepsrl` provide a general model to extract semantic roles.
The model object receive in constrcutor three parameters: 
(1) srl model path (2) propid model path (3) directory path of glove embeddings.
```
from deepsrl.model import DeepSrl

model = DeepSrl("core/conll05_model",
                "core/conll05_propid_model",
                "data/glove")
model.sr_predict("Those who dare to fail miserably can achieve greatly.")
# return:
#[
#  {'A1': 'Those', 'R-A1': 'who', 'V': 'dare', 'A2': 'to fail miserably', 'v_index': 2},
#  {'A1': 'Those', 'R-A1': 'who', 'V': 'fail', 'AM-MNR': 'miserably', 'v_index': 4}, 
#  {'A0': 'Those who dare to fail miserably', 'AM-MOD': 'can', 'V': 'achieve', 'AM-MNR': 'greatly', 'v_index': 7}
#]
```

### Model files:
Here you can find the repository of the paper [here](https://github.com/luheng/deep_srl).
Model files could be found [here](https://github.com/luheng/deep_srl/tree/master/resources).
Glove embeddings could be downloaded through this [link](http://nlp.stanford.edu/data/glove.6B.zip).