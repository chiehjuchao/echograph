EchoGraph

Paper (MedRxiv): https://www.medrxiv.org/content/10.1101/2025.05.07.25327158v1 

=========
Requirements:

python_requires='>=3.8,<3.11

```
'torch==2.2.1'
'transformers==4.39.0'
"appdirs"
'jsonpickle'
'filelock'
'h5py'
'spacy'
'nltk'
'pytest'
```
Testing:
```python
pytest
```

EchoGraph for Annotations:
```python
from radgraph import RadGraph, F1RadGraph
model_type= "echograph"
radgraph = RadGraph(model_type=model_type)
annotations = radgraph(["left ventricular systolic function is normal"])
```


EchoGraph for F1-style reward score:
```python
from radgraph import F1RadGraph
refs = ["left ventricular systolic function is normal",
        "right ventricular enlargement with reduced systolic function",
]

hyps = ["left ventricular systolic function is abnormal",
        "normal right ventricular size and systolic function",
]
f1radgraph = F1RadGraph(model_type=model_type, reward_level="all")
mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)

```

Cite the original RadGraph package as per:

```bibtex
@inproceedings{delbrouck-etal-2024-radgraph,
    title = "{R}ad{G}raph-{XL}: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports",
    author = "Delbrouck, Jean-Benoit  and
      Chambon, Pierre  and
      Chen, Zhihong  and
      Varma, Maya  and
      Johnston, Andrew  and
      Blankemeier, Louis  and
      Van Veen, Dave  and
      Bui, Tan  and
      Truong, Steven  and
      Langlotz, Curtis",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.765",
    pages = "12902--12915",
    }
```
