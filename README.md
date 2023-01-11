<<<<<<< HEAD
# NLP Project: NAMED ENTITY RECOGNITION (NER) AND RELATION EXTRACTION FROM CLINICAL TEXTS

## NER using Spacy


=======
# NLP Project: NAMED ENTITY RECOGNITION AND RELATION EXTRACTION FROM CLINICAL TEXTS
>>>>>>> 96c7e933b5e48a164cdce9474505c2ae3c89e934

## i2b2 Dataset Preparation using BLUE(Biomedical Language Understanding Evaluation) for BioBERT

### Introduction

BLUE benchmark consists of five different biomedicine text-mining tasks with ten corpora.
Here, we rely on preexisting datasets because they have been widely used by the BioNLP community as shared tasks.
These tasks cover a diverse range of text genres (biomedical literature and clinical notes), dataset sizes, and degrees of difficulty and, more importantly, highlight common biomedicine text-mining challenges.

## Dataset

| Corpus          | Train |  Dev | Test | Task                    | Metrics             | Domain     |
|-----------------|------:|-----:|-----:|-------------------------|---------------------|------------|
| i2b2-2010       |  3110 |   11 | 6293 | Relation extraction     | F1                  | Clinical   |

## How to Run Dataset Preparation Code for i2b2

```bash
$ python dataset_prep_i2b2.py dataset_raw/ dataset_prepared/
```