# NLP Project: NAMED ENTITY RECOGNITION (NER) AND RELATION EXTRACTION FROM CLINICAL TEXTS

## Dataset Introduction
The dataset used in this project is i2b2-2010. However we have to process the raw data in both tasks.

Link dataset: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
Source: https://academic.oup.com/jamia/article/18/5/552/830538


## NER using Spacy

- Process i2b2 dataset to be compatible with Spacy \
Running the data_convert.py to convert raw dataset to spacy format dataset (below) and save it as json files.
  ```
    "classes": [
        "TEST",
        "TREATMENT",
        "PROBLEM"
    ],
    "annotations": []
  ```
  ```
  $ cd Custom_Ner_Spacy
  $ python data_convert.py --data [raw_data_path] --result [spacy_data_save_path] --name [name_dataset]
  ```
- Run clinical_ner.ipynb: Please remember to change the data file path in clinical_ner notebook as you did in the previous step.

## Relation Extraction using BlueBERT benchmark

### Task
BLUE(Biomedical Language Understanding Evaluation) for BioBERT

| Corpus          | Train |  Dev | Test | Task                    | Metrics             | Domain     |
|-----------------|------:|-----:|-----:|-------------------------|---------------------|------------|
| i2b2-2010       |  3110 |   11 | 6293 | Relation extraction     | F1                  | Clinical   |

### Pre-trained models and benchmark datasets
- Pre-trained models consisting of BlueBERT weights, vocab, and config files can be found in bluebert_dir/README.txt

### Install Bert package
- After install packages in requirements.txt, please install the project in edit mode:
  ```
  $ cd Relation_Extraction_blueBERT
  $ python setup.py develop
  ```
### Process dataset
- Running the code to convert i2b2 dataset to bert dataset:
  ```
  $ cd dataset
  $ python dataset_prep_i2b2.py dataset_raw/ dataset_prepared/
  ```

### Fine-tuning
- Fine-tuning BlueBERT for relation extraction:
  ```bash
  python bluebert/run_bluebert.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --task_name="i2b2" \
  --vocab_file=bluebert_dir/[...]/vocab.txt \
  --bert_config_file=bluebert_dir/[...]/bert_config.json \
  --init_checkpoint=bluebert_dir/[...]/bert_model.ckpt \
  --num_train_epochs=10.0 \
  --data_dir=dataset/dataset_prepared \
  --output_dir=output_dir \
  --do_lower_case=true \
  --train_batch_size=64 \
  --predict_batch_size=32 \ 
  ```

### Evaluate

```bash
$ python evaluate_prediction.py --data [path to data file test.tsv] --result [path to prediction file result.tsv]
```
