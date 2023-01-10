import srsly
import json
import typer
import warnings
from pathlib import Path
import spacy
import numpy
import os
import pandas as pd

from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from spacy.training import offsets_to_biluo_tags


def _load_data(file_path):
    samples, entities_count = [], 0
    for line in srsly.read_jsonl(file_path):
        sample = {
            "text": line["text"],
            "entities": []
        }
        if "spans" in line.keys():
            entities = [(s["start"], s["end"], s["label"]) for s in line["spans"]]
            sample["entities"] = entities
            entities_count += len(entities)
        else:
            warnings.warn("Sample without entities!")
        samples.append(sample)
    return samples, entities_count


def _get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label


def _create_total_target_vector(nlp, samples):
    target_vector = []
    for sample in samples:
        doc = nlp.make_doc(sample["text"])
        ents = sample["entities"]
        bilou_ents = offsets_to_biluo_tags(doc, ents)
        vec = [_get_cleaned_label(label) for label in bilou_ents]
        target_vector.extend(vec)
    return target_vector


def _get_all_ner_predictions(nlp, text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def _create_prediction_vector(nlp, text):
    return [_get_cleaned_label(prediction) for prediction in _get_all_ner_predictions(nlp, text)]


def _create_total_prediction_vector(nlp, samples):
    prediction_vector = []
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        prediction_vector.extend(_create_prediction_vector(nlp, sample["text"]))
    return prediction_vector


def _plot_confusion_matrix(cm, classes, normalize=False, text=True, cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = "Confusion Matrix"

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if text:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, pyplot


def get_confusion_matrix(model_path: Path, output_dir: Path):
    #spacy.prefer_gpu()
    nlp = spacy.load(model_path)
    print(f"Loaded SpaCy pipeline.")
    with open('datasets/test_data.json', 'r') as f:
        data = json.load(f)
    test_data = {'classes': ["TEST", "TREATMENT", "PROBLEM"], 'annotations': []}
    entities_count = 0
    for ann in data['annotations']:
        temp_dict = {}
        temp_dict['text'] = ann['text']
        temp_dict['entities'] = []
        for entity in ann['entities']:
            temp_dict['entities'].append((entity[0],entity[1],entity[2]))
        entities_count = entities_count + len(temp_dict['entities'])
        test_data['annotations'].append(temp_dict)
    samples, entities_count = test_data['annotations'],entities_count
    print(f"Loaded {len(samples)} samples including {entities_count} entities.")
    classes = sorted(set(_create_total_target_vector(nlp, samples)))
    print(f"Identified {len(classes)} classes: {', '.join(classes)}")
    y_true = _create_total_target_vector(nlp, samples)
    print("Computed target vector!")
    print("Computing prediction vector...")
    y_pred = _create_total_prediction_vector(nlp, samples)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print("Generated confusion matrix!")
    cm_df = pd.DataFrame(matrix, columns=classes)
    cm_df.insert(0, "TARGETS", classes)
    ax, plot = _plot_confusion_matrix(matrix, classes, normalize=True, text=False)
    print("Plotted confusion matrix!")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving rendered image to: {output_dir}/confusion.png")
    pyplot.savefig(f"{output_dir}/confusion.png")
    print(f"Saving confusion matrix data to: {output_dir}/confusion.csv")
    cm_df.to_csv(f"{output_dir}/confusion.csv")
    print("Finished!")


if __name__ == "__main__":
    typer.run(get_confusion_matrix)