'''
Code Adapted from Source: https://github.com/ncbi-nlp/BLUE_Benchmark

Convert raw dataset into dataset that BERT can read.

'''

import csv
import itertools
import os
import re
from pathlib import Path
from typing import Match

import bioc
import fire
import pandas as pd
import tqdm

'''
**Relations Labels**

PIP: Two problems are related to each other.
TeCP: A test was performed to investigate a medical problem.
TeRP: A test has revealed some medical problem.
TrAP: A treatment administered for a medical problem.
TrCP: A treatment caused a medical problem.
TrIP: A certain treatment has improved or cured a medical problem.
TrNAP: The administration of a treatment was avoided because of a medical problem.
TrWP: A patient’s medical problem has deteriorated or worsened because of or in spite of a treatment being administered.

'''

re_labels = ['PIP', 'TeCP', 'TeRP', 'TrAP','TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']

def debug(sentences, entities, id1, id2):
    e1 = None
    e2 = None
    for e in entities:
        if e['id'] == id1:
            e1 = e
        if e['id'] == id2:
            e2 = e
    assert e1 is not None and e2 is not None
    ss = [s for s in sentences
          if s.offset <= e1['start'] <= s.offset + len(s.text)
          or s.offset <= e2['start'] <= s.offset + len(s.text)]
    if len(ss) != 0:
        for s in ss:
            print(s.offset, s.text)
    else:
        for s in sentences:
            print(s.offset, s.text)


def entity_replace(text, offset, annotation_1, annotation_2):
    annot_1_start = annotation_1['start'] - offset
    annot_2_start = annotation_2['start'] - offset
    annot_1_end = annotation_1['end'] - offset
    annot_2_end = annotation_2['end'] - offset

    if annot_1_start <= annot_2_start <= annot_1_end \
            or annot_1_start <= annot_2_end <= annot_1_end \
            or annot_2_start <= annot_1_start <= annot_2_end \
            or annot_2_start <= annot_1_end <= annot_2_end:
        start = min(annot_1_start, annot_2_start)
        end = max(annot_1_end, annot_2_end)
        before = text[:start]
        after = text[end:]
        return before + f'@{annotation_1["type"]}-{annotation_2["type"]}$' + after

    if annot_1_start > annot_2_start:
        before = text[:annot_2_start]
        middle = text[annot_2_end:annot_1_start]
        after = text[annot_1_end:]
        return before + f'@{annotation_2["type"]}$' + middle + f'@{annotation_1["type"]}$' + after
    else:
        before = text[:annot_1_start]
        middle = text[annot_1_end:annot_2_start]
        after = text[annot_2_end:]
        return before + f'@{annotation_1["type"]}$' + middle + f'@{annotation_2["type"]}$' + after

def read_file(path_loc):
    with open(path_loc) as file:
        text = file.read()
    sentences = []
    offset = 0
    for sent in text.split('\n'):
        sentence = bioc.BioCSentence()
        sentence.infons['filename'] = path_loc.stem
        sentence.offset = offset
        sentence.text = sent
        sentences.append(sentence)
        index = 0
        for m in re.finditer('\S+', sent):
            if index == 0 and m.start() != 0:
                annotation = bioc.BioCAnnotation()
                annotation.id = f'a{index}'
                annotation.text = ''
                annotation.add_location(bioc.BioCLocation(offset, 0))
                sentence.add_annotation(annotation)
                index += 1
            annotation = bioc.BioCAnnotation()
            annotation.id = f'a{index}'
            annotation.text = m.group()
            annotation.add_location(bioc.BioCLocation(
                m.start() + offset, len(m.group())))
            sentence.add_annotation(annotation)
            index += 1
        offset += len(sent) + 1
    return sentences

def annotation_offset(sentences, match_obj: Match,
                    start_line_group, start_token_group,
                    end_line_group, end_token_group,
                    text_group):
    assert match_obj.group(start_line_group) == match_obj.group(end_line_group)
    sentence = sentences[int(match_obj.group(start_line_group)) - 1]

    start_token_index = int(match_obj.group(start_token_group))
    end_token_index = int(match_obj.group(end_token_group))
    
    start = sentence.annotations[start_token_index].total_span.offset
    end = sentence.annotations[end_token_index].total_span.end
    text = match_obj.group(text_group)

    actual = sentence.text[start -
                           sentence.offset:end - sentence.offset].lower()
    expected = text.lower()
    assert actual == expected, 'Cannot match at %s:\n%s\n%s\nFind: %r, Matched: %r' \
                               % (
                                   sentence.infons['filename'], sentence.text, match_obj.string, actual,
                                   expected)
    return start, end, text

def annotations(path_loc, sentences):
    annotations = []
    pattern = re.compile(
        r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"(\|\|a="(.*?)")?')
    with open(path_loc) as file:
        for index, line in enumerate(file):
            line = line.strip()
            m = pattern.match(line)
            assert m is not None

            start, end, text = annotation_offset(sentences, m, 2, 3, 4, 5, 1)
            annotation = {
                'start': start,
                'end': end,
                'type': m.group(6),
                'a': m.group(7),
                'text': text,
                'line': int(m.group(2)) - 1,
                'id': f'{path_loc.name}.l{index}'
            }
            if len(m.groups()) == 9:
                annotation['a'] = m.group(8)
            annotations.append(annotation)
    return annotations

def find_annotations(annotations, start, end):
    for annotation in annotations:
        if annotation['start'] == start and annotation['end'] == end:
            return annotation
    raise ValueError


def read_relations(path_loc, sentences, concepts):
    pattern = re.compile(
        r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|r="(.*?)"\|\|c="(.*?)" (\d+):(\d+) (\d+):(\d+)')

    relations = []
    with open(path_loc) as file:
        for line in file:
            line = line.strip()
            m = pattern.match(line)
            assert m is not None

            start, end, text = annotation_offset(sentences, m, 2, 3, 4, 5, 1)
            annotation_1 = find_annotations(concepts, start, end)
            start, end, text = annotation_offset(sentences, m, 8, 9, 10, 11, 7)
            annotation_2 = find_annotations(concepts, start, end)
            relations.append({
                'docid': path_loc.stem,
                'label': m.group(6),
                'Arg1': annotation_1['id'],
                'Arg2': annotation_2['id'],
                'string': line
            })
    return relations

def find_relations(relations, annotation_1, annotation_2):
    re_labels = []
    for index in range(len(relations) - 1, -1, -1):
        rel = relations[index]
        if (rel['Arg1'] == annotation_1['id'] and rel['Arg2'] == annotation_2['id']) \
                or (rel['Arg1'] == annotation_2['id'] and rel['Arg2'] == annotation_1['id']):
            del relations[index]
            re_labels.append(rel['label'])
    return re_labels

def convert(main_dir, dest):
    file = open(dest, 'w')
    writer = csv.writer(file, delimiter='\t', lineterminator='\n')
    writer.writerow(['index', 'sentence_mask', 'sentence_origin','label'])
    with os.scandir(main_dir / 'txt') as it:
        for entry in tqdm.tqdm(it):
            if not entry.name.endswith('.txt'):
                continue
            text_path = Path(entry.path)
            docid = text_path.stem

            sentences = read_file(text_path)
            # read assertions
            concepts = annotations(main_dir / 'concept' / f'{text_path.stem}.con',
                                    sentences)
            # read relations
            relations = read_relations(main_dir / 'rel' / f'{text_path.stem}.rel',
                                       sentences, concepts)
            for index, (concept1, concept2) in enumerate(itertools.combinations(concepts, 2)):
                if concept1['line'] != concept2['line']:
                    continue

                sentence = sentences[concept1['line']]
                text = entity_replace(sentence.text, sentence.offset, concept1, concept2)
                re_labels = find_relations(relations, concept1, concept2)
                if len(re_labels) == 0:
                    writer.writerow(
                        [f'{docid}.{concept1["id"]}.{concept2["id"]}', text, sentence.text, 'false'])
                else:
                    for l in re_labels:
                        writer.writerow(
                            [f'{docid}.{concept1["id"]}.{concept2["id"]}', text,sentence.text, l])

            if len(relations) != 0:
                for rel in relations:
                    print(rel['string'])
                    debug(sentences, concepts, rel['Arg1'], rel['Arg2'])
                    print('-' * 80)
    file.close()

def doc_split(train1, train2, dev_docids, output_dir):
    train1_df = pd.read_csv(train1, sep='\t')
    train2_df = pd.read_csv(train2, sep='\t')
    train_df = pd.concat([train1_df, train2_df])

    with open(dev_docids) as file:
        dev_docids = file.readlines()

    with open(output_dir / 'train.tsv', 'w') as tfile, open(output_dir / 'dev.tsv', 'w') as dfile:
        twriter = csv.writer(tfile, delimiter='\t', lineterminator='\n')
        twriter.writerow(['index', 'sentence_mask', 'sentence_origin','label'])
        dwriter = csv.writer(dfile, delimiter='\t', lineterminator='\n')
        dwriter.writerow(['index', 'sentence_mask', 'sentence_origin','label'])
        for index, row in train_df.iterrows():
            if row[0][:row[0].find('.')] in dev_docids:
                dwriter.writerow(row)
            else:
                twriter.writerow(row)

def create_i2b2_dataset(input_directory, output_directory):
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    convert(input_path / 'reference_standard_for_test_data',
            output_path / 'test.tsv')
    convert(input_path / 'concept_assertion_relation_training_data/beth',
            output_path / 'train-beth.tsv')
    convert(input_path / 'concept_assertion_relation_training_data/partners',
            output_path / 'train-partners.tsv')
    doc_split(output_path / 'train-beth.tsv',
              output_path / 'train-partners.tsv',
              input_path / 'dev-docids.txt',
              output_path)

if __name__ == '__main__':
    # Run Code by: python dataset_prep_i2b2.py dataset_raw/ dataset_prepared/
    fire.Fire(create_i2b2_dataset)