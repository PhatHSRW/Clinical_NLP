import pandas as pd
import csv
import argparse
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description ='Evaluate prediction on test dataset.')

parser.add_argument('-d', '--data', help ='path to dataset file .tsv')
parser.add_argument('-r', '--result', help ='path to result file .tsv')
args = parser.parse_args()


header_result = ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']
header_dataset = ['index', 'sentence_mask', 'sentence_origin','label']

resultfile_path = args.result
datafile_path = args.data

test_result = pd.read_csv(resultfile_path, delimiter='\t', header=None)
if test_result.columns[0] is not str:
    test_result = pd.read_csv(resultfile_path, delimiter='\t', header=None, names=header_result)

empty_row = pd.DataFrame([[None for _ in range(len(test_result.columns))]], columns=test_result.columns)
test_result = pd.concat([empty_row, test_result], ignore_index=True)


dataset = pd.read_csv(datafile_path, delimiter='\t', header=None)
if dataset.columns[0] is not str:
    dataset = pd.read_csv(datafile_path, delimiter='\t', header=None, names=header_dataset)
dataset.loc[0] = [None]*len(dataset.columns)


y_true = []
y_pred = []
correct = 0
for i, (result, gt) in enumerate(zip(test_result.values[1:],dataset.values[1:])):
    high_prob = result.argmax()
    y_pred.append(high_prob)
    
    label = gt[-1]
    index_label = next(i for i,x in enumerate(header_result) if label.lower() == x.lower())
    y_true.append(index_label)
    

acc = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
precision = precision_score(y_true, y_pred, average="macro")

print("F1-score:", f1)
print("Accuracy: ", acc)
print("Recall: ", recall)
print("Precision: ", precision)
confusion_matrix = confusion_matrix(y_true, y_pred)
print(confusion_matrix)
plt.imshow(confusion_matrix)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# acc:  0.9523809523809523