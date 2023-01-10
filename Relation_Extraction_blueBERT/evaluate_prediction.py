import pandas as pd
import csv

header_result = ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']
header_dataset = ['index', 'sentence', 'label']

resultfile_path = "/home/phat/nlp/bluebert/zzzzzz/test_results_3.tsv"
datafile_path = "/home/phat/nlp/bluebert/zzzzzz/test_dataset.tsv"

test_result = pd.read_csv(resultfile_path, delimiter='\t', header=None)
if test_result.columns[0] is not str:
    test_result = pd.read_csv(resultfile_path, delimiter='\t', header=None, names=header_result)

empty_row = pd.DataFrame([[None for _ in range(len(test_result.columns))]], columns=test_result.columns)
test_result = pd.concat([empty_row, test_result], ignore_index=True)


dataset = pd.read_csv(datafile_path, delimiter='\t', header=None)
if dataset.columns[0] is not str:
    dataset = pd.read_csv(datafile_path, delimiter='\t', header=None, names=header_dataset)
dataset.loc[0] = [None]*len(dataset.columns)

# print(test_result[0:10])
# print('-----------'*5)
# print(dataset[0:10])

correct = 0
for i, (result, gt) in enumerate(zip(test_result.values[1:],dataset.values[1:])):
    high_prob = result.argmax()
    pred = header_result[high_prob]
    label = gt[-1]
    # print(pred, label)
    if pred.lower() == label.lower():
        correct+=1
    # if i > 10:
    #     print(correct)
    #     break

accuracy = correct/i
print(accuracy)