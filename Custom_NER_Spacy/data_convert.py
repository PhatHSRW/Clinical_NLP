import re
import glob
import json
import argparse


def create_dict_concept_type(concept_file_path):
    dict_concept_type = {}
    with open(concept_file_path, "r") as con:
        concepts = con.readlines()
        for concept in concepts:
            # list_word = concept.split("\"")
            list_word = re.split(r'c=|\|\|t=|\n', concept)
            # list_word_filter = [''.join([char for char in x if not char.isdigit() and char != ":"]).strip()[1:-1] for x in list_word if x.strip()]
            list_word_filter = [re.sub(r'\b\d+:\d+\b', '', x).strip()[1:-1] for x in list_word if x.strip()]
            dict_concept_type[list_word_filter[0]] = list_word_filter[-1]
        # if concept_file_path.endswith("124.con"):
        #     print(dict_concept_type)
    return dict_concept_type

def find_index(text, word):
    operator_list = ("*", "+")
    text = text.lower()
    result = []
    matches = None
    print(word)
    # use regex
    if word.endswith(operator_list):
        matches = re.finditer(rf'{word}[+]', text)
    elif word.startswith(operator_list):
        matches = re.finditer(rf'[+]{word}', text)
    elif word.startswith('"'):
        matches = re.finditer(rf'{word}', text)
    elif "+" not in word:
        matches = re.finditer(rf"\b{word}\b", text)
    if matches:
    # matches = re.finditer(r'\b|[+*]{}|[+*]\b'.format(word), text)
        for match in matches:
            result.append([match.start(), match.end()])
    
    if "+" in word[1:-1]:
        index = text.find(word)
        result.append([index, index+len(word)])
    return result
    
    # use find
    # index = text.find(word)
    # while index != -1:
    #     result.append([index,index+len(word)+1])
    #     index = text.find("1+", index + 1)
    # return result
        
def convert(raw_data_path, spacy_data_path, name_dataset):
    if name_dataset == "test":
        assert "test" in raw_data_path, "choose name_dataset='test' if raw_data_path is test" 
        text_files = glob.glob(raw_data_path+"/txt/*")
        concept_files_path = raw_data_path+"/concept/"
        
    else:
        assert "training"  in raw_data_path, "choose name_dataset='beth' or 'partners' if raw_data_path is training"
        text_files = glob.glob(raw_data_path+"/"+name_dataset+"/txt/*")
        concept_files_path = raw_data_path+"/"+name_dataset+"/concept/"
        
    
    training_data = {'classes' : ['TEST', "TREATMENT", "PROBLEM"], 'annotations' : []}


    for filename in text_files:
        temp_dict = {}
        with open(filename, "r") as file:
            text = file.read()
            # lines = file.readlines() 
        temp_dict['entities'] = []
        temp_dict['text'] = text
        temp_dict["file_name"] = filename.split("/")[-1]

        concept_filename = filename.split("/")[-1].replace("txt","con")
        concept_path = concept_files_path + concept_filename
        dict_concept_type = create_dict_concept_type(concept_path)
        print("==========="+filename+"===========")
        # print(dict_concept_type)
        
        for concept, _type in dict_concept_type.items():
            length_word = len(concept)+1
            indexes = find_index(text,concept)
            # if temp_dict["file_name"] == "record-142.txt" and "+" in concept:
            #     print(concept)
            #     print(indexes)    
            for index in indexes:
                temp_dict["entities"].append((index[0],index[1]+1,_type.upper()))
        training_data["annotations"].append(temp_dict)
        # print(temp_dict["entities"])
            
    with open(spacy_data_path+"/"+name_dataset+"_data.json", "w") as file:
        json.dump(training_data, file, indent=4)
        print("DONE JSON")

def main():
    parser = argparse.ArgumentParser(description ='Convert dataset i2b2 to Spacy style')
    parser.add_argument('-d', '--data', help ='path to raw dataset folder, i.e concept_assertion_relation_training_data or reference_standard_for_test_data')
    parser.add_argument('-r', '--result', help ='path to processed dataset folder')
    parser.add_argument('-n', '--name', help ='name of dataset', choices=['beth', 'partners', 'test'])
    args = parser.parse_args()
    
    data_name = "test"    
    convert(args.data, args.result, args.name)

if __name__ == "__main__":
    main()


# #TESTING ALGORITHM
# print("TESTING ALGORITHM")
# # JSON file
# f = open(data_name+"_data.json", "r")
# # Reading from file
# training_data = json.loads(f.read())
# # print(training_data["annotations"][0]["file_name"])
# for annot in training_data["annotations"]:
#     if annot["file_name"] == "record-142.txt":
#         print(len(annot["entities"]))
#         for entity in annot["entities"]:
#             print(annot["text"][entity[0]:entity[1]-1]+"|"+ entity[-1]+"|"+str(entity))
       
