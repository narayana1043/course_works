"""
----------------------
IF THERE IS AN TIE BETWEEN ANY TWO CLASSES THE MAX FUNCTION RESOLVES THE TIE AUTOMATICALLY.
----------------------
"""

from operator import itemgetter
import pandas as pd
from sklearn.metrics import accuracy_score

def class_setter(input1,input2):
    if bool(input1 != input2):
        return "+"
    else:
        return "-"

def data_gen():
    dataframe = pd.DataFrame(columns=["input1","input2","class_label"])
    len = 0
    for input1 in range(2):
        for input2 in range(2):
            dataframe.loc[len] = [input1,input2,class_setter(input1,input2)]
            len += 1
    #print(dataframe)
    return dataframe

def prob(column_value,label_value):
    return training_data[training_data[column_value] == label_value][column_value].count()/training_data["class_label"].count()

def prod_conditional_probabilities(record, class_label):
    product = 1
    for attribute in range(len(record)-1):
        # print(record[attribute])
        # print(training_data[training_data.columns[attribute]])
        # print(training_data[training_data.columns[attribute]] == record[attribute])
        # print(training_data[(training_data[training_data.columns[attribute]] == record[attribute]) & (training_data["class_label"] == class_label)]["class_label"].count())
        product *= training_data[(training_data[training_data.columns[attribute]] == record[attribute]) & (training_data["class_label"] == class_label)]["class_label"].count()/training_data[training_data["class_label"] == class_label]["class_label"].count()

    return product

def posterior_prob_cal(class_label,record):
    posterior_prob = prob("class_label",class_label) * prod_conditional_probabilities(record,class_label)
    return posterior_prob

def input_space_transformation(sample_training_data):
    columns_names = [("input",index) for index in range(1,7)]+["class_label"]
    input_space = pd.DataFrame(columns=columns_names)
    len = 0
    for index,record in sample_training_data.iterrows():
        input_space.loc[len] = [1,record[0],record[1],record[0]*record[1],(record[0]**2),(record[1]**2),record[2]]
        len += 1
    return input_space

def navie_bayes_classifier():
    sample_training_data = data_gen()
    total_tested = []
    total_classified = []
    global training_data
    training_data = input_space_transformation(sample_training_data)
    for index,record in training_data.iterrows():
        total_tested.append(record[-1])
        posterior_prob_class_dict = {}
        for class_label in ["+","-"]:
            posterior_prob_class_dict[class_label] = posterior_prob_cal(class_label,record)
            print("Posterior Probability for the record ",record[0],",",record[1],",",record[2],",",record[3],",",record[4],",",record[5],
                  " given class is ",class_label," is ",posterior_prob_class_dict[class_label])
        sum_values = sum(posterior_prob_class_dict.values())
        posterior_prob_class_dict.update((key, value/sum_values) for key,value in posterior_prob_class_dict.items())
        classified_class = max(posterior_prob_class_dict.items(), key=itemgetter(1))[0]
        total_classified.append(classified_class)
        print("The record ",record[0]," ",record[1]," is classifed as", classified_class)
    print("\n \nAccuracy: ",accuracy_score(total_classified,total_tested))


navie_bayes_classifier()

