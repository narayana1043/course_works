from math import log10
from operator import itemgetter
import random
from copy import deepcopy as dc
from statistics import mode
from statistics import mean

def fileRead(file_name):
    if file_name[1] == 'comma':
        file_data_lines = open(file_name[0],'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split(','))
    elif file_name[1] == 'tab':
        file_data_lines = open(file_name[0],'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split('\t'))
    elif file_name[1] == 'space':
        file_data_lines = open(file_name[0],'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split(' '))
    return data_list

def breakPointsGen(list):
    if len(set(list)) != 1:
        max_point = max(list)
        min_point = min(list)
        splits = 5
        #if len(list) == 150:
            #splits = 5
        dist_bw_points = (max_point - min_point)/splits
        break_points = []
        point = min(list)
        while True:
            point += dist_bw_points
            if point < max_point:
                break_points.append(point)
            else:
                break
        return break_points
    else:
        return set(list)

def countOfValuesOfSameClass(list,class_column,class_values):
    value_dict = {}
    count_value_dict = {}
    for value in class_values:
        value_dict[value] = []
        count_value_dict[value] = []
    for data in list:
        for value in class_values:
            if data[class_column] == value:
                value_dict[value].append(data)
    for value in class_values:
        count_value_dict[value]=len(value_dict[value])
    return count_value_dict

def splitWeightFind(count_value_dict,class_values):
    values_list=[]
    split_weight = 0
    sum_values = 0
    if formula == 1:
        split_weight = 1
    for value in class_values:
        values_list.append(count_value_dict[value])
    sum_values = sum(values_list)
    for value in class_values:
        if count_value_dict[value] != 0 and formula == 0:
            split_weight -= (count_value_dict[value]/sum_values)*log10(count_value_dict[value]/sum_values)
        elif count_value_dict[value] != 0 and formula == 1:
            split_weight -= (count_value_dict[value]/sum_values)*log10((count_value_dict[value]/sum_values)**2)
    return (split_weight,sum_values)

def calFun(data_list,break_points,column,class_column,class_values):
    break_points_split_value_list = []
    #print(break_points)
    for value in break_points:
        left_list = []
        right_list = []
        count_value_list_dict=[]
        child_split_weight_list = []
        child_split_weight = 0
        parent_split_weight = 0
        tot_sum = 0
        for data in data_list:
            if float(data[column]) <= value:
                left_list.append(data)
            else:
                right_list.append(data)
        count_value_list_dict.append(countOfValuesOfSameClass(left_list,class_column,class_values))
        count_value_list_dict.append(countOfValuesOfSameClass(right_list,class_column,class_values))
        for count_value_dict in count_value_list_dict:
            child_split_weight_list.append(splitWeightFind(count_value_dict,class_values))
        for item in child_split_weight_list:
            tot_sum += item[1]
        for item in child_split_weight_list:
            child_split_weight += item[0]*(item[1]/tot_sum)
        parent_split_weight = splitWeightFind(countOfValuesOfSameClass(data_list,class_column,class_values),class_values)
        #print(break_points)
        #print(parent_split_weight[0] - child_split_weight)
        break_points_split_value_list.append([value,parent_split_weight[0]-child_split_weight])
    #print(break_points_split_value_list)
    return break_points_split_value_list

def setMissData(data_list,column):
    temp = []
    #print(data_list[column])
    for miss_data in range(len(data_list)):
        if data_list[miss_data][column] != '':
            temp.append(float(miss_data))
    data_list[miss_data][column] = mean(temp)
    return data_list[miss_data][column]

def splitCriteriaCal(data_list,column,class_column,class_values):
    list = []
    for data in data_list:
        if data[column] == '':
            data[column] = setMissData(data_list,column)
        list.append(float(data[column]))
    break_points = breakPointsGen(list)
    #print(break_points)
    break_points_info_gain = calFun(data_list,break_points,column,class_column,class_values)
    #print(break_points_info_gain)
    return sorted(break_points_info_gain, key=itemgetter(1),reverse=True)[0]

def splitCriteria(data_list, column_names, class_column):
    col_dict = []
    class_values = set()
    for data in data_list:
        class_values.add(data[class_column])
    for col in range(0, len(column_names)):
        if col != class_column:
            col_dict.append([splitCriteriaCal(data_list,col,class_column,set(class_values)),col])
    #print(col_dict)
    return sorted(col_dict, key= lambda x:x[0][1],reverse=True)[0]

def splitingTree(data_list, split_col, split_value):
    left_list = []
    right_list = []
    for record in data_list:
        if float(record[split_col]) <= split_value:
            left_list.append(record)
        else:
            right_list.append(record)
    #print(len(data_list))
    #print(len(left_list))
    #print(len(right_list))
    return left_list,right_list


def stopCondition(list,class_column):
    temp_list = []
    for element in list:
        #print(element)
        temp_list.append(element[class_column])
    if len(set(temp_list)) == 1:
        #print(len(list),list)
        return True
    return False

def leaf_class(list,class_column):
    max_class_list = []
    for element in list:
        max_class_list.append(element[class_column])
    return mode(max_class_list)

def buildTree(node):
    #column_names=["sepalLength", "sepalWidth", "petalLength", "petalWidth","irisClass"]
    #class_column = "irisClass"
    node.stop_condition = stopCondition(node.data,class_column)
    if node.stop_condition == True:
        node.name = leaf_class(node.data,class_column)
        node.node_type = 'Leaf'
        #print(node.node_type)
        return False
    else:
        split_data = splitCriteria(node.data[:], column_names, class_column)
        split_value = split_data[0][0]
        split_col = split_data[1]
        #print(node.name)
        node.split_column = split_col
        node.split_value = split_value
        node.name = '%f , %f'%(split_col,split_value)
        #print(node.name)
        node.left_list,node.right_list = splitingTree(node.data, split_col, split_value)
        #print(len(node.left_list),node.left_list)
        #print(len(node.right_list),node.right_list)
        if len(node.right_list) != 0:
            node.left_child=(Node(node.left_list))
            node.right_child=(Node(node.right_list))
            node.children.append(node.left_child)
            node.children.append(node.right_child)
        else:
            node.name = '%f , %f'%(split_col,split_value)
            #print(node.name,node.data)
            node.node_type = 'Leaf'
            return False                                                             #stop condition
        for i in node.children:
            #print(i)
            temp = True
            while temp:
                temp = buildTree(i)

def class_finder(element,temp):
    col = temp.split_column
    value = temp.split_value
    while(temp.node_type != 'Leaf'):
        if element[temp.split_column] == '':
            element[temp.split_column] = temp.split_value+1                     #pushing the missing value to the right is improving accuracy
        if float(element[temp.split_column]) <= temp.split_value:
            temp_obj = temp.left_child
            #print(temp.split_value)
        elif float(element[temp.split_column]) > temp.split_value:
            temp_obj = temp.right_child
            #print(temp.split_value)
        #print(temp_obj.node_type)
        if temp_obj.node_type == 'Leaf':
            return temp_obj.name
        temp = temp_obj

def accuracy(test_list,predicted_list):
    count = 0
    len_test_list = len(test_list)
    for index in range(len_test_list):
        if test_list[index][class_column] == predicted_list[index]:
            count += 1
    return (count/len(test_list)*100)


class Node(object):
    #column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth","irisClass"]
    #class_column = column_names.index("irisClass")
    def __init__(self,data):
        self.name = None
        self.node_type = None
        self.split_column = None
        self.split_value = None
        self.data = data
        self.stop_condition = None
        self.left_child = None
        self.right_child = None
        self.children = []
    def add_child(self,obj):
        self.children.append(obj)

def startSplitData(data_list):
    random_list = dc(data_list)
    random.shuffle(random_list)
    mark = 0
    acc_list = []
    for i in range(1):                                                                   #fold range
        test_list = []
        training_list = []
        index = 0
        while (mark < int(len(random_list))):
            for train_e in range(0,mark):
                training_list.append(random_list[train_e])
            else:
                index = mark
                mark = int(len(random_list)/10)+index
                for test_element in range(index,mark):
                    test_list.append(random_list[test_element])
                for training_element in range(mark,int(len(random_list))):
                    training_list.append(random_list[training_element])

        # fold completion
            root = Node(training_list)
            root.node_type = 'root'
            buildTree(root)
            predicted_list = []
            temp = dc(root)
            for element in test_list:
                predicted_list.append(class_finder(element,temp))
            acc_list.append(accuracy(test_list,predicted_list))
            break
    print (mean(acc_list))
        #print(test_list)
        #print(predicted_list)

        #print(len(test_list))

def startQuestion1(file_name):
    data_list = fileRead(file_name)
    startSplitData(data_list)


formula_input = int(input('Enter\n0 for Information Gain \n1 for Gini \n2 for both\n Your input here:'))
for file_loc in range(9):
    files_names = [['iris.data.txt','comma']
                   ,['haberman.data.txt','comma']
                   ,['seeds_dataset.txt','tab']
                   ,['column_3c.dat','space']
                   ,['wine.data.txt','comma']
                   ,['data_banknote_authentication.txt','comma']
                   ,['SPECT.train_features.csv.txt','comma']
                   ,['ionosphere.data.txt','comma']
                   ,['tae.data.txt','comma']]
    column_names_list = [["sepalLength", "sepalWidth", "petalLength", "petalWidth","irisClass"]
                         ,['Age of patient at time of operation', "Patient's year of operation",'Number of positive axillary nodes detected (numerical)','Survival status']
                         ,['area','perimeter','compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove','varieties of wheat']
                         ,['pelvic incidence','pelvic tilt','lumbar lordosis angle','sacral slope','pelvic radius','grade of spondylolisthesis','class labels']
                         ,['cultivator','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
                         ,['variance of Wavelet Transformed image','skewness of Wavelet Transformed image','curtosis of Wavelet Transformed image','entropy of image','class']
                         ,['OverAllDiagnosis','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22']
                         ,[i for i in range(1,36)]
                         ,[i for i in range(1,7)]]
    class_column_list = ['irisClass','Survival status','varieties of wheat','class labels','cultivator','class','OverAllDiagnosis',35,6]
    column_names = column_names_list[file_loc]
    #print(column_names)
    class_column = column_names.index(class_column_list[file_loc])
    formula_list = ['Information Gain','Gini']
    if formula_input == formula_list.index('Information Gain'):
        formula = formula_list.index('Information Gain')
        print('The Accuracy measures on %s for 10-fold validation using Information Gain'%(files_names[file_loc][0]))
        startQuestion1(files_names[file_loc])
    elif formula_input == formula_list.index('Gini'):
        formula = formula_list.index('Gini')
        print('The Accuracy measures on %s for 10-fold validation using Gini'%(files_names[file_loc][0]))
        startQuestion1(files_names[file_loc])
    elif formula_input == 2:
        for measure in formula_list:
            #formula = int(input('Enter 0 for Information Gain and 1 for Gene'))
            formula = formula_list.index(measure)
            print('The Accuracy measures on %s for 10-fold validation using %s'%(files_names[file_loc][0],measure))
            startQuestion1(files_names[file_loc])