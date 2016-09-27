from math import log10
from operator import itemgetter
import random
from copy import deepcopy as dc
from statistics import mean
from collections import Counter
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def file_read(file_name):
    if file_name[1] == 'comma':
        file_data_lines = open(file_name[0], 'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split(','))
    elif file_name[1] == 'tab':
        file_data_lines = open(file_name[0], 'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split('\t'))
    elif file_name[1] == 'space':
        file_data_lines = open(file_name[0], 'r').readlines()
        data_list = []
        for line in file_data_lines:
            data_list.append(line.rstrip('\n').split(' '))
    return data_list


def break_points_gen(breakp_list):
    if len(set(breakp_list)) != 1:
        max_point = max(breakp_list)
        min_point = min(breakp_list)
        splits = 5
        dist_bw_points = (max_point - min_point) / splits
        break_points = []
        point = min(breakp_list)
        while True:
            point += dist_bw_points
            if point < max_point:
                break_points.append(point)
            else:
                break
        return break_points
    else:
        return set(breakp_list)


def count_values_same_class(cvsc_list, class_column, class_values):
    value_dict = {}
    count_value_dict = {}
    for value in class_values:
        value_dict[value] = []
        count_value_dict[value] = []
    for data in cvsc_list:
        for value in class_values:
            if data[class_column] == value:
                value_dict[value].append(data)
    for value in class_values:
        count_value_dict[value] = len(value_dict[value])
    return count_value_dict


def split_weight_find(count_value_dict, class_values):
    values_list = []
    split_weight = 0
    sum_values = 0
    for value in class_values:
        values_list.append(count_value_dict[value])
    sum_values = sum(values_list)
    for value in class_values:
        if count_value_dict[value] != 0:
            split_weight -= (count_value_dict[value] / sum_values) * log10(
                count_value_dict[value] / sum_values)
    return (split_weight, sum_values)


def cal_fun(data_list, break_points, column, class_column, class_values):
    break_points_split_value_list = []
    for value in break_points:
        left_list = []
        right_list = []
        count_value_list_dict = []
        child_split_weight_list = []
        child_split_weight = 0
        tot_sum = 0
        for data in data_list:
            if float(data[column]) <= value:
                left_list.append(data)
            else:
                right_list.append(data)
        count_value_list_dict.append(
            count_values_same_class(left_list, class_column, class_values))
        count_value_list_dict.append(
            count_values_same_class(right_list, class_column, class_values))
        for count_value_dict in count_value_list_dict:
            child_split_weight_list.append(
                split_weight_find(count_value_dict, class_values))
        for item in child_split_weight_list:
            tot_sum += item[1]
        for item in child_split_weight_list:
            child_split_weight += item[0] * (item[1] / tot_sum)
        parent_split_weight = split_weight_find(
            count_values_same_class(data_list, class_column, class_values),
            class_values)
        break_points_split_value_list.append(
            [value, parent_split_weight[0] - child_split_weight])
    return break_points_split_value_list


def set_missing_data(data_list, column):
    temp_miss = []
    for miss_data in range(len(data_list)):
        if data_list[miss_data][column] != '':
            temp_miss.append(float(miss_data))
    data_list[miss_data][column] = mean(temp_miss)
    return data_list[miss_data][column]


def split_criteria_cal(data_list, column, class_column, class_values):
    list = []
    for data in data_list:
        if data[column] == '':
            data[column] = set_missing_data(data_list, column)
        list.append(float(data[column]))
    break_points = break_points_gen(list)
    break_points_info_gain = cal_fun(data_list, break_points, column,
                                     class_column, class_values)
    return sorted(break_points_info_gain, key=itemgetter(1), reverse=True)[0]


def split_criteria(data_list, column_names, class_column):
    col_dict = []
    class_values = set()
    for data in data_list:
        class_values.add(data[class_column])
    for col in range(0, len(column_names)):
        if col != class_column:
            col_dict.append([split_criteria_cal(data_list, col, class_column,
                                                set(class_values)), col])
    return sorted(col_dict, key=lambda x: x[0][1], reverse=True)[0]


def spliting_tree(data_list, split_col, split_value):
    left_list = []
    right_list = []
    for record in data_list:
        if float(record[split_col]) <= split_value:
            left_list.append(record)
        else:
            right_list.append(record)
    return left_list, right_list


def stop_condition(node, class_column):
    temp_list = []
    stop_list = node.data
    if node.node_type == 'Leaf':
        return True
    for stop_element in stop_list:
        temp_list.append(stop_element[class_column])
    if len(set(temp_list)) == 1:
        return True
    return False


def max_class(max_list, class_column):
    max_class_list = []
    for max_element in max_list:
        max_class_list.append(max_element[class_column])
    counter = Counter(max_class_list)
    max_count = max(counter.values())
    return [k for k, v in counter.items() if v == max_count][0]


def node_err_cal(err_list, max_class_name, class_column):
    corr_count = 0
    for err_element in err_list:
        if max_class_name != err_element[class_column]:
            corr_count += 1
    return (corr_count)


def build_tree(node):
    # column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth",
    #                 "irisClass"]
    # class_column = "irisClass"
    # print(len(node.data),node.data)
    node.stop_condition = stop_condition(node, class_column)
    if node.stop_condition == True:
        node.max_class_name = max_class(node.data, class_column)
        node.node_type = 'Leaf'
        node.node_err = 0
        Node.leaf_children.append(node)
        node.node_error = node_err_cal(node.data, node.max_class_name,
                                       class_column)
        # print(node.node_type)
        return False
    else:
        split_data = split_criteria(node.data, column_names, class_column)
        split_value = split_data[0][0]
        split_col = split_data[1]
        # print(node.name)
        node.split_column = split_col
        node.split_value = split_value
        node.name = '%d , %f' % (split_col, split_value)
        node.max_class_name = max_class(node.data, class_column)
        node.node_error = node_err_cal(node.data, node.max_class_name,
                                       class_column)
        # print(node.name,node.split_column)
        node.left_list, node.right_list = spliting_tree(node.data,
                                                        split_col, split_value)
        # print(len(node.left_list),node.left_list)
        # print(len(node.right_list),node.right_list)
        node.left_child = (Node(node.left_list, node))
        node.left_child.max_class_name = max_class(node.left_child.data,
                                                   class_column)
        node.left_child.node_error = \
            node_err_cal(node.left_child.data, node.left_child.max_class_name,
                         class_column)
        Node.new_children.append(node.left_child)
        if len(node.right_list) != 0:
            node.right_child = (Node(node.right_list, node))
            node.right_child.max_class_name = max_class(
                node.right_child.data, class_column)
            node.right_child.node_error = node_err_cal(
                node.right_child.data, node.right_child.max_class_name,
                class_column)
            Node.new_children.append(node.right_child)
            if node.node_type == 'root':
                Node.children = Node.new_children
                Node.new_children = []
            else:
                return False
        else:
            node.node_type = 'Leaf'
            Node.leaf_children.append(node)
            Node.leaf_children.append(node)
            return False
        while Node.children != []:
            while Node.children != []:
                # print(Node.children)
                child = Node.children.pop(0)
                Node.temp_children.append(child)
                # print(Node.temp_children)
                build_tree(child)
                # if Node.children == []:
            sum_node_err = 0
            count_nodes = 0
            for child_node in Node.temp_children:
                count_nodes += 1
                # print(len(child_node.data), child_node.max_class_name,
                #       child_node.node_error)
                sum_node_err += child_node.node_error
            for child_node in Node.leaf_children:
                count_nodes += 1
            pessi_err = (
                (sum_node_err + count_nodes * 0.5) / Node.len_training_list)
            if pessi_err < Node.old_pessi_err:
                Node.old_pessi_err = pessi_err
                Node.children = Node.new_children
                Node.new_children = []
                Node.temp_children = []
            else:
                while Node.temp_children != []:
                    temp_child = Node.temp_children.pop(0)
                    temp_child.parent_node.node_type = 'Leaf'
                    temp_child.max_class_name = max_class(temp_child.data,
                                                          class_column)
                    # print(temp_child.max_class_name)


def class_finder(find_element, temp):
    while (True):
        if temp.split_column == None:
            return temp.max_class_name
        if find_element[temp.split_column] == '':
            find_element[
                temp.split_column] = temp.split_value + 1
            # pushing the missing value to the right is improving accuracy
        if float(find_element[temp.split_column]) <= temp.split_value:
            temp_obj = temp.left_child
            # print(temp.split_value)
        elif float(find_element[temp.split_column]) > temp.split_value:
            temp_obj = temp.right_child
            # print(temp.split_value)
        # print(temp_obj.node_type)
        if temp_obj.node_type == 'Leaf':
            return temp_obj.max_class_name
        temp = temp_obj


def accuracy(test_list, predicted_list, class_column):
    count = 0
    len_test_list = len(test_list)
    for index in range(len_test_list):
        if test_list[index] == predicted_list[index]:
            count += 1
    return (count / len(test_list) * 100)


class Node(object):
    children = []
    leaf_children = []
    temp_children = []
    new_children = []
    len_training_list = 0
    old_pessi_err = 0

    def __init__(self, data, parent_node=None):
        self.name = None
        self.max_class_name = None
        self.node_type = None
        self.split_column = None
        self.split_value = None
        self.data = data
        self.stop_condition = None
        self.left_child = None
        self.right_child = None
        self.node_error = None
        self.parent_node = None
        if parent_node is not None:
            self.parent_node = parent_node
        else:
            self.parent_node = None

    def add_child(self, obj):
        self.children.append(obj)


def class_list_gen(list):
    class_list = []
    for element in list:
        class_list.append(int(element[class_column]))
    return class_list


def start_split_data(data_list):
    random_list = dc(data_list)
    random.shuffle(random_list)
    predicted_list = []
    mark = 0
    acc_list = []
    act_class_list = []
    for i in range(10):  # fold range
        test_list = []
        training_list = []
        while (mark < int(len(random_list))):
            for train_ele in range(0, mark):
                training_list.append(random_list[train_ele])
            else:
                index = mark
                mark = int(len(random_list) / 10) + index
                for test_element in range(index, mark):
                    test_list.append(random_list[test_element])
                for training_element in range(mark, int(len(random_list))):
                    training_list.append(random_list[training_element])
                    # print(training_list)
                    # fold completion
                Node.children = []
                Node.leaf_children = []
                Node.temp_children = []
                Node.new_children = []
                Node.len_training_list = len(training_list)
                Node.old_pessi_err = (node_err_cal(training_list, max_class(
                    training_list, class_column), class_column) + 1) / \
                                     Node.len_training_list
                root = Node(training_list)
                # print(root.data)
                root.node_type = 'root'
                build_tree(root)
                predicted_temp_list = []
                actual_list = []
                temp_root = dc(root)
                for test_element in test_list:
                    actual_list.append(int(test_element[class_column]))
                    found = int(class_finder(test_element, temp_root))
                    predicted_temp_list.append(found)
                    predicted_list.append(found)
                acc_list.append(
                    accuracy(actual_list, predicted_temp_list, class_column))
                break
    print(mean(acc_list))
    act_class_list = class_list_gen(random_list)
    # print(len(act_class_list),len(predicted_list))
    while (len(act_class_list) > len(predicted_list)):
        del act_class_list[-1]
    c_matrix = cm(act_class_list, predicted_list)
    print('Confusion matrix\n', c_matrix)
    c_report = cr(act_class_list, predicted_list)
    print("All Measures required for this data set \n", c_report)
    fpr, tpr, thd = rc(act_class_list, predicted_list)
    roc_auc = auc(fpr, tpr)
    if formula_input == 2:
        plt.title('ROC for %s with information gain(red) and gini(blue)'
                  % file_name[0])
        plt.plot(fpr, tpr,
                 label='%s  AUC = %0.2f' % (formula_measure, roc_auc))
        plt.legend(loc='lower right')
    else:
        plt.title('ROC for %s ' % file_name[0])
        plt.plot(fpr, tpr, label='%s  AUC = %0.2f' % (formula_measure,
                                                      roc_auc))
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()


def start_question1(file_name):
    data_list = file_read(file_name)
    class_set = []
    for i in range(len(data_list)):
        if data_list[i][class_column] not in class_set:
            class_set.append(data_list[i][class_column])
        data_list[i][class_column] = int(
            class_set.index(data_list[i][class_column])) % 2

    # print(data_list)
    start_split_data(data_list)


formula_input = int(input("Enter\n0 for Information Gain \n1 for Gini \n2 "
                          "for both\n Your input here:"))

for file_loc in range(9):
    files_names = [['iris.data.txt', 'comma'], ['haberman.data.txt',
                                                'comma'],
                   ['seeds_dataset.txt', 'tab'], ['column_3c.dat', 'space'],
                   ['wine.data.txt', 'comma'],
                   ['data_banknote_authentication.txt', 'comma'],
                   ['SPECT.train_features.csv.txt', 'comma'],
                   ['ionosphere.data.txt', 'comma'],
                   ['tae.data.txt', 'comma'], ]

    column_names_list = [['sepalLength', 'sepalWidth', 'petalLength',
                          'petalWidth", "irisClass'],
                         ['Age of patient at time of operation',
                          "Patient's year of operation",
                          'Number of positive axillary nodes detected ('
                          'numerical)', 'Survival status'],
                         ['area', 'perimeter', 'compactness', 'length of '
                                                              'kernel',
                          'width of kernel', 'asymmetry coefficient',
                          'length of kernel groove', 'varieties of wheat'],
                         ['pelvic incidence', 'pelvic tilt', 'lumbar '
                                                             'lordosis angle',
                          'sacral slope', 'pelvic radius',
                          'grade of spondylolisthesis', 'class labels'],
                         ['cultivator', 'Alcohol', 'Malic acid', 'Ash',
                          'Alcalinity of ash', 'Magnesium', 'Total phenols',
                          'Flavanoids', 'Nonflavanoid phenols',
                          'Proanthocyanins', 'Color intensity', 'Hue',
                          'OD280/OD315 of diluted wines', 'Proline'],
                         ['variance of Wavelet Transformed image',
                          'skewness of Wavelet Transformed image',
                          'curtosis of Wavelet Transformed image',
                          'entropy of image',
                          'class'],
                         ['OverAllDiagnosis', 'F1', 'F2', 'F3', 'F4', 'F5',
                          'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13',
                          'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                          'F21', 'F22'],
                         [i for i in range(1, 36)],
                         [i for i in range(1, 7)]
                         ]

    class_column_list = ['irisClass', 'Survival status', 'varieties of wheat',
                         'class labels', 'cultivator', 'class',
                         'OverAllDiagnosis', 35, 6]
    column_names = column_names_list[file_loc]
    # print(column_names)
    class_column = column_names.index(class_column_list[file_loc])
    file_name = files_names[file_loc]
    formula_list = ['Information Gain', 'Gini']
    if formula_input == formula_list.index('Information Gain'):
        formula = formula_list.index('Information Gain')
        print('The Accuray measures on %s for 10-fold validation using'
              ' Information Gain'
              % (file_name[0]))
        start_question1(file_name)
    elif formula_input == formula_list.index('Gini'):
        formula = formula_list.index('Gini')
        print('The Accuray measures on %s for 10-fold validation using Gini'
              % (file_name[0]))
        start_question1(file_name)
    elif formula_input == 2:
        for measure in formula_list:
            # formula = int(input('Enter 0 for Information Gain and 1 '
            #                     'for Gene'))
            formula_measure = measure
            formula = formula_list.index(measure)
            print('The Accuray measures on %s for 10-fold validation using %s'
                  % (file_name[0], measure))
            start_question1(file_name)
