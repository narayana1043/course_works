# libraries used
import sys
from math import log
from operator import itemgetter
from collections import Counter

# Global Variable declarations
column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
train_data = list()
actual_class = list()
test_data = list()
prediction = list()

# reading data functions
with open('./data/monks-1.train.txt') as f:

    for line in f:
        train_data.append([int(i) for i in line.split(' ')[1:8]])

with open('./data/monks-1.test.txt') as f:

    for line in f:
        test_data.append([int(i) for i in line.split(' ')[1:8]])
        actual_class.append([int(i) for i in line.split(' ')[1]][0])

class Node:
    # Class structure to build the tree
    def __init__(self, data,col=0, value=0, parent=None):
        # class constructor
        # class variables declaration
        self.data = data
        self.split_col = col
        self.split_value = value
        self.leaf = False
        self.parent = parent
        self.class_name = None

    # setter functions
    def set_children(self, left_data, right_data, parent):
        # sets the children to the current node by assigning data and current node as parent to children
        self.next_left = Node(data=left_data, parent=parent)
        self.next_right = Node(data=right_data, parent=parent)

    def node_class_name(self, class_name):
        # sets the class label for the node when the node can no longer grow
        if self.leaf == True:
            self.class_name = class_name

    # getter functions
    def get_children(self):
        # returns the children for the current node
        return self.next_left, self.next_right

    def get_parent(self):
        # returns the parent of the current node
        return self.parent

def counter(data, class_labels, class_col):
    # Counter funtions to counts the number of each class variables
    label_counter = [0 for i in class_labels]

    for dpt in data:

        label_counter[dpt[class_col]] += 1

    return label_counter

def calculate_entropy(data, feature_values, col, class_labels):
    # Entropy calculator
    class_col = 0
    entropy = list()    # To store entropy at different values for each feature
    feature_values = list(feature_values)

    for value in feature_values:
        left_data = list()
        right_data = list()

        for dpt in data:

            if dpt[col] < value:

                left_data.append(dpt)

            else:

                right_data.append(dpt)

        # print(left_data)
        # print(right_data)
        tot_entropy_of_splits = 0   # Total entropy of the splits
        entropy_in_splits = [0, 0]  # To store entropy of each split i.e: "binary split so [0, 0]"
        len_data = len(data)
        data_splits = [left_data, right_data]

        for data_split_index in range(len(data_splits)):
            # calculating the individual entropy at each split
            data_split = data_splits[data_split_index]
            len_data_split = len(data_split)
            if len_data_split <= 0:

                pass

            else:

                label_counter = counter(data_split, class_labels, class_col)
                # print(label_counter)
                # print(len(label_counter))
                for label_index in range(len(label_counter)):

                    if label_counter[label_index] != 0:
                        # formula for entropy of each split
                        entropy_in_splits[data_split_index] += (label_counter[label_index]/len_data_split)\
                                                               *log(1/(label_counter[label_index]/len_data_split))

                    else:

                        pass
            # adding the entropy of the splits inorder to compare with other splits
            tot_entropy_of_splits += (len_data_split/len_data)*entropy_in_splits[data_split_index]
        entropy.append([tot_entropy_of_splits, value])
    # print(entropy)

    # Choosing the best entropy of the feature and returning that
    if (len(entropy) > 1):

        best_selection_on_feature = min(entropy, key=itemgetter(0))

    else:

        return entropy[0]
    # print(best_selection_on_feature)
    return best_selection_on_feature


def split_condtion(data):
    # Function to determine which is the best split at each depth for the nodes at that depth
    # If there is no split required returns None otherwise returns the best split
    class_labels = set([dpt[0] for dpt in data])

    if (len(class_labels) < 2):
        return None

    best_selection_of_feature_and_feature_value = list()

    for col in range(1,7):

        feature_values = list()
        for dpt in data:

            feature_values.append(dpt[col])

        feature_values = set(feature_values)
        best_selection_of_feature_and_feature_value.append([calculate_entropy(data, feature_values, col, class_labels), col])
    selection = min(best_selection_of_feature_and_feature_value, key=itemgetter(0))
    # print(selection)
    return selection

def create_leaf_node(node):
    # Makes a node as leaf of the branch when the branch can no longer grow and assigns the leaf the name of the
    # class_label that best suites depending on the majority class present in the node
    node.leaf = True
    class_names = set(dpt[0] for dpt in node.data)
    class_count = dict()
    for class_name in class_names:

        class_count[class_name] = 0

    for dpt in node.data:

        class_count[dpt[0]] += 1

    class_count = Counter(class_count)
    node.class_name = class_count.most_common(1)[0][0]
    return class_count

def tree_builder(node, depth, temp):
    # Builds the tree by making a binary split at each node. If a binary split is not possible it checks for the leaf
    # node condition and converts the node into leaf node if the condition is satisfied by calling respective functions.
    while depth > 0 and temp != False:

        # print(node.data)
        split = split_condtion(node.data)
        if split == None:
            create_leaf_node(node)
            temp = False
            return temp
        else:
            node.split_col = split[1]
            node.split_value = split[0][1]
            left_data = list()
            right_data = list()

            for dpt in node.data:

                if dpt[node.split_col] < node.split_value:
                    left_data.append(dpt)
                else:
                    right_data.append(dpt)
            # print(len(left_data), len(right_data))
            if len(left_data) == 0 or len(right_data) == 0:
                create_leaf_node(node)
                temp = False
                return temp
            node.set_children(left_data, right_data, node)

            for child_node in node.get_children():

                temp = tree_builder(child_node, depth-1, True)

    else:

        if (depth == 0):

            create_leaf_node(node)
            temp = False

    return temp


def print_tree(node, depth):
    # used to print the tree in top down fashion
    print("depth: ",depth)
    print("col: ", node.split_col, "value: ",node.split_value, "len: ", len(node.data))
    if node.class_name != None:

        print('class name: ',node.class_name)

    else:

        print('class name: None')

    if node.leaf == False:

        side = ['L', 'R']
        k=0

        for node in node.get_children():

            # print(side[k])
            print_tree(node, depth+1)
            k += 1


def test_case(node, dpt, rec_breaker):
    # recursively moves the test data point inside and then finally decides the class label depending on the leaf node
    # the data point ends in
    if node.leaf == False:

        col = node.split_col
        value = node.split_value
        # print(col, value)
        if dpt[col] < value:

            rec_breaker = test_case(node.next_left, dpt, rec_breaker)

        else:

            rec_breaker = test_case(node.next_right, dpt, rec_breaker)

    else:

        prediction.append(node.class_name)
        rec_breaker = False

    return rec_breaker

def test(node_root, test_data):
    # used to increment the data point one after the other
    for dpt in test_data:
        test_case(node_root, dpt, True)

def decision_tree(depth):
    # takes depth of the tree as parameter and prints the accuracy of the tree using the test data
    node_root = Node(data=train_data)
    tree_builder(node_root, depth-1, True)

    # print(actual_class)
    # print(prediction)
    # print_tree(node_root, 0)

    test(node_root, test_data)

    correct_count = 0
    wrong_count = 0

    test(node_root, test_data)

    for index in range(len(actual_class)):
        if prediction[index] == actual_class[index]:
            correct_count += 1
        else:
            wrong_count += 1

    print(correct_count/(correct_count + wrong_count))

def decision_tree_start(args):
    for arg in range(2,args):
        print(arg)
        decision_tree(arg)

if __name__ == '__main__':
    # print(sys.argv[1])
    decision_tree_start(int(sys.argv[1]))

# decision_tree(9)
