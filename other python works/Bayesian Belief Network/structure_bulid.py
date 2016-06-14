import pandas as pd
import itertools

class Node:

    def __init__(self, name):

        self.name = name
        self.parents = []
        self.parents_names = []

    def add_parent(self, parent):

        self.parents.append(parent)
        self.parents_names.append(parent.name)

    def truth_values(self):

        if len(self.parents) == 0:
            # self.probability = input('Enter the probability of for node %s True : ' %(self.name))
            self.probability = 0.5

        else:
            tot_no_conditions = len(self.parents)
            print(tot_no_conditions)
            condition_names = self.parents_names
            print(condition_names)
            self.prob_dataframe = pd.DataFrame(columns= condition_names + ['probability'])
            table = list(itertools.product([True, False], repeat= (len(self.prob_dataframe.columns)-1)))

            set_probabilities(self, table)


    def show_parents(self):

        return self.parents



def set_probabilities(node, table):

    for i in range(0, 2 ** (len(node.prob_dataframe.columns) - 1)):

        table[i] = list(table[i]) + [0]
        print('*************************************')
        node.prob_dataframe.loc[i] = table[i]
        print(node.prob_dataframe.loc[i][:-1])
        # node.prob_dataframe.loc[i, 'probability'] = input('Enter probability for %s for above case:  '%(node.name))
        node.prob_dataframe.loc[i, 'probability'] = 0.5

    print(node.prob_dataframe)

node = {}
node_names = []

# parent1 = Node('burgulary')
# parent1.truth_values()
# node_names.append(parent1.name)
#
# parent2 = Node('earthquake')
# parent2.truth_values()
# node_names.append(parent2.name)
#
# child1 = Node('alarm')
# child1.add_parent(parent1)
# child1.add_parent(parent2)
# child1.truth_values()
# node_names.append(child1.name)
#
# child2 = Node('john')
# child2.add_parent(child1)
# child2.truth_values()
# node_names.append(child2.name)
#
# child3 = Node('mary')
# child3.add_parent(child1)
# child3.truth_values()
# node_names.append(child3.name)


def bayesianNW_builder():

    while (True):

        node_type = input('Enter "p" for parent nodes or "c" for child nodes or "q" to quit adding nodes to network')

        if (node_type != 'q' or node_type == 'Q'):

            node_name = input('Enter the node name')

        if (node_type == 'P' or node_type == 'p'):

            node[node_name] = Node(node_name)
            node[node_name].truth_values()

        elif (node_type == 'C' or node_type == 'c'):

            node[node_name] = Node(node_name)
            no_parents = int(input('Enter the number of parents for node %s [max = %d] : '%(node_name, len(node_names))))
            print('Choose the parents from the list : '+', '.join(node_names))
            parent_nodes = input('Enter the parent nodes names separated by space').split()

            for num in range(no_parents):

                node[node_name].add_parent(node[parent_nodes[num]])

            node[node_name].truth_values()

        elif (node_type == 'Q' or node_type == 'q'):

            break

        else:

            print('Please enter P/p C/c or Q/q')

        node_names.append(node_name)

bayesianNW_builder()
