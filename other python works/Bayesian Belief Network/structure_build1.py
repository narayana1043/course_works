import pandas as pd

class Node:

    def __init__(self, name, states, parents):

        self.name = name
        self.states = states
        self.parents = parents

    def setProbability(self):

        if len(self.parents) == 0:

            self.probability = {}

            count = len(self.states)
            temp = 1

            while temp < count:

                self.probability[self.states[temp]] = float(input('enter the probability for ', self.name,' when state is ',self.states[temp]))
                temp += 1

            else:

                tempSum = 0

                for state in self.states:

                    tempSum += self.setProbability[state]

                self.probability[self.states[temp]] = 1 - tempSum

        elif len(self.parents) > 0:

            df = pd.Dataframe()


def tableCreate(df, parents, states)

    for


def printNodeList(nodesList):

    for node,num in zip(nodesList,range(len(nodesList))):

        print(num+1,'. ',node.name)


def setNodeParents(nodesList):

    parentNodes = list()

    if len(nodesList) != 0:

        noParents = input('If no parents press y')

        if noParents == 'y':

            return parentNodes

        print('enter the selection numbers separated by space from the nodes shown')
        printNodeList(nodesList)
        parents = input().split(sep=' ')

        for parent in parents:

            parent = int(parent)-1
            parentNodes.append(nodesList[parent])

        if len(parents) == len(set(parents)):

            return parentNodes

        else:

            print('parent nodes list must be unique')
            return setNodeParents(nodesList)

    else:

        return parentNodes

def setNodeStates():

    return input('Enter all states of Node separated by a space:').split(sep=' ')


def networkPrinter(nodesList):
    for node in nodesList:
        print(node.name, '\n', node.states, '\n', node.parents)

def nodeNamer(nodeNamesList):

    name = input('Node name:').capitalize()

    while len(nodeNamesList) != 0:

        if name not in nodeNamesList:

            return name

        else:

            print('the name already in list, select another name')
            name = input('Node name:').capitalize()
    else:

        return

def makeBayesianNetwork():

    nodesList = list()
    nodeNamesList = list()

    print('First node must be a parent node')

    while True:

        name = input('Node name:')
        nodeNamesList.append(name)
        states = setNodeStates()
        parents = setNodeParents(nodesList)
        node = Node(name, states, parents)
        nodesList.append(node)

    networkPrinter(nodesList)

makeBayesianNetwork()