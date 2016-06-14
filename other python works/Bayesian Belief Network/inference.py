from structure_bulid import node_names

def enumerationAsk(queryVar, evidenceVarList, bn):

    output = None

    for

def inputVar():

    print('List of Nodes:')

    for i in range(len(node_names)):

        print(i + 1, ' ', node_names[i])

    queryVarNum = int(input('Enter the Query Node number from the above list:'))
    queryVar = node_names[queryVarNum - 1]

    evidenceVarNum = int(input('Enter the number evidence variables'))

    evidenceVarList = list()
    for i in range(evidenceVarNum):

        evidenceVar = int(input('Enter the evidence variable number for evidence %d' %(i+1)))
        evidenceVarList.append(node_names[evidenceVar - 1])

    print('the query variable is', queryVar)
    print('the evidence variables are', evidenceVarList)

    return queryVar, evidenceVarList

inputVar()

