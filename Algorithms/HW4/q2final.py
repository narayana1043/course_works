
def rodCut(costs, n, c):
    k = []
    cuts = []
    for i in range(n-len(costs)+1):
        k.append(0)
    costs.extend(k)
    for i in range(n+1):
        cuts.append([])
    r = [0 for i in range(0,len(costs)+1)]
    for j in range(1,len(costs)):
        q = costs[j]
        cuts[j]=[j]
        for i in range(1,j):
            temp = costs[i]+r[j-i]-c
            if q<temp:
                s = []
                s.append(i)
                s.extend(cuts[j-i])
                cuts[j] = s
            q = max(q, temp)
        r[j] = q
    return r[n],cuts[n]


# Please update these values to check for various settings
costs = [0,1,5,8,9,10,17,17,20,24,30]
settings = [(15,3),(20,2)]
for set in settings:
    print('Printing for the setting: ',set)
    lengthOftheRod = set[0]
    cutCost = set[1]
    solution = rodCut(costs, lengthOftheRod, cutCost)
    print('Total Cost of the tree:', solution[0])
    print('lenggth of the rods resulting in optimal price',solution[1])

