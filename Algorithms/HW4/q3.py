import operator

List = [[0,1],[1,3],[0,4],[1,2],[4,6],[2,9],[5,8],[3,5],[4,5],[11,17]]
start = 1
finish = 9

def activitySelection(L, st, ft):
    L = sorted(L, key=operator.itemgetter(1))
    for activity in L:
        if activity[0] < st:
            del L[L.index(activity)]
        elif activity[1] > ft:
            del L[L.index(activity)]
    SA = []
    SA.append(L[0])
    prevActivityFt = SA[0][1]
    next_checkpt = 1
    condition = True
    while condition:
        activity = L[next_checkpt]
        activitySt = activity[0]
        if activitySt >= prevActivityFt:
            SA.append(activity)
            prevActivityFt = activity[1]
        elif prevActivityFt == ft:
            condition = False
        next_checkpt += 1
        if next_checkpt == len(L):
            break
    return SA

print(activitySelection(L=List,st=start,ft=finish))
