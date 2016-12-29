import copy

# matrix the is used
matrix = [[1,2,3],[4,6,5],[3,2,1]]
# K - value
k = 12

src = (0,0)
dest = (len(matrix),len(matrix[0]))
i = dest[0] - 1
j = dest[1] - 1

paths_to_explore = []
paths_to_explore.append([[[0,0]],matrix[0][0]])
paths_found = []

while paths_to_explore:
    curr_path = paths_to_explore.pop(0)
    path = copy.deepcopy(curr_path)
    pathlist = path[0]
    cell = pathlist[-1]
    if cell[0] < i:
        temp = path[1] + matrix[cell[0] + 1][cell[1]]
        path[0].append([cell[0]+1,cell[1]])
        path[1] = temp
        if temp <= k:
            paths_to_explore.append(path)
    path = curr_path
    if cell[1] < j:
        temp = path[1] + matrix[cell[0]][cell[1] + 1]
        path[0].append([cell[0], cell[1] + 1])
        path[1] = temp
        if temp <= k:
            paths_to_explore.append(path)
    if cell[0] == i and cell[1] == j:
        paths_found.append(path)

for path in paths_found:
    if path[1] == k:
        print(path)
