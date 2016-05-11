class Graph(object):
    def graph_gen(self, dist_bw_places):
        self.dist_bw_places = dist_bw_places
        temp = []
        for i in dist_bw_places:
            temp.append(i[::-1])
        self.dist_bw_places += temp
        self.nodes = set(i[0] for i in self.dist_bw_places)
        graph_data = {}
        for node in self.nodes:
            for i in self.dist_bw_places:
                if node == i[0] and node not in graph_data.keys():
                    graph_data.update({node: [(i[2], i[1])]})
                elif node == i[0] and node in graph_data.keys():
                    graph_data[node].append((i[2], i[1]))
        return graph_data

class Node(object):
    def __init__(self, curr_node_name, parent_node=None, path_cost=0):
        self.__node_name = curr_node_name
        self.__parent_node = parent_node
        if parent_node is not None:
            self.__tot_path_cost = parent_node.tot_path_cost() + path_cost
        else:
            self.__tot_path_cost = path_cost

    def node_name(self):
        return self.__node_name

    def parent_node(self):
        return self.__parent_node

    def tot_path_cost(self):
        return self.__tot_path_cost

    def add_child(self, child_name, path_cost):
        return Node(child_name, self, path_cost)

def path_finder(present_node):
    if present_node.parent_node() != None:
        return path_finder(present_node.parent_node()) + ' ' + present_node.node_name()
    else:
        return present_node.node_name()

def bfs_start(curr_node, dest_node, graph_data, visited_nodes, visited_node_names):
    src_node = Node(curr_node)
    pipe = []
    pipe_node_names = []
    pipe.append(src_node)
    if src_node.node_name() != dest_node:
        while True:
            present_node = pipe.pop(0)
            visited_nodes.append(present_node)
            visited_node_names.append(present_node.node_name())
            if dest_node not in visited_node_names:
                next_nodes = graph_data[present_node.node_name()]
                for child in next_nodes:
                    if child[0] not in visited_node_names:
                        next_node = present_node.add_child(child[0], int(child[1]))
                        if next_node.node_name() not in pipe_node_names:
                            pipe.append(next_node)
                            pipe_node_names.append(next_node.node_name())
            else:
                print("Traversed path in BFS:",visited_node_names)
                print("BFS path:", path_finder(present_node))
                print("Total Path Cost:",visited_nodes.pop().tot_path_cost())
                break

def dfs_start(curr_node, dest_node, graph_data, visited_nodes, visited_node_names):
    src = Node(curr_node)
    pipe = []
    pipe.append(src)
    if src.node_name() != dest_node:
        while True:
            present_node = pipe.pop()
            visited_nodes.append(present_node)
            visited_node_names.append(present_node.node_name())
            if dest_node not in visited_node_names:
                next_nodes = graph_data[present_node.node_name()]
                for child in next_nodes:
                    if child[0] not in visited_node_names:
                        next_node = present_node.add_child(child[0], int(child[1]))
                        pipe.insert(0, next_node)
            else:
                print("Travesd path in DFS:",visited_node_names)
                print("DFS path:", path_finder(present_node))
                print("Total Path Cost:",visited_nodes.pop().tot_path_cost())
                break

def dls_start(curr_node, dest_node, graph_data, depth, visited_nodes, visited_node_names):
    result = ""
    visited_nodes.append(curr_node)
    visited_node_names.append(curr_node.node_name())
    #print(visited_node_names)
    if dest_node in visited_node_names:
        #print(visited_node_names)
        print("Travesd path in IDS:",path_finder(curr_node))
        print("Total Path Cost:",curr_node.tot_path_cost())
        return "success"
    elif depth == 0:
        return "cutoff"
    elif depth != 0:
        while True:
            next_nodes = graph_data[curr_node.node_name()]
            for child in next_nodes:
                if child[0] not in visited_node_names:
                    next_node = curr_node.add_child(child[0], int(child[1]))
                    result = dls_start(next_node, dest_node, graph_data, depth-1, visited_nodes, visited_node_names)
                    if result == "success":
                        return result
            result = "cutoff"
            return result

def idls_start(curr_node, dest_loc, graph_data, depth):
    state = "cutoff"
    step = depth
    while state == "cutoff":
        visited_nodes = []
        visited_node_names = []
        state = dls_start(curr_node, dest_loc, graph_data, step,visited_nodes,visited_node_names)
        if state == "cutoff":
            step = step+depth
        else:
            print(state)
            break

def open_file(file_name):
    # Reading data from the input file and making a list of lists from that. The name of the list is dist_bw_places
    file_name = file_name
    dist_file = open(file_name, "r")
    lines = dist_file.readlines()

    dist_bw_places = [list(w.split()) for w in lines]
    dist_file.close()
    places = []
    for i in range(len(dist_bw_places) - 1):
        for j in range(3):
            if dist_bw_places[i][j] not in places and j != 1:
                places.append(dist_bw_places[i][j])
    return places, dist_bw_places

def src_in(places):
    src_loc = input("Enter the source point from the list above")
    if src_loc not in places:
        src_loc = src_in(places)
    return src_loc

def dest_in(places):
    dest_loc = input("Enter the destination point from the list above")
    if dest_loc not in places:
        dest_loc = dest_in(places)
    return dest_loc

def algo_in():
    algo = input("Enter the desired Algorithm(valid inputs are: 'BFS','DFS','IDS')")
    if algo not in ["BFS", "DFS", "IDS"]:
        algo = algo_in()
    return algo

def depth_in():
    try:
        depth = int(input("Enter the desired depth"))
        if depth > 0:
            return depth
        else:
            depth = depth_in()
    except ValueError:
        depth = depth_in()

def uninfromed_search(file_name):
    places, dist_bw_places = open_file(file_name)
    print("Please enter inputs as shown. All inputs are case sensitive")
    print(*places,sep= "\n")
    src_loc = src_in(places)
    dest_loc = dest_in(places)
    algo = algo_in()
    sample_graph = Graph()
    visited_nodes = []
    visted_node_names = []
    graph_data = sample_graph.graph_gen(dist_bw_places)
    if algo == "BFS":
        bfs_start(src_loc,dest_loc,graph_data,visited_nodes,visted_node_names)
    elif algo == "DFS":
        dfs_start(src_loc,dest_loc, graph_data,visited_nodes,visted_node_names)
    elif algo == "IDS":
        depth = depth_in()
        idls_start(Node(src_loc), dest_loc, graph_data, depth)

x = uninfromed_search("distances_input_file.txt")
