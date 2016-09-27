class Graph(object):
    def graph_gen(self, src, dest, dist_bw_places):
        self.src = src
        self.dest = dest
        self.dist_bw_places = dist_bw_places
        temp = []
        for i in dist_bw_places:
            temp.append(i[::-1])
        self.dist_bw_places += temp
        self.nodes = set(i[0] for i in self.dist_bw_places)
        graph_data = {}
        graph_weight = {}
        for node in self.nodes:
            for i in self.dist_bw_places:
                if node == i[0] and node not in graph_data.keys():
                    graph_data.update({node: [i[2]]})
                    graph_weight.update({node: [i[1]]})
                elif node == i[0] and node in graph_data.keys():
                    graph_data[node].append(i[2])
                    graph_weight[node].append(i[1])
        return graph_data, graph_weight


class BFS(object):
    def explore(self, node, graph_data):
        self.node = node
        self.graph_data = graph_data
        return self.graph_data[self.node]

    def bfs_start(self, src_node, dest_node, graph_data):
        self.src_node = src_node
        self.dest_node = dest_node
        self.graph_data = graph_data
        visted_nodes = []
        pipe = []
        pipe.append(self.src_node)
        curr_node = pipe.pop(0)
        while True:
            pipe += self.explore(curr_node, self.graph_data)
            if curr_node not in visted_nodes:
                visted_nodes.append(curr_node)
            if self.dest_node == curr_node:
                break
            curr_node = pipe.pop(0)
        print("Traversed path for BFS\n" + "\n".join(visted_nodes))


class DFS(object):
    def explore(self, node, graph_data):
        self.node = node
        self.graph_data = graph_data
        return self.graph_data[self.node]

    def dfs_start(self, src_node, dest_node, graph_data):
        self.src_node = src_node
        self.dest_node = dest_node
        self.graph_data = graph_data
        self.visted_nodes = []
        pipe = []
        pipe.append(self.src_node)
        while self.dest_node not in self.visted_nodes:
            self.visted_nodes.append(self.pipe[0])
            for i in self.explore(self.pipe.pop(0), self.graph_data):
                if i not in self.visted_nodes:
                    if i in self.pipe:
                        self.pipe.remove(i)
                    self.pipe.insert(0, i)
            print(self.visted_nodes)
            print(self.pipe)
        print("Traversed path for DFS\n" + "\n".join(self.visted_nodes))


class IDS(object):
    def explore(self, node, graph_data):
        self.node = node
        self.graph_data = graph_data
        return self.graph_data[self.node]

    def dls(self, curr_node, dest_node, depth, visted_nodes):
        self.curr_node = curr_node
        self.dest_node = dest_node
        self.depth = depth
        pipe = []
        for node in (self.explore(pipe[0])):
            if self.depth <= depth and node not in visted_nodes:
                visted_nodes = pipe.pop()
                pipe.insert(0, node)


class DataInput(object):
    def open_file(self, file_name):
        # Reading data from the input file and making a list of lists from that
        #  The name of the list is dist_bw_places
        self.file_name = file_name
        dist_file = open(self.file_name, "r")
        lines = dist_file.readlines()

        dist_bw_places = [list(w.split()) for w in lines]
        dist_file.close()
        places = []
        for i in range(len(dist_bw_places) - 1):
            for j in range(3):
                if dist_bw_places[i][j] not in places and j != 1:
                    places.append(dist_bw_places[i][j])
        return places, dist_bw_places

    def display(self, places):
        # print('\n'.join(places))
        # print("Enter exactly as shown above")
        pass

    def src_in(self, places):
        self.src_loc = "Arad"
        # input("Enter the source point from the list above")
        if self.src_loc not in places:
            self.src_in(places)
        return self.src_loc

    def dest_in(self, places):
        self.dest_loc = "Bucharest"
        # (input("Enter the destination point from the list above"))
        if self.dest_loc not in places:
            self.dest_in(places)
        return self.dest_loc

    def algo_in(self):
        self.algo = "BFS"
        # (input("Enter the desired Algorithm(valid inputs are: 'BFS','DFS')"))
        if self.algo not in ["BFS", "DFS", "IDS"]:
            self.algo_in()
        return self.algo

    def depth_in(self):
        self.depth = 2  # (input("Enter the desired depth"))
        try:
            if int(self.depth) > 0:
                return self.depth
            else:
                self.depth_in()
        except ValueError:
            self.depth_in()

    def __init__(self, file_name):

        self.file_name = file_name
        self.places, self.dist_bw_places = self.open_file(self.file_name)
        self.display(self.places)
        self.src_loc = self.src_in(self.places)
        self.dest_loc = self.dest_in(self.places)
        self.algo = self.algo_in()
        sample_graph = Graph()
        self.graph_data, self.graph_weight = sample_graph.graph_gen(
            self.src_loc, self.dest_loc, self.dist_bw_places)
        # print (self.graph_data)
        # print(self.graph_weight)
        if self.algo == "BFS":
            sample_bfs = BFS()
            sample_bfs.bfs_start(self.src_loc, self.dest_loc, self.graph_data)
        elif self.algo == "DFS":
            sample_dfs = DFS()
            sample_dfs.dfs_start(self.src_loc, self.dest_loc, self.graph_data)
        elif self.algo == "IDS":
            sample_ids = IDS()
            self.depth = self.depth_in()
            # sample_ids.ids_start(self.src_loc,self.dest_loc,self.graph_data,self.depth)
            sample_ids.depth(self.src_loc, self.graph_data, self.depth)


x = DataInput("distances_input_file.txt")
