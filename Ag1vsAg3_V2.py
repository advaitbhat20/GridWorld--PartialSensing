from queue import PriorityQueue
from dataclasses import dataclass, field
import random
import math
import sys
import pandas as pd
from time import time


############### Agent 1 ########################

max = sys.maxsize
blocks_encountered = 0
total_trajectory_length = 0
blocks_encounteredA = 0
total_trajectory_lengthA = 0

# Node class for storing position, cost and heuristic for each grid encountered
class Node:
    # Initialize the class
    def __init__(self, position=None, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    def __ne__(self, other):
        return not (self.position == other.position)

    def __lt__(self, other):
        return (self.f < other.f)

    def __gt__(self, other):
        return (self.f > other.f)

    def __hash__(self):
        # hash(custom_object)
        return hash((self.position, self.parent))

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))

    #This returns the neighbours of the Node
    def get_neigbours(self, matrix):
        neighbour_cord = [(-1, 0),(0, -1),(0, 1),(1, 0)]
        current_x = self.position[0]
        current_y = self.position[1]
        neighbours = []
        for n in neighbour_cord:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                c = Node()
                c.position = (x, y)
                c.parent = self
                c.g = self.g + 1
                neighbours.append(c)
        return neighbours

####################################################################################

#creating a randomized grid world
def create_grid(n, p):
    matrix = [ [ 0 for i in range(n) ] for j in range(n) ]
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i==n-1 and j==n-1):
                matrix[i][j] = 0
            else:
                prob = random.uniform(0, 1)
                if prob >= p:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = 1
    return matrix

#print the gridworld/knowledge
def print_grid(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end=" ")
        print("")

def count_blocks(matrix):
    count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                count+=1
    return count

def calc_manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def calc_euclidean(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def calc_chebyshev(a,b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

####################################################################################
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field()

def Astar(knowledge_grid, start, end, flag=True, heuristic="manhattan"):
    grid_len = len(knowledge_grid)

    # Initialize a priority queue
    pQueue = PriorityQueue()
    pQueue.put(PrioritizedItem(0.0, start))
    closed_hash = {}    
    counter = 0

    while not pQueue.empty():
        # print(counter, len(pQueue.queue))
        if counter > 20000:
            return [None]

        counter+=1

        current = pQueue.get().item

        #Using dictionary instead of a list, to make retrival easier
        closed_hash[current.position] = True

        # Check if we have reached the goal, return the path
        if current == end:
            path = []
            while current != start:
                path.append(current.position)
                current = current.parent
            path.append(start.position)
            # Return reversed path
            return [path[::-1]]

        for n in current.get_neigbours(knowledge_grid):
            #check if neighbor is in closed set
            if n.position in closed_hash:
                continue
            #calculate heuristics for the neighbor
            if heuristic == "manhattan":
                n.h = calc_manhattan(n.position, [grid_len-1,grid_len-1])
            elif heuristic == "euclidean":
                n.h = calc_euclidean(n.position, [grid_len-1,grid_len-1])
            elif heuristic == "chebyshev":
                n.h = calc_chebyshev(n.position, [grid_len-1,grid_len-1])

            if flag:
                n.f = n.g + n.h
                #check if node is in priority queue if yes does it have lower value?

                #add n to priority queue
                (x, y) = n.position
                if knowledge_grid[x][y] != 1:
                    pQueue.put(PrioritizedItem(float(n.f), n))

            #When using ASTAR to verify our solution consider undiscovered node's g to be infinity
            else:
                if knowledge_grid[n.position[0]][n.position[1]] ==  '-':
                    n.g = max
                n.f = n.g + n.h
                #check if node is in priority queue if yes does it have lower value?

                #add n to priority queue
                (x, y) = n.position
                if knowledge_grid[x][y] != 1:
                    pQueue.put(PrioritizedItem(float(n.f), n))

    return [None]


def implement(matrix, knowledge, path):
    #follow the path provided by A-star and update the knowledge according to the actual grid
    for itr in range(1,len(path)):
        global total_trajectory_length
        total_trajectory_length += 1
        i = path[itr][0]
        j = path[itr][1]
        
        if matrix[i][j] == 1:
            global blocks_encountered
            knowledge[i][j] = 1
            blocks_encountered += 1
            return path[itr-1]
        else:
            knowledge[i][j] = 0
    return path[len(path)-1]

def repeated(matrix, knowledge, start, end, heuristic="manhattan"):
    #repeat A-star from the last node where the next node encountered is a block
    while True:
        res = Astar(knowledge, start, end, heuristic)
        path = res[0]
        if path:
            last = implement(matrix, knowledge, path)
            last_Node = Node(last)
            if path[len(path)-1] == last:
                break
            start = last_Node
        # if no path found. It means A-star was not able to find a path and the grid is unsolvable
        else:
            break
    return path

####################################################################################

############### Agent 3 ########################

from queue import PriorityQueue
from dataclasses import dataclass, field
import random
import math
import sys

max = sys.maxsize
neighbour_cord = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
neighbour_cardinal = [(-1, 0),(0, -1),(0, 1),(1, 0)]

# NodeAA class for storing position, cost and heuristic for each grid encountered
class NodeAA:
    # Initialize the class
    def __init__(self, position=None, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        #no. of neighboring  cells 
        self.nx = 0
        # no. of cells sensed to be blocked
        self.cx = 0
        # no. of cells confirmed to be blocked
        self.bx = 0
        # no. of cells confired to be empty
        self.ex = 0
        # no. of unconfirmed cells
        self.hx = 0
        # boolean variable to check if the cell has been visited
        self.visited = False
        # variable to check if the cell is blocked or unblocked or uncomfirmed - values can be "unconfirmed", "blocked", "empty"
        self.status = "unconfirmed"


    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    def __ne__(self, other):
        return not (self.position == other.position)

    def __lt__(self, other):
        return (self.f < other.f)

    def __gt__(self, other):
        return (self.f > other.f)

    def __hash__(self):
        # hash(custom_object)
        return hash((self.position, self.parent))

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))
    
    def print_attributes(self):
        print("Attributes for ", self.position)
        print("neighbors: ", self.nx)
        print("sensed blocked: ", self.cx)
        print("confirmed blocked: ", self.bx)
        print("confirmed empty: ", self.ex)
        print("unkown: ", self.hx)

    #This returns the neighbours of the NodeAA
    def get_neigbours(self, matrix):
        current_x = self.position[0]
        current_y = self.position[1]
        neighbours = []
        for n in neighbour_cardinal:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                c = NodeAA()
                c.position = (x, y)
                c.parent = self
                c.g = self.g + 1
                neighbours.append(c)
        return neighbours

    #This returns all the neighbours of the NodeAA
    def get_all_neigbours(self, matrix, parent):
        current_x = parent.position[0]
        current_y = parent.position[1]
        neighbours = []
        for n in neighbour_cord:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                c = NodeAA()
                c.position = (x, y)
                c.parent = self
                c.g = self.g + 1
                neighbours.append(c)
        return neighbours

    #This returns all the 8 or 3 neighbours of the NodeAA for sensing
    def partial_sensing(self, matrix):
        current_x = self.position[0]
        current_y = self.position[1]
        for n in neighbour_cord:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                c = hash_map[(x, y)]
                c.parent = self
                if matrix[x][y] == 1:
                    self.cx += 1


    def inference(self, grid_len, hash_map):
        flag = self.bx + self.cx + self.hx == self.nx
        if self.hx == 0 and flag:
            if self.ex + self.bx == self.nx:
                # print("All is known")
                return
        if (self.cx == self.bx and self.cx != 0) or (self.cx == self.bx and self.visited) and flag :
            # print("agent at" , self.position)
            for n in neighbour_cord:
                x = self.position[0] + n[0]
                y = self.position[1] + n[1]
                # print("neighbor update", x,y)
                if 0 <= x < grid_len and 0 <= y < grid_len:
                    if hash_map[(x, y)].status == "unconfirmed":
                        hash_map[(x, y)].status = "empty"
                        print("confirmed", self.bx)
                        print("sensed", self.cx)
                        print("empty!!",x,y)
                        print("agent:", self.position)
                        knowledge[x][y] = 0
                        self.ex = self.hx
                        self.hx = 0
        if (self.nx - self.cx == self.ex and self.cx != 0) or  (self.nx - self.cx == self.ex and self.cx == 0 and self.visited) and flag:
            self.bx += self.hx
            self.hx = 0
            # print("2nd inf!!!!")
            for n in neighbour_cord:                
                x = self.position[0] + n[0]
                y = self.position[1] + n[1]
                if 0 <= x < grid_len and 0 <= y < grid_len:
                    # print("inf", hash_map[(x, y)].status, hash_map[(x,y)].visited)
                    if hash_map[(x, y)].status == "uncomfirmed":
                        hash_map[(x, y)].status = "blocked"
                        print("total neigh", self.nx)
                        print("sensed", self.cx)
                        print("empty:", self.ex)
                        print("blocked!!",x,y)
                        print("agent:", self.position)
                        if knowledge[x][y] == '-':
                            knowledge[x][y] = 1
        

####################################################################################
@dataclass(order=True)
class PrioritizedItemAA:
    priority: float
    item: object = field()

def update_var(x, y, status):
    for n in neighbour_cord:
        n_x = x + n[0]
        n_y = y + n[1]
        if (n_x, n_y) in hash_map:
            c = hash_map[(n_x, n_y)]
            if status == "blocked":
                c.bx += 1
                c.hx -= 1
            
            if status == "empty":
                c.ex += 1
                c.hx -= 1

def AstarAA(knowledge_grid, start, end, flag=True, heuristic="manhattan"):
    grid_len = len(knowledge_grid)

    # Initialize a priority queue
    pQueue = PriorityQueue()
    pQueue.put(PrioritizedItemAA(0.0, start))
    closed_hash = {}    
    counter = 0

    while not pQueue.empty():
        # print(counter, len(pQueue.queue))
        if counter > 20000:
            return [None]

        counter+=1

        current = pQueue.get().item

        #Using dictionary instead of a list, to make retrival easier
        closed_hash[current.position] = True

        # Check if we have reached the goal, return the path
        if current == end:
            path = []
            while current != start:
                path.append(current.position)
                current = current.parent
            path.append(start.position)
            # Return reversed path
            return [path[::-1]]

        for n in current.get_neigbours(knowledge_grid):
            #check if neighbor is in closed set
            if n.position in closed_hash:
                continue
            #calculate heuristics for the neighbor
            if heuristic == "manhattan":
                n.h = calc_manhattan(n.position, [grid_len-1,grid_len-1])
            elif heuristic == "euclidean":
                n.h = calc_euclidean(n.position, [grid_len-1,grid_len-1])
            elif heuristic == "chebyshev":
                n.h = calc_chebyshev(n.position, [grid_len-1,grid_len-1])

            if flag:
                n.f = n.g + n.h
                #check if node is in priority queue if yes does it have lower value?

                #add n to priority queue
                (x, y) = n.position
                if knowledge_grid[x][y] != 1:
                    pQueue.put(PrioritizedItemAA(float(n.f), n))

            #When using ASTAR to verify our solution consider undiscovered node's g to be infinity
            else:
                if knowledge_grid[n.position[0]][n.position[1]] ==  '-':
                    n.g = max
                n.f = n.g + n.h
                #check if node is in priority queue if yes does it have lower value?

                #add n to priority queue
                (x, y) = n.position
                if knowledge_grid[x][y] != 1:
                    pQueue.put(PrioritizedItemAA(float(n.f), n))

    return [None]


def implementAA(matrix, knowledge, path):
    #follow the path provided by A-star and update the knowledge according to the actual grid
    # print(path)
    
    #adding in the actual traversed path
    if path[0] not in actual_path:
        actual_path.append(path[0])
    actual_path.append(path[0])
    for itr in range(1,len(path)):
        global total_trajectory_lengthA
        total_trajectory_lengthA += 1
        x = path[itr][0]
        y = path[itr][1]
        #get the current node from the hash_map containg all nodes and update visited
        current = hash_map[(x,y)]
        current.visited = True

        #if blocked update the knowledge and node
        if matrix[x][y] == 1:
            global blocks_encounteredA
            blocks_encounteredA += 1
            current.status = "blocked"
            knowledge[x][y] = 1
            update_var(x, y, "blocked")
            return path[itr-1]
        elif matrix[x][y] == 0:
            #adding in the actual traversed path
            if path[itr] not in actual_path:
                actual_path.append(path[itr])
            actual_path.append(path[itr])
            current.status = "empty"
            knowledge[x][y] = 0
            update_var(x, y, "empty")
        

        #call partial sensing
        current.partial_sensing(matrix)

        # #update the current node using neighbor 3*3 nodes
        # current.ex = 0
        # current.bx = 0
        # current.hx = current.nx
        # for n in neighbour_cord:
        #     n_x = current.position[0] + n[0]
        #     n_y = current.position[1] + n[1]
        #     if 0 <= n_x < len(matrix) and 0 <= n_y < len(matrix):
        #         c = hash_map[(n_x, n_y)]
        #         if c.status == "empty":
        #             current.ex+=1
        #             current.hx-=1
        #         if c.status == "blocked":
        #             current.bx+=1
        #             current.hx-=1
        
        #call inference method
        current.inference(grid_len, hash_map)

        #call inference on neighbours
        for n in neighbour_cord:
            n_x = current.position[0] + n[0]
            n_y = current.position[1] + n[1]
            if 0 <= n_x < len(matrix) and 0 <= n_y < len(matrix):             
                c = hash_map[(n_x, n_y)]
                c.inference(grid_len, hash_map)

        #call inference on entire path
        # for i in range(itr, len(path)):
        #     x = path[itr][0]
        #     y = path[itr][1]
        #     #get the current node from the hash_map containg all nodes and update visited
        #     curr = hash_map[(x,y)]
        #     curr.inference(grid_len, hash_map)
        #     if curr.status == "blocked":
        #         print("blocked!!!!")
        #         knowledge[x][y] = 1
        #         return path[i-1]
        #     elif curr.status == "empty":
        #         knowledge[x][y] = 0

    return path[len(path)-1]

def agent_3AA(matrix, knowledge, start, end, heuristic="manhattan"):
    #repeat A-star from the last node where the next node encountered is a block
    while True:
        res = AstarAA(knowledge, start, end, heuristic)
        path = res[0]
        if path:
            last = implementAA(matrix, knowledge, path)
            last_Node = NodeAA(last)
            if path[len(path)-1] == last:
                break
            start = last_Node

        # if no path found. It means A-star was not able to find a path and the grid is unsolvable
        else:
            break
        #adding in shortest path
    return path

####################################################################################


if __name__ == "__main__":
    
    result = {
        "Probability": [],
        "Density": [],
        "blocks_for_A1":[],
        "blocks_for_A3": [],
        "trajectory_length_A1": [],
        "trajectory_length_A3": [],
        "shortest_path_A1": [],
        "shortest_path_A3": [],
        'time_for_A1': [],
        'time_for_A3': []
    }

    p0 = 34
    grid_len = 101

    for i in range(30,p0, 4):
        iter=i
        for j in range(1):
            
            print("iteration", iter)
            result["Probability"].append(iter/100)
            actual_path = []
            shortest_path = []
            hash_map = {}
            neighbour_cord = [(-1,-1),(-1, 0),(-1,1),(0, -1),(0, 1),(1,-1),(1, 0),(1,1)]
            for i in range(grid_len):
                for j in range(grid_len):
                    c = NodeAA((i, j))
                    c.status = "uncomfirmed"
                    if (i, j) in [(0, 0), (0, grid_len-1), (grid_len-1, 0), (grid_len-1, grid_len-1)]:
                        c.nx = 3
                        c.hx = c.nx
                    elif i == grid_len - 1 or j == grid_len-1 or i == 0 or j == 0:
                        c.nx = 5
                        c.hx = c.nx
                    else:
                        c.nx = 8
                        c.hx = c.nx
                    hash_map[(i,j)] = c
            
            c = hash_map[(0,0)]
            c.status = "empty"
            c = hash_map[(grid_len-1,grid_len-1)]
            c.status = "empty"


            # print(hash_map)

            #creating random grid world by providing it a p value
            matrix = create_grid(grid_len, 0.26)
            blocks = sum([i.count(1) for i in matrix])
            desnity = blocks/(grid_len*grid_len)
            print("density", desnity)
            result["Density"].append(desnity)
            print("FULL KNOWN GRIDWORLD")
            print_grid(matrix)

            #initializing knowledge of agent
            knowledge = [ [ "-" for i in range(grid_len) ] for j in range(grid_len) ]
            
            start = NodeAA()
            start.position = (0, 0)
            knowledge[0][0] = 0
            knowledge[grid_len-1][grid_len-1] = 0
            goal = NodeAA()
            goal.position = (grid_len-1, grid_len-1)

            path_len = 0

            begin = time()
            res = agent_3AA(matrix, knowledge, start, goal, "manhattan")
            end = time() - begin
            print("time:", end)
            
            if res != None:
                # print("actual path",actual_path)
                #applying A-star from start on complete updated knowledge of the agent
                start = NodeAA()
                start.position = (0, 0)
                shortest_path = AstarAA(knowledge, start, goal, False, "manhattan")[0]

                # print("shortest path", shortest_path)
                # print("length of actual path", len(actual_path))
                # print("length of shortest path", len(shortest_path))
                path_len = len(shortest_path)
            else:
                print("NO PATH FOUND")

            print("EXPLORED GRIDWORLD")
            print_grid(knowledge)
            print("Agent 3 blocks encountered", blocks_encounteredA)
            print("length of shortest path", path_len)
            print("Agent 3 trajectory length", total_trajectory_lengthA)
            result["blocks_for_A3"].append(blocks_encounteredA)
            result["trajectory_length_A3"].append(total_trajectory_lengthA)
            result["shortest_path_A3"].append(path_len)
            result["time_for_A3"].append(end)


            ##################################################################################################################

            #initializing knowledge of agent
            knowledge = [ [ "-" for i in range(grid_len) ] for j in range(grid_len) ]
            
            start = Node()
            start.position = (0, 0)
            goal = Node()
            goal.position = (grid_len-1, grid_len-1)

            res = repeated(matrix, knowledge, start, goal, "manhattan")
            print()
            print("EXPLORED GRIDWORLD Agent1")
            print_grid(knowledge)

            #applying A-star from start on complete updated knowledge of the agent
            start = Node()
            start.position = (0, 0)

            begin = time()
            res = Astar(knowledge, start, goal, False, "manhattan")
            end = time() - begin            
            print("time:", end)

            path_len = 0
            if res and res[0]:
                path_len = len(res[0])

            # print("final path", res)
            print("Agent1 blocks encountered", blocks_encountered)
            print("Agent 1 Shortest Path", path_len)
            print("Agent 1 trajectory length", total_trajectory_length)
            
            result["blocks_for_A1"].append(blocks_encountered)
            result["trajectory_length_A1"].append(total_trajectory_length)
            result["shortest_path_A1"].append(path_len)
            result["time_for_A1"].append(end)
    
    data = pd.DataFrame(result)
    data.to_csv("Agent1VSAgent3.csv", index=False, encoding='utf-8')