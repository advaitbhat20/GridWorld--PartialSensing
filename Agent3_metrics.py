from queue import PriorityQueue
from dataclasses import dataclass, field
import random
import math
import sys

max = sys.maxsize
neighbour_cord = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
neighbour_cardinal = [(-1, 0),(0, -1),(0, 1),(1, 0)]
blocks_encountered = 0

# Node class for storing position, cost and heuristic for each grid encountered
class Node:
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
        print("neighbors: ", self.nx)
        print("sensed blocked: ", self.cx)
        print("confirmed blocked: ", self.bx)
        print("confirmed empty: ", self.ex)
        print("unkown: ", self.hx)

    #This returns the neighbours of the Node
    def get_neigbours(self, matrix):
        current_x = self.position[0]
        current_y = self.position[1]
        neighbours = []
        for n in neighbour_cardinal:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                c = Node()
                c.position = (x, y)
                c.parent = self
                c.g = self.g + 1
                neighbours.append(c)
        return neighbours

    #This returns all the 8 or 3 neighbours of the Node for sensing
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
        # print("sensed to be blocked:", self.cx)

    def inference(self, matrix, hash_map):
        if self.cx == self.bx:
            self.ex += self.hx
            self.hx = 0
            for n in neighbour_cord:
                x = self.position[0] + n[0]
                y = self.position[1] + n[1]
                if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                    if hash_map[(x, y)].status == "uncomfirmed":
                        hash_map[(x, y)].status = "empty"
        if self.nx - self.cx == self.ex:
            self.bx += self.hx
            self.hx = 0
            for n in neighbour_cord:
                x = self.position[0] + n[0]
                y = self.position[1] + n[1]
                if 0 <= x < len(matrix) and 0 <= y < len(matrix):
                    if hash_map[(x, y)].status == "uncomfirmed":
                        hash_map[(x, y)].status = "blocked"
        if self.hx == 0:
            if self.ex + self.bx == self.nx:
                print("All is known")
    
    

                    
        

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
    # print(path)
    for itr in  range(1,len(path)):
        x = path[itr][0]
        y = path[itr][1]
        #get the current node from the hash_map containg all nodes and update visited
        current = hash_map[(x,y)]
        current.visited = True

        #if blocked update the knowledge and node
        if matrix[x][y] == 1:
            current.status = "blocked"
            knowledge[x][y] = 1
            global blocks_encountered
            blocks_encountered += 1
            return path[itr-1]
        elif matrix[x][y] == 0:
            current.status = "empty"
            knowledge[x][y] = 0

        #call partial sensing
        current.partial_sensing(matrix)

        #update the current node using neighbor 3*3 nodes
        for n in neighbour_cord:
            n_x = current.position[0] + n[0]
            n_y = current.position[1] + n[1]
            if 0 <= n_x < len(matrix) and 0 <= n_y < len(matrix):
                c = hash_map[(n_x, n_y)]
                if c.status == "empty":
                    current.ex+=1
                    current.hx-=1
                if c.status == "blocked":
                    current.bx+=1
                    current.hx-=1
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
if __name__ == "__main__":
    grid_len = 101

    hash_map = {}
    neighbour_cord = [(-1,-1),(-1, 0),(-1,1),(0, -1),(0, 1),(1,-1),(1, 0),(1,1)]
    for i in range(grid_len):
        for j in range(grid_len):
            c = Node((i, j))
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

    # print(hash_map)

    #creating random grid world by providing it a p value
    matrix = create_grid(grid_len, 0.3)
    print("FULL KNOWN GRIDWORLD")
    print_grid(matrix)

    #initializing knowledge of agent
    knowledge = [ [ "-" for i in range(grid_len) ] for j in range(grid_len) ]
    
    start = Node()
    start.position = (0, 0)
    goal = Node()
    goal.position = (grid_len-1, grid_len-1)

    res = repeated(matrix, knowledge, start, goal, "manhattan")
    print()
    print("EXPLORED GRIDWORLD")
    print_grid(knowledge)

    #applying A-star from start on complete updated knowledge of the agent
    start = Node()
    start.position = (0, 0)
    res = Astar(knowledge, start, goal, False, "manhattan")

    print("final path", res)
    print("blocks encountered", blocks_encountered)