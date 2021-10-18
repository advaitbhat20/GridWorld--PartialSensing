from queue import PriorityQueue
from dataclasses import dataclass, field
import random
import math
import sys

max = sys.maxsize
neighbour_cord = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
neighbour_cardinal = [(-1, 0),(0, -1),(0, 1),(1, 0)]

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
        print("Attributes for ", self.position)
        print("neighbors: ", self.nx)
        print("sensed blocked: ", self.cx)
        print("confirmed blocked: ", self.bx)
        print("confirmed empty: ", self.ex)
        print("unkown: ", self.hx)
        print("visted: ", self.visited)
        print("status: ", self.status)


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
                hash_map[(x, y)].parent = self
                if matrix[x][y] == 1:
                    self.cx += 1

    def inference(self, grid_len, hash_map):
        # print(flag)
        if self.hx <= 0:
            if self.ex + self.bx == self.nx:
                # print("All is known")
                return
        if (self.cx == self.bx and self.cx != 0) or (self.cx == self.bx and self.visited) and self.status!= "blocked":
            # print("agent at" , self.position)
        
            if self.ex + self.hx + self.bx == self.nx:
                self.ex += self.hx
                self.hx = 0
                for n in neighbour_cord:
                    x = self.position[0] + n[0]
                    y = self.position[1] + n[1]
                    if 0 <= x < grid_len and 0 <= y < grid_len:
                        if hash_map[(x, y)].status == "unconfirmed":
                            hash_map[(x, y)].status = "empty"
                            if knowledge[x][y] == '-':
                                knowledge[x][y] = matrix[x][y]
                            hash_map[(x, y)].inference(grid_len, hash_map)
                            
        if ((self.nx - self.cx == self.ex and self.cx != 0) or (self.nx - self.cx == self.ex and self.cx == 0 and self.visited)) and self.status != "blocked":
            if self.ex + self.hx + self.bx == self.nx:
                self.bx += self.hx
                self.hx = 0
                for n in neighbour_cord:                
                    x = self.position[0] + n[0]
                    y = self.position[1] + n[1]
                    if 0 <= x < grid_len and 0 <= y < grid_len:
                        if hash_map[(x, y)].status == "unconfirmed":
                            hash_map[(x, y)].status = "blocked"
                            if knowledge[x][y] == '-':
                                knowledge[x][y] = matrix[x][y]
                                update_neighbours(x,y, "blocked")
                            hash_map[(x, y)].inference(grid_len, hash_map)


        #CHESS KNIGHT INFERRENCE - FIGURES 15 -17 Assignment
        cross_coords = [(-1, -2), (-1, 2), (1, -2), (1, 2), (2, 1), (-2, 1), (2, -1), (-1, -2)]
        if ((self.cx == 2)):
            for n in cross_coords:
                flag = True
                x = self.position[0]
                y = self.position[1]
                x2 = self.position[0] + n[0]
                y2 = self.position[1] + n[1]
                if (x2, y2) in hash_map:
                    if x2 > x and y2 > y:
                        for ne in neighbour_cord:
                            #CHECKING IF THE REQUIRED NEIGHBORS FLAGS ARE NOT UNCONFIRMED
                            if (x + ne[0] == x and y + ne[1] > y) or (x + ne[0] > x and y + ne[1] == y) or (x+ne[0] > x and y+ne[1] > y ):
                                continue
                            n_x = x + ne[0]
                            n_y = y + ne[1]
                            if (n_x, n_y) in hash_map and (hash_map[(n_x, n_y)].status == "unconfirmed"):
                                flag = False
                            
                            #CHECKING IF THE REQUIRED NEIGHBORS FLAGS ARE NOT UNCONFIRMED
                            if  (x2 + ne[0] < x2 and y2 +ne[1] < y2) or (x2 + ne[0] == x2 and y2 +ne[1] < y2) or (x2 + ne[0] < x2 and y2 +ne[1] == y2):
                                continue
                            n_x2 = x2 + ne[0]
                            n_y2 = y2 + ne[1]
                            if (n_x2, n_y2) in hash_map and (hash_map[(n_x2, n_y2)].status == "unconfirmed"):
                                flag = False
                    if x2 < x and y2 > y:
                        for ne in neighbour_cord:
                            #CHECKING IF THE REQUIRED NEIGHBORS FLAGS ARE NOT UNCONFIRMED
                            if (x + ne[0] < x and y + ne[1] == y) or (x + ne[0] < x and y + ne[1] > y) or (x+ne[0] == x and y+ne[1] > y ):
                                continue
                            n_x = x + ne[0]
                            n_y = y + ne[1]
                            if (n_x, n_y) in hash_map and (hash_map[(n_x, n_y)].status == "unconfirmed"):
                                flag = False
                            
                            #CHECKING IF THE REQUIRED NEIGHBORS FLAGS ARE NOT UNCONFIRMED
                            if  (x2 + ne[0] == x2 and y2 +ne[1] < y2) or (x2 + ne[0] > x2 and y2 +ne[1] < y2) or (x2 + ne[0] > x2 and y2 +ne[1] == y2):
                                continue
                            n_x2 = x2 + ne[0]
                            n_y2 = y2 + ne[1]
                            if (n_x2, n_y2) in hash_map and (hash_map[(n_x2, n_y2)].status == "unconfirmed"):
                                flag = False

                    #Setting the inferred cells to either 1 or zero, and updating Nodes
                    if flag and hash_map[(x2, y2)].cx == 1:
                        hash_map[(x, y)].bx += 1
                        hash_map[(x, y)].hx -= 1
                        hash_map[(x2, y2)].ex +=1
                        hash_map[(x2, y2)].hx -= 1
                        if (x-1, y) in hash_map and (x2+1, y) in hash_map:
                            knowledge[x-1][y] = matrix[x-1][y]
                            knowledge[x2+1][y] = matrix[x2+1][y]


        #ALTERNATE NEIGHBOR INFERRENCE - FIGURES 17 -19 Assignment
        alternate_coords = [(0, 2), (2, 0), (-2, 0), (0, -2)]
        if ((self.cx == 1)):
            #CHECKING FOR ALL ALTERNATE NEIGHBOURS 
            for n in alternate_coords:
                flag = True
                x = self.position[0]
                y = self.position[1]
                x2 = self.position[0] + n[0]
                y2 = self.position[1] + n[1]
                if (x2, y2) in hash_map:
                    if y2 > y:
                        for ne in neighbour_cord:
                            if (x + ne[0] == x and y + ne[1] > y) or (x + ne[0] > x and y + ne[1] == y) or (x+ne[0] > x and y+ne[1] > y ):
                                continue
                            n_x = x + ne[0]
                            n_y = y + ne[1]
                            if (n_x, n_y) in hash_map and (hash_map[(n_x, n_y)].status == "unconfirmed"):
                                flag = False
                            if (x2 + ne[0] == x2 and y2 +ne[1] < y2) or (x2 + ne[0] > x2 and y2 +ne[1] < y2) or (x2 + ne[0] > x2 and y2 +ne[1] == y2):
                                continue
                            n_x2 = x2 + ne[0]
                            n_y2 = y2 + ne[1]
                            if (n_x2, n_y2) in hash_map and (hash_map[(n_x2, n_y2)].status == "unconfirmed"):
                                flag = False
                        if flag and hash_map[(x2, y2)].cx == 0:
                            hash_map[(x,y)].bx += 1
                            hash_map[(x,y)].hx -= 1
                            hash_map[(x,y)].ex += 2
                            hash_map[(x,y)].hx -= 2
                            hash_map[(x2,y2)].ex += 3
                            hash_map[(x2,y2)].hx -= 3

                            knowledge[x][y+1] = matrix[x][y+1]

                            knowledge[x2-1][y2] = matrix[x2-1][y2]
                            knowledge[x2-1][y2+1] = matrix[x2-1][y2+1]
                            knowledge[x2][y2+1] = matrix[x2][y2+1]

    
####################################################################################
##########################    HELPER FUNCTIONS    ##################################
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

####################################################################################
##########################   ASTAR and Implement  ##################################
####################################################################################


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field()

def update_neighbours(x, y, status):
    for n in neighbour_cord:
        n_x = x + n[0]
        n_y = y + n[1]
        if (n_x, n_y) in hash_map:
            if status == "blocked":
                hash_map[(n_x, n_y)].bx += 1
                hash_map[(n_x, n_y)].hx -= 1
            if status == "empty":
                hash_map[(n_x, n_y)].ex += 1
                hash_map[(n_x, n_y)].hx -= 1

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
        
        x = path[itr][0]
        y = path[itr][1]
        #get the current node from the hash_map containg all nodes and update visited
        hash_map[(x,y)].visited = True

        #if blocked update the knowledge and node
        if matrix[x][y] == 1:
            if hash_map[(x,y)].status == "unconfirmed":
                hash_map[(x,y)].status = "blocked"
                
                knowledge[x][y] = 1
                update_neighbours(x, y, "blocked")
                hash_map[(x,y)].inference(grid_len, hash_map)
                return path[itr-1]
        elif matrix[x][y] == 0:
            #call partial sensing

            if hash_map[(x,y)].status == "unconfirmed":
                hash_map[(x,y)].status = "empty"
                knowledge[x][y] = 0
                hash_map[(x,y)].partial_sensing(matrix)
                update_neighbours(x, y, "empty")
                #call inference method
                hash_map[(x,y)].inference(grid_len, hash_map)

        #call inference on neighbours
        for n in neighbour_cord:
            n_x = hash_map[(x,y)].position[0] + n[0]
            n_y = hash_map[(x,y)].position[1] + n[1]
            if 0 <= n_x < len(matrix) and 0 <= n_y < len(matrix):
                if hash_map[(n_x, n_y)].status != "blocked":
                    hash_map[(n_x, n_y)].inference(grid_len, hash_map)

    return path[len(path)-1]

def agent_4(matrix, knowledge, start, end, heuristic="manhattan"):
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
        #adding in shortest path
    return path

####################################################################################
#################################     MAIN    ######################################
####################################################################################

if __name__ == "__main__":
    grid_len = 10
    actual_path = []
    shortest_path = []
    hash_map = {} #Dictionary to store nodes
    neighbour_cord = [(-1,-1),(-1, 0),(-1,1),(0, -1),(0, 1),(1,-1),(1, 0),(1,1)]
    for i in range(grid_len):
        for j in range(grid_len):
            c = Node((i, j))
            c.status = "unconfirmed"
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

    #creating random grid world by providing it a p value
    matrix = create_grid(grid_len, 0.34)
    # matrix = [
    #     [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    # ]

    print("FULL KNOWN GRIDWORLD")
    print_grid(matrix)

    #initializing knowledge of agent
    knowledge = [ [ "-" for i in range(grid_len) ] for j in range(grid_len) ]
    
    #Set the start and goal
    start = Node()
    start.position = (0, 0)
    goal = Node()
    goal.position = (grid_len-1, grid_len-1)

    #Call the Agent
    res = agent_4(matrix, knowledge, start, goal, "manhattan")

    #Using A* to find the shortest path on the discovered grid
    start = Node()
    start.position = (0, 0)
    shortest_path = Astar(knowledge, start, goal, False, "manhattan")[0]
    if res != None and shortest_path!=None:
        print("EXPLORED GRIDWORLD")
        print_grid(knowledge)
        print("shortest path", shortest_path)
        print("length of shortest path", len(shortest_path))
    else:
        print("NO PATH FOUND")
        print_grid(knowledge)

    
    #CHECKING CONSISTENCY OF DATA IN KNOWLEDGE AND MATRIX
    for i in range(grid_len):
        for j in range(grid_len):
            if (matrix[i][j] != knowledge[i][j] and knowledge[i][j] != '-'):
                print(i, j, "mismatch should be ", matrix[i][j], " is ", knowledge[i][j])
                hash_map[(i,j)].print_attributes()
                break

    
