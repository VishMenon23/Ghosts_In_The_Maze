from pickle import TRUE
import random
import numpy as np
from queue import PriorityQueue
from collections import deque
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt

no_of_mazes = 60        
row_size = 50
col_size = 50
Maze_Set = {}                               # Dictionary used to store the usable mazes
Path_Set = {}                               # Dictionary used to store the paths from the source to the destination of the mazes in Maze_Set

class Matrix:
  def __init__(self, row, column,distance, parent, parentx, parenty):
    self.row = row
    self.column = column
    self.distance = distance
    self.parent = parent
    self.parentx = parentx
    self.parenty = parenty
    self.g = 0
    self.h = 0
    self.f = 0    

def common_member(l1, l2):                  # Returns True if any member of list 'a' is present in list 'b'
    result = False
    for a in l1:
        for b in l2:
            if a == b:
                result = True
                return result 
    return result

def Maze_Creation():
    count = 0
    count_bfs=0
    i=0
    while i< no_of_mazes:
        p1 = 0.28                                                                                           # Probability to define a cell as a wall. 
        p2 = 0.72                                                                                           # Probability to define an unblocked cell. 
        matrix = np.array(np.random.choice([0, 1], size=(row_size,col_size), p=[p1, p2]))                   # Maze Creation with 1s and 0s. 1 represents an unblocked cell while 0 represents a wall
        visited = np.array(np.random.choice(['N', 'N'], size=(row_size, col_size), p=[p1, p2]))             # A 2d matrix of the same dimension created to keep track of the visited nodes. 
        
        matrix[0][0] = 1                                                                                    # Making sure the source and destination nodes are unblocked
        matrix[row_size-1][col_size-1] = 1
        source = Matrix(0,0,0,None,0,0)                                                                     # An object of the class Matrix is created to denote the starting cell
        q = deque()                                                                                         # A deque is created to keep track of the cells the current cell can travel to. 
        q.append(source)
        visited[source.row][source.column] = 'V'
        while (q):                                                                                          # BFS is implemented to make sure that there is a path from the source to the destination
            temp = q.popleft()
            if(temp.row == row_size-1 and temp.column == col_size-1):                                       # Goal has been reached
                ans = temp.distance
                path = []
                t = temp
                while(t):
                    l = []
                    l.append(t.parentx)
                    l.append(t.parenty)
                    path.append(l)
                    t = t.parent
            x = temp.row
            y = temp.column
            l = []
            l.append(x)
            l.append(y)
            if (y - 1 >= 0 and matrix[x][y - 1] == 1 and visited[x][y - 1] == 'N'):                         # Inserting the adjacent nodes if they have not yet been visited and they are unblocked.
                q.append(Matrix(x,y - 1, temp.distance+1,temp,x,y))
                visited[x][y-1] = 'V'
            if (y + 1 <= col_size-1 and matrix[x][y + 1] == 1 and visited[x][y + 1] == 'N'):
                q.append(Matrix(x, y + 1,temp.distance+1,temp,x,y))
                visited[x][y+1] = 'V'
            if (x - 1 >= 0 and matrix[x - 1][y] == 1 and visited[x - 1][y] == 'N'):
                q.append(Matrix(x - 1, y,temp.distance+1,temp,x,y))
                visited[x-1][y] = 'V'
            if (x + 1 <= row_size-1 and matrix[x + 1][y] == 1 and visited[x + 1][y] == 'N'):
                q.append(Matrix(x + 1, y,temp.distance+1,temp,x,y))
                visited[x+1][y] = 'V'
        
        if (visited[row_size-1][col_size-1] == 'V'):                                                        # Once the path is found
            Maze_Set[count] = matrix                                                                        # The maze is stored into a dictionary with key being the maze number and value the maze itself
            Path_Set[count] = path                                                                          # The path is stored into a dictionary with key being the maze number and value the BFS path from source to destination for that maze
            count = count + 1
            i=i+1                                                  

#A-Star heuristic 
def heuristic_evaluation(cell1,cell2):
    x1,y1=cell1
    x2,y2=cell2
    return abs(x1-x2) + abs(y1-y2)                                                          # Manhattan Distance between the cells.         

def astar_original(temp_matrix, s, temp_ghost_position):
    count_astar=0
    source=(s[0],s[1])
    g_score={}                                                                  # Dictionary to store the distance between the source and each cell
    f_score={}                                                                  # Dictionary to store the value of g(n) + h(n)
    for i in range(row_size):
        for j in range(col_size):
            g_score[(i,j)]=float('inf')                                         # Initializing the g and f value of each cell to a maximum 
            f_score[(i,j)]=float('inf')
    g_score[source]=0                                                           # Initializing the scores of the source node
    f_score[source]=heuristic_evaluation(source,(row_size-1,col_size-1))                                                 
    pq=PriorityQueue()
    pq.put((heuristic_evaluation(source,(row_size-1,col_size-1)),heuristic_evaluation(source,(row_size-1,col_size-1)),source))            
    Path={}
    flag=0
    while not pq.empty():
        current_cell=pq.get()[2]                                                                       # The cell with the smallest f score is returned by the priority queue
        if current_cell==(row_size-1,col_size-1):                                                      # The agent has reached the destination
            flag=1
            break
        x = current_cell[0]
        y = current_cell[1]
        l1=[x,y-1]
        l2=[x,y+1]
        l3=[x-1,y]
        l4=[x+1,y]                                                                                      # The neighbors are now examined
        if (y - 1 >= 0 and temp_matrix[x][y - 1] == 1 and l1 not in temp_ghost_position):               # Left child
            child_cell=(x,y-1)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(row_size-1,col_size-1))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(row_size-1,col_size-1)),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                   
        if (y + 1 <= col_size-1 and temp_matrix[x][y + 1] == 1 and l2 not in temp_ghost_position):      # Right child
            child_cell=(x,y+1)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(row_size-1,col_size-1))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(row_size-1,col_size-1)),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                    
        if (x - 1 >= 0 and temp_matrix[x - 1][y] == 1 and l3 not in temp_ghost_position):                # Up child
            child_cell=(x-1,y)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(row_size-1,col_size-1))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(row_size-1,col_size-1)),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
                      
        if (x + 1 <= row_size-1 and temp_matrix[x + 1][y] == 1 and l4 not in temp_ghost_position):     # Down child
            child_cell=(x+1,y)
            temp_g_score=g_score[current_cell]+1
            temp_f_score=temp_g_score+heuristic_evaluation(child_cell,(row_size-1,col_size-1))

            if temp_f_score < f_score[child_cell]:                                                      # If evaluated 'f' score is less than the existing 'f' score of the cell, replace it with the updated value 
                g_score[child_cell]= temp_g_score
                f_score[child_cell]= temp_f_score
                pq.put((temp_f_score,heuristic_evaluation(child_cell,(row_size-1,col_size-1)),child_cell))
                count_astar=count_astar+1
                Path[child_cell]=current_cell
    ans=[]
    if(flag==0):
        return ans        
    fwdPath={}
    cell=(row_size-1,col_size-1)
    ans.append((row_size-1,col_size-1))
    while cell!=source:                                                                                 # Traversing the path using the parent to reconstruct the path
        ans.append(Path[cell])
        cell=Path[cell]
    return ans[::-1]

# GHOST MOVEMENT        
def ghost(ghost_start_row, ghost_start_column, temp_matrix):
    final_pos = []
    #print(ghost_start_row - 1, ghost_start_row + 1, ghost_start_column - 1, ghost_start_column + 1)
    flag=0
    if((ghost_start_row - 1 < 0 and ghost_start_column - 1 < 0) or (ghost_start_row - 1 < 0 and ghost_start_column + 1 > col_size-1)    # For Corner Cases
        or (ghost_start_row + 1 > row_size-1 and ghost_start_column - 1 <0) or (ghost_start_row + 1 > row_size-1 and ghost_start_column + 1 > col_size-1)):
        if(ghost_start_row - 1 < 0 and ghost_start_column - 1 < 0):
            directions = ["Down","Right"]
            flag=1
        if(ghost_start_row - 1 < 0 and ghost_start_column + 1 > col_size-1):
            directions = ["Down","Left"]
            flag=1 
        if(ghost_start_row + 1 > row_size-1 and ghost_start_column - 1 <0):
            directions = ["Up","Right"]
            flag=1 
        if(ghost_start_row + 1 > row_size-1 and ghost_start_column + 1 > col_size-1):
            directions = ["Up","Left"]
            flag=1  
    else:                  
        if(ghost_start_row - 1 < 0):                                                        # For first row
            directions = ["Down","Left","Right"]
            flag=1
        if(ghost_start_row + 1 > row_size-1):                                               # For last row
            directions = ["Up","Left","Right"]
            flag=1
        if(ghost_start_column - 1 < 0):                                                     # For first column
            directions = ["Down","Up","Right"]
            flag=1
        if(ghost_start_column + 1 > col_size-1):                                            # For last column
            directions = ["Down","Left","Up"]
            flag=1
    if(flag==0):
        directions = ["Up","Down","Left","Right"]   
    direction_choice = random.choice(directions)
    blocked = ["Move","Stay"]                                                               # In the case the ghost is in a blocked cell, a choice from this list is used to determine if the ghost moves or stays.
    Movement_choice = "Move"
    if(temp_matrix[ghost_start_row][ghost_start_column]==0):                                # Checking if the current ghost position is in a blocked cell
        Movement_choice = random.choice(blocked)
    if(Movement_choice == "Stay"):
        final_pos.append(ghost_start_row)
        final_pos.append(ghost_start_column)
    else:    
        # UP
        if(direction_choice == "Up"):                                                       
            if(temp_matrix[ghost_start_row - 1][ghost_start_column] == 1):
                final_pos.append(ghost_start_row - 1)
                final_pos.append(ghost_start_column)
            elif(temp_matrix[ghost_start_row - 1][ghost_start_column] == 0):
                blocked_choice = random.choice(blocked)
                if(blocked_choice == "Move"):
                    final_pos.append(ghost_start_row - 1)
                    final_pos.append(ghost_start_column)
                elif(blocked_choice == "Stay"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column)
            elif(temp_matrix[ghost_start_row - 1][ghost_start_column] == 6):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column)         
        # DOWN
        if(direction_choice == "Down"):
            if(temp_matrix[ghost_start_row + 1][ghost_start_column] == 1):
                final_pos.append(ghost_start_row + 1)
                final_pos.append(ghost_start_column)
            elif(temp_matrix[ghost_start_row + 1][ghost_start_column] == 0):
                blocked_choice = random.choice(blocked)
                if(blocked_choice == "Move"):
                    final_pos.append(ghost_start_row + 1)
                    final_pos.append(ghost_start_column)
                elif(blocked_choice == "Stay"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column)
            elif(temp_matrix[ghost_start_row + 1][ghost_start_column] == 6):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column)        
        # LEFT
        if(direction_choice == "Left"):
            if(temp_matrix[ghost_start_row][ghost_start_column - 1] == 1):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column - 1)
            elif(temp_matrix[ghost_start_row][ghost_start_column - 1] == 0):
                blocked_choice = random.choice(blocked)
                if(blocked_choice == "Move"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column - 1)
                elif(blocked_choice == "Stay"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column)
            elif(temp_matrix[ghost_start_row][ghost_start_column - 1] == 6):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column)        
        # RIGHT
        if(direction_choice == "Right"):
            if(temp_matrix[ghost_start_row][ghost_start_column + 1] == 1):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column + 1)
            elif(temp_matrix[ghost_start_row][ghost_start_column + 1] == 0):
                blocked_choice = random.choice(blocked)
                if(blocked_choice == "Move"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column + 1)
                elif(blocked_choice == "Stay"):
                    final_pos.append(ghost_start_row)
                    final_pos.append(ghost_start_column) 
            elif(temp_matrix[ghost_start_row][ghost_start_column + 1] == 6):
                final_pos.append(ghost_start_row)
                final_pos.append(ghost_start_column)         
    return final_pos                                  

def Agent1(ghost_position_list):
    agent_1_death_count=0                                                                   # Initializing the number of deaths
    for i in range(0,len(Maze_Set)):
        print("Agent 1 ",i)
        flag=0
        temp_matrix = Maze_Set[i]
        temp_path=Path_Set[i]                                                               # Accessing the BFS generated path for maze number i
        temp_path = temp_path[::-1]
        temp_path.pop(0)
        for i in temp_path:
            if([i[0],i[1]] in ghost_position_list):                                         # Checkng if the ghost and Agent1 are in the same position which results in the death of Agent 1.
                flag=1
                agent_1_death_count=agent_1_death_count+1                                   # Incrementing the death counter
                break

            for i in range(len(ghost_position_list)):                                       # Iterating through every ghost position 
                ghost_current=ghost_position_list[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)     # Finding the new ghost position after it has moved
                ghost_position_list[i]=[ghost_new_position[0],ghost_new_position[1]]

    print("agent_1_death_count ",agent_1_death_count)     
    return agent_1_death_count    
       
def Agent2(ghost_position_list_2):
    agent_2_death_count=0
    for i in range(0,len(Maze_Set)):                                                          # Moving over all the valid mazes 
        flag=0
        temp_matrix = Maze_Set[i]    
        print("Agent 2 ",i)
        current_row=0 
        current_column=0 
        new_path=[]
        ghost_position=[]
        for i in ghost_position_list_2:
            ghost_position.append(i)
        while(TRUE):
            if([current_row,current_column] in ghost_position):                               # Checking if the agent and ghost are in the same position
                flag=1
                agent_2_death_count=agent_2_death_count+1
                break
                
            if(flag==1):
                break 

            if(current_row==row_size-1 and current_column==col_size-1):                       # The Agent has reached the destination cell.
                break
            
            adjacent_cells=[[current_row,current_column-1],[current_row,current_column+1],[current_row-1,current_column],[current_row+1,current_column],
                            [current_row+1,current_column-1],[current_row+1,current_column+1],[current_row-1,current_column-1],[current_row-1,current_column+1]]
                            
                           
            if(len(new_path)==0 or common_member(adjacent_cells,ghost_position)==True):         # Recalculating the path if the ghosts are present in the adjacent nodes.
                new_path=astar_original(temp_matrix,(current_row,current_column),ghost_position)           

            if(len(new_path)==0):                                                               # As there is no viable path from the current cell to the destination, the agent moves away from the nearest ghost
                min=100000
                for i in range(len(ghost_position)):                                            # Finding closest ghost
                    dist=abs(ghost_position[i][1]-current_column)+(abs(ghost_position[i][0]-current_row))
                    if(dist<min):
                        min=dist
                        closest_ghost=ghost_position[i]
                        max=0
                t_current_row=current_row
                t_current_column=current_column                                                # Moving the ghost away from the closest ghost
                if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position):
                    d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                    if(d>max):
                        t_current_row=current_row
                        t_current_column=current_column-1
                        max=d   
                if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                    d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                    if(d>max):
                        t_current_row=current_row
                        t_current_column=current_column+1
                        max=d 
                if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                    d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                    if(d>max):
                        t_current_row=current_row-1
                        t_current_column=current_column
                        max=d  
                if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                    d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                    if(d>max):
                        t_current_row=current_row+1
                        t_current_column=current_column
                        max=d 
                current_row=t_current_row
                current_column=t_current_column        
            else:
                current_row=new_path[1][0]
                current_column=new_path[1][1]
                new_path.pop(0)

            
            for i in range(len(ghost_position)):                                                            # GHOST MOVEMENT
                ghost_current=ghost_position[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
                ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]]

            if(flag==1):
                break     
         
    print("agent_2_death_count ",agent_2_death_count)             
    return agent_2_death_count 

def Agent2_implementation(temp_matrix,s,temp_ghost_position):
    agent_2_death_count=0
    flag=0   
    current_row=s[0]
    current_column=s[1]
    new_path=[]
    ghost_position=[]
    for i in temp_ghost_position:
        ghost_position.append(i)
    while(TRUE):
        if([current_row,current_column] in ghost_position):
            flag=1
            agent_2_death_count=agent_2_death_count+1
            break
        
        if(flag==1):
            break 

        if(current_row==row_size-1 and current_column==col_size-1):
                break
        
        adjacent_cells=[[current_row,current_column-1],[current_row,current_column+1],[current_row-1,current_column],[current_row+1,current_column]] #13th oct
            
        #if(len(new_path)==0 or common_member(new_path,ghost_position)==True): 13th oct
        if(len(new_path)==0 or common_member(adjacent_cells,ghost_position)==True):        #13th Oct 
            new_path=astar_original(temp_matrix,(current_row,current_column),ghost_position)                               
        
        if(len(new_path)==0):                                                                           # Checking nearest ghost and moving away
            min=100000
            for i in range(len(ghost_position)):
                dist=abs(ghost_position[i][1]-current_column)+(abs(ghost_position[i][0]-current_row))
                if(dist<min):
                    min=dist
                    closest_ghost=ghost_position[i]
                    max=0
            t_current_row=current_row
            t_current_column=current_column
            if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position):
                d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                if(d>max):
                    t_current_row=current_row
                    t_current_column=current_column-1
                    max=d   
            if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                if(d>max):
                    t_current_row=current_row
                    t_current_column=current_column+1
                    max=d 
            if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                if(d>max):
                    t_current_row=current_row-1
                    t_current_column=current_column
                    max=d  
            if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                if(d>max):
                    t_current_row=current_row+1
                    t_current_column=current_column
                    max=d 
            current_row=t_current_row
            current_column=t_current_column        
        else:
            current_row=new_path[1][0]
            current_column=new_path[1][1]

        for i in range(len(ghost_position)):                                                            # Ghost Movement
            ghost_current=ghost_position[i]
            ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
            ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]]

        if(flag==1):
            break  

    if(flag==0):
        return 1
    else:
        return []   
        
def Agent3(ghost_position_list_3):
    agent_3_death_count=0                                                                   # Initializing the number of deaths.
    for i in range(0,len(Maze_Set)):
        temp=i
        flag=0
        temp_matrix = Maze_Set[i]    
        print("Agent 3 ",i)
        current_row=0      
        current_column=0   
        ghost_position=[]
        for i in ghost_position_list_3:
            ghost_position.append(i) 
        new_path=[]
        while(TRUE):
            print("current_row ",current_row,"current_column ",current_column)
            if([current_row,current_column] in ghost_position):                             # Checking if the agent and the ghost are in the same position.
                flag=1
                agent_3_death_count=agent_3_death_count+1
                break  
            
            if(flag==1):
                break 

            if(current_row==row_size-1 and current_column==col_size-1):                     # Destination cell has been reached
                break
            
            if(len(new_path)==0 or common_member(new_path,ghost_position)==True):           # Running the Simulation if there is no path or a ghost has wandered into the path
                count_stay=0
                count_left=0
                count_right=0
                count_up=0
                count_down=0
                val=0
                for i in range(5):                                                                                  # In place simulation of the paths.
                    val=Agent2_implementation(temp_matrix,(current_row,current_column),ghost_position)
                    if(val==1):
                        count_stay=count_stay+1
                if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position): 
                    count_left=0
                    val=0
                    for i in range(5):                                                                               # Simulation of the paths from the left adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row,current_column-1),ghost_position)
                        if(val==1):
                            count_left=count_left+1
                if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                    count_right=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the right adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row,current_column+1),ghost_position)
                        if(val==1):
                            count_right=count_right+1
                if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                    count_up=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the top adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row-1,current_column),ghost_position)
                        if(val==1):
                            count_up=count_up+1
                if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                    count_down=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the bottom adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row+1,current_column),ghost_position)
                        if(val==1):
                            count_down=count_down+1            
                if(count_stay ==0 and count_left==0 and count_right==0 and count_up==0 and count_down==0):          # In case survivability is zero from all cells, move away from the nearest ghost
                    min=100000
                    for i in range(len(ghost_position)):
                        dist=abs(ghost_position[i][1]-current_column)+(abs(ghost_position[i][0]-current_row))
                        if(dist<min):
                            min=dist
                            closest_ghost=ghost_position[i]
                            max=0
                    t_current_row=current_row
                    t_current_column=current_column
                    if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column-1
                            max=d   
                    if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column+1
                            max=d 
                    if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                        if(d>max):
                            t_current_row=current_row-1
                            t_current_column=current_column
                            max=d  
                    if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                        if(d>max):
                            t_current_row=current_row+1
                            t_current_column=current_column
                            max=d 
                    current_row=t_current_row
                    current_column=t_current_column        
                else:
                    l=[count_stay,count_left,count_right,count_up,count_down]          
                    max_val = sorted(l)[-1]
                    if(count_down == max_val):                                                                      # Moving to the cell with the highest rate of survivability.  
                        current_row=current_row+1
                        current_column=current_column
                    elif(count_right == max_val):
                        current_row=current_row
                        current_column=current_column+1
                    elif(count_stay == max_val):
                        current_row=current_row
                        current_column=current_column   
                    elif(count_up == max_val):
                        current_row=current_row-1
                        current_column=current_column 
                    elif(count_left == max_val):
                        current_row=current_row
                        current_column=current_column-1
            
            if(len(new_path)==0 or common_member(new_path,ghost_position)==True):                               
                new_path=astar_original(temp_matrix, (current_row,current_column) , ghost_position)                 # Calculating the new path
            else:                                                                                     
                new_path.pop(0)
                current_row=new_path[0][0]                                # Moving to the next step in the path if a ghost has not moved into any cell in the path
                current_column=new_path[0][1]    

            for i in range(len(ghost_position)):                                                            # GHOST MOVEMENT
                ghost_current=ghost_position[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
                ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]] 
            if(flag==1):
                break
         
    print("agent_3_death_count ",agent_3_death_count)             
    return agent_3_death_count

def Agent4(ghost_position_list_4):
    agent_4_death_count=0                                                                                   # Initializing the number of deaths.
    for i in range(0,len(Maze_Set)):
        flag=0
        temp_matrix = Maze_Set[i]    
        print("Agent 4 ",i)
        current_row=0
        current_column=0
        ghost_position=[]
        for i in ghost_position_list_4:
            ghost_position.append(i)
        new_path=[]    
        while(TRUE):
            if([current_row,current_column] in ghost_position):                                             # Checking if the agent and the ghost are in the same position.
                flag=1
                agent_4_death_count=agent_4_death_count+1
                break
            
            if(flag==1):
                break 

            if(current_row==row_size-1 and current_column==col_size-1):
                break
            # Adjacent cells
            adjacent_cells=[[current_row,current_column-1],[current_row,current_column+1],[current_row-1,current_column],[current_row+1,current_column],
                            [current_row+1,current_column-1],[current_row+1,current_column+1],[current_row-1,current_column-1],[current_row-1,current_column+1]]

            print("current_row,current_column ",current_row,current_column)
            #Running the Simulation if there is no path or a ghost has wandered into the path or if any adjacent cell has a ghost
            if(len(new_path)==0 or common_member(adjacent_cells,ghost_position)==True or common_member(new_path,ghost_position)==True):  
                count_left=0
                count_right=0
                count_up=0
                count_down=0
                if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position):
                    count_left=0
                    val=0
                    for i in range(5):
                        val=Agent2_implementation(temp_matrix,(current_row,current_column-1),ghost_position)    # Simulation of the paths from the left adjacent cell
                        if(val==1):
                            count_left=count_left+1
                if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                    count_right=0
                    val=0
                    for i in range(5):
                        val=Agent2_implementation(temp_matrix,(current_row,current_column+1),ghost_position)    # Simulation of the paths from the right adjacent cell
                        if(val==1):
                            count_right=count_right+1
                if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                    count_up=0
                    val=0
                    for i in range(5):
                        val=Agent2_implementation(temp_matrix,(current_row-1,current_column),ghost_position)    # Simulation of the paths from the above adjacent cell
                        if(val==1):
                            count_up=count_up+1
                if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                    count_down=0
                    val=0
                    for i in range(5):
                        val=Agent2_implementation(temp_matrix,(current_row+1,current_column),ghost_position)   # Simulation of the paths from the below adjacent cell
                        if(val==1):
                            count_down=count_down+1            
                if(count_left==0 and count_right==0 and count_up==0 and count_down==0):                        # In case survivability is zero from all cells, move away from the nearest ghost
                    min=100000
                    for i in range(len(ghost_position)):
                        dist=abs(ghost_position[i][1]-current_column)+(abs(ghost_position[i][0]-current_row))
                        if(dist<min):
                            min=dist
                            closest_ghost=ghost_position[i]
                            max=0
                    t_current_row=current_row
                    t_current_column=current_column
                    if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column-1
                            max=d   
                    if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column+1
                            max=d 
                    if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                        if(d>max):
                            t_current_row=current_row-1
                            t_current_column=current_column
                            max=d  
                    if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghost_position):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                        if(d>max):
                            t_current_row=current_row+1
                            t_current_column=current_column
                            max=d 
                    current_row=t_current_row
                    current_column=t_current_column        
                    #print("Next point ",current_row,current_column)   
                else:
                    #max_count=max(count_left,count_right,count_up,count_down)
                    if(count_left>=count_right and count_left>=count_up and count_left>=count_down):
                        max_count=count_left
                    elif(count_right>=count_left and count_right>=count_up and count_right>=count_down):
                        max_count=count_right
                    elif(count_up>=count_right and count_up>=count_left and count_up>=count_down):
                        max_count=count_up
                    elif(count_down>=count_right and count_down>=count_up and count_down>=count_left):
                        max_count=count_down    

                    if(count_down == max_count):
                        current_row=current_row+1
                        current_column=current_column
                    elif(count_right == max_count):
                        current_row=current_row
                        current_column=current_column+1
                    elif(count_up == max_count):
                        current_row=current_row-1
                        current_column=current_column 
                    elif(count_left == max_count):
                        current_row=current_row
                        current_column=current_column-1
            if(len(new_path)==0 or common_member(adjacent_cells,ghost_position)==True or common_member(new_path,ghost_position)==True):   
                new_path=astar_original(temp_matrix, (current_row,current_column) , ghost_position)   # Calculating the new path
            else:
                new_path.pop(0)
                current_row=new_path[0][0]                      # Moving to the next step in the path if a ghost is not in any cell in the path or any adjacent cell
                current_column=new_path[0][1]

            for i in range(len(ghost_position)):                                                                        # GHOST MOVEMENT
                ghost_current=ghost_position[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
                ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]] 

            if(flag==1):
                break 
         
    print("agent_4_death_count ",agent_4_death_count)             
    return agent_4_death_count

def Agent3_Low_Info(ghost_position_list_3):
    agent_3_LI_death_count=0                                                                   # Initializing the number of deaths.
    for i in range(0,len(Maze_Set)):
        temp=i
        flag=0
        temp_matrix = Maze_Set[i]    
        print("Agent 3 ",i)
        current_row=0      
        current_column=0   
        ghost_position=[]
        for i in ghost_position_list_3:
            ghost_position.append(i) 
        new_path=[]
        while(TRUE):
            print("current_row ",current_row,"current_column ",current_column)
            if([current_row,current_column] in ghost_position):                             # Checking if the agent and the ghost are in the same position.
                flag=1
                agent_3_LI_death_count=agent_3_LI_death_count+1
                break  
            
            if(flag==1):
                break 

            if(current_row==row_size-1 and current_column==col_size-1):                     # Destination cell has been reached
                break

            ghosts_not_in_walls=[]                                                          # Making sure only the ghosts in the unblocked cells are sent for simulation
            for i in ghost_position:
                if(temp_matrix[i[0]][i[1]]==1):
                    ghosts_not_in_walls.append([i[0],i[1]])
            
            if(len(new_path)==0 or common_member(new_path,ghosts_not_in_walls)==True):           # Running the Simulation if there is no path or a ghost has wandered into the path
                count_stay=0
                count_left=0
                count_right=0
                count_up=0
                count_down=0
                val=0
                for i in range(5):                                                                                  # In place simulation of the paths.
                    val=Agent2_implementation(temp_matrix,(current_row,current_column),ghosts_not_in_walls)
                    if(val==1):
                        count_stay=count_stay+1
                if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghosts_not_in_walls): 
                    count_left=0
                    val=0
                    for i in range(5):                                                                               # Simulation of the paths from the left adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row,current_column-1),ghosts_not_in_walls)
                        if(val==1):
                            count_left=count_left+1
                if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghosts_not_in_walls):
                    count_right=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the right adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row,current_column+1),ghosts_not_in_walls)
                        if(val==1):
                            count_right=count_right+1
                if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghosts_not_in_walls):
                    count_up=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the top adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row-1,current_column),ghosts_not_in_walls)
                        if(val==1):
                            count_up=count_up+1
                if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghosts_not_in_walls):
                    count_down=0
                    val=0
                    for i in range(5):                                                                              # Simulation of the paths from the bottom adjacent cell
                        val=Agent2_implementation(temp_matrix,(current_row+1,current_column),ghosts_not_in_walls)
                        if(val==1):
                            count_down=count_down+1            
                if(count_stay ==0 and count_left==0 and count_right==0 and count_up==0 and count_down==0):          # In case survivability is zero from all cells, move away from the nearest ghost
                    min=100000
                    for i in range(len(ghosts_not_in_walls)):
                        dist=abs(ghosts_not_in_walls[i][1]-current_column)+(abs(ghosts_not_in_walls[i][0]-current_row))
                        if(dist<min):
                            min=dist
                            closest_ghost=ghosts_not_in_walls[i]
                            max=0
                    t_current_row=current_row
                    t_current_column=current_column
                    if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghosts_not_in_walls):
                        d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column-1
                            max=d   
                    if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghosts_not_in_walls):
                        d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                        if(d>max):
                            t_current_row=current_row
                            t_current_column=current_column+1
                            max=d 
                    if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghosts_not_in_walls):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                        if(d>max):
                            t_current_row=current_row-1
                            t_current_column=current_column
                            max=d  
                    if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghosts_not_in_walls):
                        d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                        if(d>max):
                            t_current_row=current_row+1
                            t_current_column=current_column
                            max=d 
                    current_row=t_current_row
                    current_column=t_current_column        
                else:
                    l=[count_stay,count_left,count_right,count_up,count_down]          
                    max_val = sorted(l)[-1]
                    if(count_down == max_val):                                                                      # Moving to the cell with the highest rate of survivability.  
                        current_row=current_row+1
                        current_column=current_column
                    elif(count_right == max_val):
                        current_row=current_row
                        current_column=current_column+1
                    elif(count_stay == max_val):
                        current_row=current_row
                        current_column=current_column   
                    elif(count_up == max_val):
                        current_row=current_row-1
                        current_column=current_column 
                    elif(count_left == max_val):
                        current_row=current_row
                        current_column=current_column-1
            
            if(len(new_path)==0 or common_member(new_path,ghost_position)==True):                               
                new_path=astar_original(temp_matrix, (current_row,current_column) , ghost_position)                 # Calculating the new path
            else:                                                                                     
                new_path.pop(0)
                current_row=new_path[0][0]                                # Moving to the next step in the path if a ghost has not moved into any cell in the path
                current_column=new_path[0][1]    

            for i in range(len(ghost_position)):                                                            # GHOST MOVEMENT
                ghost_current=ghost_position[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
                ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]] 
            if(flag==1):
                break
         
    print("agent_3_LI_death_count ",agent_3_LI_death_count)             
    return agent_3_LI_death_count

def Agent2_Low_Info(ghost_position_list_2):
    agent_2_LI_death_count=0
    for i in range(0,len(Maze_Set)):                                                          # Moving over all the valid mazes 
        flag=0
        temp_matrix = Maze_Set[i]    
        print("Agent 2 ",i)
        current_row=0 
        current_column=0 
        new_path=[]
        ghost_position=[]
        for i in ghost_position_list_2:
            ghost_position.append(i)
        while(TRUE):
            if([current_row,current_column] in ghost_position):                               # Checking if the agent and ghost are in the same position
                flag=1
                agent_2_LI_death_count=agent_2_LI_death_count+1
                break
                
            if(flag==1):
                break 

            if(current_row==row_size-1 and current_column==col_size-1):                       # The Agent has reached the destination cell.
                break
            
            adjacent_cells=[[current_row,current_column-1],[current_row,current_column+1],[current_row-1,current_column],[current_row+1,current_column],
                            [current_row+1,current_column-1],[current_row+1,current_column+1],[current_row-1,current_column-1],[current_row-1,current_column+1]]
                            
            ghosts_not_in_walls=[]                                                           # Making sure only the ghosts in the unblocked cells are sent for simulation
            for i in ghost_position:
                if(temp_matrix[i[0]][i[1]]==1):
                    ghosts_not_in_walls.append([i[0],i[1]])

            if(len(new_path)==0 or common_member(adjacent_cells,ghosts_not_in_walls)==True):         # Recalculating the path if the ghosts are present in the adjacent nodes.
                new_path=astar_original(temp_matrix,(current_row,current_column),ghosts_not_in_walls)           

            if(len(new_path)==0):                                                               # As there is no viable path from the current cell to the destination, the agent moves away from the nearest ghost
                min=100000
                for i in range(len(ghosts_not_in_walls)):                                            # Finding closest ghost
                    dist=abs(ghosts_not_in_walls[i][1]-current_column)+(abs(ghosts_not_in_walls[i][0]-current_row))
                    if(dist<min):
                        min=dist
                        closest_ghost=ghosts_not_in_walls[i]
                        max=0
                t_current_row=current_row
                t_current_column=current_column                                                # Moving the ghost away from the closest ghost
                if(current_column - 1 >= 0 and temp_matrix[current_row][current_column - 1] == 1 and [current_row,current_column-1] not in ghosts_not_in_walls):
                    d=abs(closest_ghost[1]-(current_column-1))+(abs(closest_ghost[0]-current_row))
                    if(d>max):
                        t_current_row=current_row
                        t_current_column=current_column-1
                        max=d   
                if(current_column + 1 <= col_size-1 and temp_matrix[current_row][current_column + 1] == 1 and [current_row,current_column+1] not in ghosts_not_in_walls):
                    d=abs(closest_ghost[1]-(current_column+1))+(abs(closest_ghost[0]-current_row))  
                    if(d>max):
                        t_current_row=current_row
                        t_current_column=current_column+1
                        max=d 
                if (current_row - 1 >= 0 and temp_matrix[current_row - 1][current_column] == 1 and [current_row-1,current_column] not in ghosts_not_in_walls):
                    d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row-1))  
                    if(d>max):
                        t_current_row=current_row-1
                        t_current_column=current_column
                        max=d  
                if (current_row + 1 <= row_size-1 and temp_matrix[current_row + 1][current_column] == 1 and [current_row+1,current_column] not in ghosts_not_in_walls):
                    d=abs(closest_ghost[1]-(current_column))+(abs(closest_ghost[0]-current_row+1))  
                    if(d>max):
                        t_current_row=current_row+1
                        t_current_column=current_column
                        max=d 
                current_row=t_current_row
                current_column=t_current_column        
            else:
                current_row=new_path[1][0]
                current_column=new_path[1][1]
                new_path.pop(0)

            
            for i in range(len(ghost_position)):                                                            # GHOST MOVEMENT
                ghost_current=ghost_position[i]
                ghost_new_position=ghost(ghost_current[0],ghost_current[1],temp_matrix)
                ghost_position[i]=[ghost_new_position[0],ghost_new_position[1]]

            if(flag==1):
                break     
         
    print("agent_2_LI_death_count ",agent_2_LI_death_count)             
    return agent_2_LI_death_count 


Maze_Creation()
print("Maze_Set ")
print(Maze_Set)
print("Path_Set ")
print(Path_Set)
index=list(range(0,50))
print("index", index)
ghost_position_list=[]
Path_Set_Agent_2=Path_Set
count=0

number_of_ghost_list=[35,45,55,65,75]
ag1_survival=[]
ag2_survival=[]
ag3_survival=[]
ag4_survival=[]
ag2_li_survival=[]
ag3_li_survival=[]
for i in number_of_ghost_list:
    print("Iteration ",i)
    count=0
    ghost_position_list=[]
    while count<i:
        ghost_start_row=random.choice(index)
        ghost_start_column=random.choice(index)
        if(ghost_start_row==0 and ghost_start_column==0):
            count=count
        else:
            ghost_position_list.append([ghost_start_row,ghost_start_column])  
            count=count+1
    ghost_position_list_2=[] 
    for i in ghost_position_list:
        ghost_position_list_2.append(i)  
    ghost_position_list_3=[]    
    for i in ghost_position_list:
        ghost_position_list_3.append(i)  
    ghost_position_list_4=[]    
    for i in ghost_position_list:
        ghost_position_list_4.append(i)         
        
    ag1=Agent1(ghost_position_list)
    ag1_survivability=(1-(ag1/len(Maze_Set)))*100
    ag1_survival.append(ag1_survivability)

    ag2=Agent2(ghost_position_list_2)
    ag2_survivability=(1-(ag2/len(Maze_Set)))*100
    ag2_survival.append(ag2_survivability)
    
    ag3=Agent3(ghost_position_list_3)
    ag3_survivability=(1-(ag3/len(Maze_Set)))*100
    ag3_survival.append(ag3_survivability)
    
    ag4=Agent4(ghost_position_list_4)
    ag4_survivability=(1-(ag4/len(Maze_Set)))*100
    ag4_survival.append(ag4_survivability)
    
    ag2_li=Agent2_Low_Info(ghost_position_list_2)
    ag2_li_survivability=(1-(ag2_li/len(Maze_Set)))*100
    ag2_li_survival.append(ag2_li_survivability)
    
    ag3_li=Agent3_Low_Info(ghost_position_list_3)
    ag3_li_survivability=(1-(ag3_li/len(Maze_Set)))*100
    ag3_li_survival.append(ag3_li_survivability)
    
    
print("ag1_survivability ",ag1_survival)
print("ag2_survivability ",ag2_survival)
print("ag3_survivability ",ag3_survival)
print("ag4_survivability ",ag4_survival)
print("ag2_li_survival ",ag2_li_survival)
print("ag3_li_survival ",ag3_li_survival)


plt.plot(number_of_ghost_list, ag1_survival, label="Agent 1")
plt.plot(number_of_ghost_list, ag2_survival, label="Agent 2")
plt.plot(number_of_ghost_list, ag3_survival, label="Agent 3")
plt.plot(number_of_ghost_list, ag4_survival, label="Agent 4")
plt.title("Survivability Graph")
plt.xlabel("Number of Ghosts")
plt.ylabel("Survivability")
leg=plt.legend()
plt.show()
print("---------------------END-----------------------------")


