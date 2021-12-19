"""
@Title: Best First Search

@author: Boris Kundu

@Usage: python best_first_search.py

## Function to create Romaina Map
Takes as input the file containing city (node) and distance (cost).
"""

import time

def Create_Map(inputFileName):
  Romania_map = {}

  with open(inputFileName, 'r') as file:
    lines = file.readlines()
    for line in lines:
      node1, node2, cost = line.split(',')
      node1 = node1.strip()
      node2 = node2.strip()
      cost = int(cost.strip())
      
      if (node1 not in Romania_map):
        Romania_map[node1] = []
      if (node2 not in Romania_map):
        Romania_map[node2] = []
      
      Romania_map[node1].append((node2, cost))
      Romania_map[node2].append((node1, cost))

    return Romania_map

map = Create_Map('WeightedMap.txt')

"""## Create Heuristic adj list

Here we use triangle inequality to calculate the Heuristic (straight line distance) between each pair of nodes for which a direct connection (edge) doesn't exist. 
>$$a <= b + c$$

"""

import copy 
Heuristic_Map = copy.deepcopy(map)

for node in Heuristic_Map.keys():
  connected_cities = len(Heuristic_Map[node])
  for i in range(connected_cities):
    for j in range(i+1, connected_cities):

      v1 = Heuristic_Map[node][i][0]
      v2 = Heuristic_Map[node][j][0]

      l1 = [city[0] for city in Heuristic_Map[v1]]
      
      if (v2 not in l1):

        v1_cost = Heuristic_Map[node][i][1]
        v2_cost = Heuristic_Map[node][j][1]

        v1_v2_cost = int((v1_cost + v2_cost) * 0.9)

        if (v1 not in Heuristic_Map):
          Heuristic_Map[v1] = []
        if (v2 not in Heuristic_Map):
          Heuristic_Map[v2] = []

        Heuristic_Map[v1].append((v2, v1_v2_cost))
        Heuristic_Map[v2].append((v1, v1_v2_cost))


inputFileName = 'SLD_Bucarest.txt'
# Read Existing SLD from textbook for Bucarest
with open(inputFileName, 'r') as file:
  lines = file.readlines()
  for line in lines:
    node2, cost = line.split(',')
    node1 = 'Bucharest'
    node2 = node2.strip()
    cost = int(cost.strip())

    Heuristic_Map[node1] = [(k,cost) if ( k == node2) else (k, v) for (k,v) in Heuristic_Map[node1]]
    Heuristic_Map[node2] = [(k,cost) if ( k == 'Bucharest') else (k, v) for (k,v) in Heuristic_Map[node2]]

"""## Implementing a Priority Queue"""

class PriorityQueue():

  registeredElements = []

  def __init__(self):
    self.nodes = []
    
  def __str__(self):
    return [str(i) for i in self.nodes]

  def isEmpty(self):
    return len(self.nodes) == 0

  def addNode(self, node, cost):
    self.nodes.append((node, cost))      
    self.nodes.sort(key = lambda x: x[1], reverse=True) #Sorting after intertion to maintain priority based on cost
    self.registeredElements = [i for i in self.nodes] #Updating registered elements in traversed path

  def popNode(self):
    a,b = self.nodes.pop()
    return a

  def getTotalCost(self):
    a,b = self.nodes.pop()
    return b

# Helper function to retrieve costs of given node pair
def getCost(CostMap, node1, node2):
  for (city, cost) in CostMap[node1]:
    if city.strip().lower() == node2.strip().lower():
        return cost

"""## Implementing Best First search algorithm"""

def Best_First_Search(graph, start, end):
  BestFS= PriorityQueue()

  cumulative_cost = 0
  traverser = []

  BestFS.addNode([(start, 0)], 0)
  explored = set()

  while not BestFS.isEmpty():

    # display run status of A-Star
    for a,b in BestFS.registeredElements:
      traverser = []
      total_cost = b
      for x,y in a:
        traverser.append(x)
      pg = ' --> '.join(traverser)
      print('Path Traversed: {}'.format(pg))
    print()

    # Pop the path with least F(x) cost  (best path) from the Priority Queue
    current_path = BestFS.popNode()
        
    # Initializing starting node
    node = current_path[-1][0]
    cost = current_path[-1][1]
    explored.add(node)

    # Check if current node is the destination:
    if node == end:
        ans = [x for x, y in current_path]
        ans_txt = ' --> '.join(ans)
        total_cost = BestFS.registeredElements[-1][1]
        print('Goal reached!!')
        return True
    
    #Else add new node to the queue
    print("-"*85)
    print("|{:<20}|{:<30}|{:<30}|".format("City Name",f"Distance From City: {node}",f"SLD from City: {end}"))
    print("-"*85)
    chosen = ('zzzzzzz',99999999999999)
    for neighbor, distance in graph[node]:
      if neighbor == end:
        cost = 0
      else:
        cost = getCost(Heuristic_Map, neighbor, end)
      print("|{:<20}|{:<30n}|{:<30n}|".format(neighbor,distance,cost))
      if (cost < chosen[1] or (cost == chosen[1] and neighbor < chosen[0])):
        chosen = (neighbor, cost)

    print("-"*85)        
    new_path = current_path + [chosen]
    BestFS.addNode(new_path,cost + chosen[1])
  return False

# Note timings
start_time = time.time()

#for i in range (0,200):

Best_First_Search(map, 'Arad', 'Neamt')

print("Performance in seconds of Best First algorithm's single execution is => %s " % (time.time() - start_time))

""" *** BEST FIRST OUTPUT *** 

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Arad      |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Zerind              |75                            |589                           |
|Sibiu               |140                           |494                           |
|Timisoara           |118                           |611                           |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Sibiu     |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Arad                |140                           |562                           |
|Oradea              |151                           |567                           |
|Fagaras             |99                            |450                           |
|Rimnicu Vilcea      |80                            |531                           |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Fagaras   |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Sibiu               |99                            |494                           |
|Bucharest           |211                           |234                           |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras --> Bucharest

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Bucharest |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Fagaras             |211                           |450                           |
|Pitesti             |101                           |378                           |
|Urziceni            |85                            |267                           |
|Giurgiu             |90                            |369                           |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras --> Bucharest --> Urziceni

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Urziceni  |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Bucharest           |85                            |234                           |
|Hirsova             |98                            |327                           |
|Vaslui              |142                           |161                           |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras --> Bucharest --> Urziceni --> Vaslui

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Vaslui    |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Urziceni            |142                           |267                           |
|Lasi                |92                            |87                            |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras --> Bucharest --> Urziceni --> Vaslui --> Lasi

-------------------------------------------------------------------------------------
|City Name           |Distance From City: Lasi      |SLD from City: Neamt          |
-------------------------------------------------------------------------------------
|Vaslui              |92                            |161                           |
|Neamt               |87                            |0                             |
-------------------------------------------------------------------------------------
Path Traversed: Arad --> Sibiu --> Fagaras --> Bucharest --> Urziceni --> Vaslui --> Lasi --> Neamt

Goal reached!!
Performance in seconds of Best First algorithm's single execution is => 0.08149528503417969

""" 