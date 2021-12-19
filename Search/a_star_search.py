"""
    #A* search algorithm for Romaina map problem
    Author: Boris Kundu

    !!File dependency = 
    place the 2 txt files [WeightedMap.txt; SLD_Bucharest.txt] in the 
    same directory as this python file before executing

    # @Usage: python a_star_search.py

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

Here we use triangle inequality to calculate the Heuristic (straight line distance) 
between each pair of nodes for which a direct connection (edge) doesn't exist. 
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
    if city.lower() == node2.lower():
      return cost

"""## Implementing A Star search algorithm"""

def Search_A_Star(graph, start, end):

  Traversing_AStar = PriorityQueue()
  cumulative_cost = 0
  h_cost = 0
  g_cost = 0
  f_cost = 0
  traverser = []

  # path is a list of tuples of the form ([(node, cost)], 'path cost')
  Traversing_AStar.addNode([(start, 0)],0)
  explored = set()

  while not Traversing_AStar.isEmpty():

    # display run status of A-Star
    for a,b in Traversing_AStar.registeredElements:
      traverser = []
      total_cost = b
      for x,y in a:
        traverser.append(x)
      pg = ' --> '.join(traverser)
      print('Total cost = {} || {}'.format(total_cost, pg))
    print()

    # Pop the path with least F(x) cost  (best path) from the Priority Queue
    current_path = Traversing_AStar.popNode()
        
    # Initializing starting node
    node = current_path[-1][0]
    g_cost = current_path[-1][1]

    explored.add(node)

    # Check if current node is the destination:
    if node == end:
        ans = [x for x, y in current_path]
        ans_txt = ' --> '.join(ans)
        total_cost = Traversing_AStar.registeredElements[-1][1]
        print('Goal reached!! Best path:\nTotal Cost: {} || {}'.format(total_cost,ans_txt))
        return True

    # Check current node's neighbor to calculate F(x)
    for neighbor, distance in graph[node]:
        cumulative_cost = g_cost + distance           
        h_cost = getCost(Heuristic_Map, node, end)           
        f_cost = cumulative_cost + h_cost    
        
        #Update new path with currently explored path and its cost     
        new_path = current_path + [(neighbor, cumulative_cost)]
        
        # Avoiding already traversed paths
        if neighbor not in explored:
            Traversing_AStar.addNode(new_path, f_cost)

  return False

# Note timings
start_time = time.time()

#for i in range (0,100):

Search_A_Star(map, 'Arad', 'Neamt')

print("Performance in seconds of A* algorithm's single execution is => %s " % (time.time() - start_time))

""" *** A* OUTPUT *** 

Total cost = 0 || Arad

Total cost = 702 || Arad --> Sibiu
Total cost = 680 || Arad --> Timisoara
Total cost = 637 || Arad --> Zerind

Total cost = 735 || Arad --> Zerind --> Oradea
Total cost = 702 || Arad --> Sibiu
Total cost = 680 || Arad --> Timisoara

Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 735 || Arad --> Zerind --> Oradea
Total cost = 702 || Arad --> Sibiu

Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 785 || Arad --> Sibiu --> Oradea
Total cost = 735 || Arad --> Zerind --> Oradea
Total cost = 733 || Arad --> Sibiu --> Fagaras
Total cost = 714 || Arad --> Sibiu --> Rimnicu Vilcea

Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 848 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti
Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 785 || Arad --> Sibiu --> Oradea
Total cost = 735 || Arad --> Zerind --> Oradea
Total cost = 733 || Arad --> Sibiu --> Fagaras

Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 848 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti
Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 785 || Arad --> Sibiu --> Oradea
Total cost = 735 || Arad --> Zerind --> Oradea

Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 848 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti
Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 785 || Arad --> Sibiu --> Oradea
Total cost = 735 || Arad --> Zerind --> Oradea

Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 848 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti
Total cost = 840 || Arad --> Timisoara --> Lugoj
Total cost = 785 || Arad --> Sibiu --> Oradea
Total cost = 735 || Arad --> Zerind --> Oradea

Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 848 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti

Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 833 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova
Total cost = 796 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest

Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 833 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova
Total cost = 742 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Giurgiu
Total cost = 737 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni

Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 868 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova
Total cost = 833 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova
Total cost = 742 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Giurgiu

Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 868 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova
Total cost = 833 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova
Total cost = 742 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Giurgiu

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova
Total cost = 868 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 1014 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova --> Eforie
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest
Total cost = 897 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 1080 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova --> Dobreta
Total cost = 1014 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova --> Eforie
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 1080 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova --> Dobreta
Total cost = 1014 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova --> Eforie
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 912 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui
Total cost = 900 || Arad --> Sibiu --> Fagaras --> Bucharest

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 1080 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova --> Dobreta
Total cost = 1014 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova --> Eforie
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 898 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui --> Lasi

Total cost = 1169 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Craiova --> Dobreta
Total cost = 1080 || Arad --> Sibiu --> Rimnicu Vilcea --> Craiova --> Dobreta
Total cost = 1014 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Hirsova --> Eforie
Total cost = 951 || Arad --> Timisoara --> Lugoj --> Mehadia
Total cost = 911 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui --> Lasi --> Neamt

Goal reached!! Best path:
Total Cost: 911 || Arad --> Sibiu --> Rimnicu Vilcea --> Pitesti --> Bucharest --> Urziceni --> Vaslui --> Lasi --> Neamt
Performance in seconds of A* algorithm's single execution is => 0.17765450477600098

""" 