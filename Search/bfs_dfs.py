# @Title: BFS and DFS
# 
# @author: Boris Kundu
# 
# @Usage: python bfs_dfs.py

import time

#City as a class in itself
class City:
    #Defines a new city
    def __init__ (self, cityName):
        self.name = cityName
        self.linkedTo = {}
    #Displays current city info
    def __str__ (self):
        return str(self.cityName + ' is linked to '+ str([city.name for city in self.linkedTo]))
    #Adds a neigboring city to current city
    def addNearByCity (self, adjacentCity, distance = 0):
        self.linkedTo[adjacentCity] = distance
    #Displays connected cities to current city
    def getLinks (self):
        return self.linkedTo.keys()
    #Displays name of current city
    def getCityName (self):
        return self.name
    #Displays distance of a  city from current city
    def getDistanceFrom (self, adjacentCity):
        return self.linkedTo[adjacentCity]

#Romania as a class
class Romania:
    #Defines Romania
    def __init__ (self):
        self.listOfCities = {}
        self.numberOfCities = 0
    #Add a new city to Romania
    def addCity (self,cityName):
        self.numberOfCities = self.numberOfCities + 1
        newCity = City(cityName)
        self.listOfCities[cityName] = newCity
        return newCity
    #Get a city
    def getCity (self,cityName):
        if cityName in self.listOfCities:
            return self.listOfCities[cityName]
        else:
            return None
    #Check if city exists in Romania
    def __contains__ (self,cityName):
        return cityName in self.listOfCities
    #Adds connectivity between two cities in Romania
    def addConnectivity(self,fromCity,toCity,distance = 0):
        if fromCity not in self.listOfCities:
            newCity = self.addCity(fromCity)
        if toCity not in self.listOfCities:
            newCity = self.addCity(toCity)
        self.listOfCities[fromCity].addNearByCity(self.listOfCities[toCity],distance)
    #Get all cities in Romania
    def getCities (self):
        return self.listOfCities.keys()
    #Iterate through Romania
    def __iter__(self):
        return iter(self.listOfCities.values())

#Function that prints route from start city to end city in a country using BFS algorithm
def breadth_first_search(country,startCity,endCity):
    #Initialize start city and goal city
    start = startCity
    goal = endCity
    
    #Initialize visited to null and queue to start city
    visited = set()
    queue = []
    queue.append([start])
    
    #If start and goal city are the same then exit.
    if(start == goal):
        print('Start and destination city are the same.')
        return None

    #Keep processing till queue is empty or we have reched our goal.
    while (queue):
        #Add first city i queue to path and last element
        route = queue.pop(0)
        currentCity = route[-1]
        #Check if current city has not been visited yet
        if (currentCity not in visited):
            visited.add(currentCity)
            #If destination reached then display the route and all visited citie
            if currentCity == goal:
                    print(f'*** BFS OUTPUT ***')
                    print(f'Route between {startCity} and {endCity} => {route}')
                    print(f'Visited cities => {visited}')
                    return None
            #Get neighbouring cities
            neighbours = [nbr.getCityName() for nbr in romania.getCity(currentCity).getLinks()]
            #Initlize and append current path to queue
            for neighbour in neighbours:
                newRoute = route
                newRoute.append(neighbour)
                queue.append(newRoute)
    print (f'No path found between {startCity} and {endCity}')
    print (f'Visited Cities => {visited}')
    return None

#Function that prints route from start city to end city in a country using DFS algorithm
def depth_first_search(country,startCity,endCity):
    #Initialize start city and goal city
    start = startCity
    goal = endCity
    
    #Initialize visited to null and queue to start city
    queue = [(start, [start])]
    visited = set()
    
    #If start and goal city are the same then exit.
    if(start == goal):
        print('Start and destination city are the same.')
        return None
    
    #Keep processing till queue is empty or we have reched our goal.
    while (queue):
        #Pop queue
        (currentCity, route) = queue.pop()
        #Check if current city has not been visited yet
        if (currentCity not in visited):
            visited.add(currentCity)
            #If destination reached then display the route and all visited cities
            if (currentCity == goal):
                    print(f'*** DFS OUTPUT ***')
                    print(f'Route between {startCity} and {endCity} => {route}')
                    print(f'Visited cities => {visited}')
                    return None
            #Get neighbouring cities
            neighbours = [nbr.getCityName() for nbr in romania.getCity(currentCity).getLinks()]
            #Initlize and append current path to queue
            for neighbour in neighbours:
                queue.append((neighbour, route + [neighbour]))
    print (f'No path found between {startCity} and {endCity}')
    print (f'Visited Cities => {visited}')
    return None

#Creating map of Romania
romaniaCities = set()
romaniaCities = ['Oradea','Zerind','Arad','Timisoara','Lugoj','Mehadia','Dobreta',
                 'Sibiu','Rimnicu Vilcea','Craiova','Fagaras','Pitesti','Bucharest',
                 'Giurgiu','Urziceni','Hirsova','Eforie','Vaslui','Iasi','Neamt']
#Create Romania
romania  = Romania()
#Add cities to Romania
for city in romaniaCities:
    romania.addCity(city)

#Connect cities with some distance
romania.addConnectivity('Oradea','Zerind',71)
romania.addConnectivity('Oradea','Sibiu',151)
romania.addConnectivity('Zerind','Oradea',71)
romania.addConnectivity('Zerind','Arad',75)
romania.addConnectivity('Arad','Zerind',75)
romania.addConnectivity('Arad','Timisoara',118)
romania.addConnectivity('Arad','Sibiu',140)
romania.addConnectivity('Timisoara','Lugoj',111)
romania.addConnectivity('Timisoara','Arad',118)
romania.addConnectivity('Sibiu','Oradea',151)
romania.addConnectivity('Sibiu','Arad',140)
romania.addConnectivity('Sibiu','Fagaras',99)
romania.addConnectivity('Sibiu','Rimnicu Vilcea',80)
romania.addConnectivity('Lugoj','Timisoara',111)
romania.addConnectivity('Lugoj','Mehadia',70)
romania.addConnectivity('Mehadia','Lugoj',70)
romania.addConnectivity('Mehadia','Dobreta',75)
romania.addConnectivity('Dobreta','Mehadia',75)
romania.addConnectivity('Dobreta','Craiova',120)
romania.addConnectivity('Rimnicu Vilcea','Sibiu',80)
romania.addConnectivity('Rimnicu Vilcea','Craiova',146)
romania.addConnectivity('Rimnicu Vilcea','Pitesti',97)
romania.addConnectivity('Craiova','Dobreta',120)
romania.addConnectivity('Craiova','Rimnicu Vilcea',146)
romania.addConnectivity('Craiova','Pitesti',138)
romania.addConnectivity('Pitesti','Rimnicu Vilcea',97)
romania.addConnectivity('Pitesti','Craiova',138)
romania.addConnectivity('Pitesti','Bucharest',101)
romania.addConnectivity('Fagaras','Sibiu',99)
romania.addConnectivity('Fagaras','Bucharest',211)
romania.addConnectivity('Bucharest','Fagaras',211)
romania.addConnectivity('Bucharest','Pitesti',101)
romania.addConnectivity('Bucharest','Giurgiu',90)
romania.addConnectivity('Bucharest','Urziceni',85)
romania.addConnectivity('Giurgiu','Bucharest',90)
romania.addConnectivity('Urziceni','Bucharest',85)
romania.addConnectivity('Urziceni','Hirsova',98)
romania.addConnectivity('Urziceni','Vaslui',142)
romania.addConnectivity('Hirsova','Urziceni',98)
romania.addConnectivity('Hirsova','Eforie',86)
romania.addConnectivity('Eforie','Hirsova',86)
romania.addConnectivity('Vaslui','Urziceni',142)
romania.addConnectivity('Vaslui','Iasi',92)
romania.addConnectivity('Iasi','Vaslui',92)
romania.addConnectivity('Iasi','Neamt',87)
romania.addConnectivity('Neamt','Iasi',87)

#Display map of Romania
print("*** ROMANIA MAP DISPLAY ***")
for city in romania:
    for nbr in city.getLinks():
        print(f'{city.getCityName()} => {nbr.getCityName()} => {city.getDistanceFrom(nbr)}')

"""

*** ROMANIA MAP DISPLAY ***
Oradea => Zerind => 71
Oradea => Sibiu => 151
Zerind => Oradea => 71
Zerind => Arad => 75
Arad => Zerind => 75
Arad => Timisoara => 118
Arad => Sibiu => 140
Timisoara => Lugoj => 111
Timisoara => Arad => 118
Lugoj => Timisoara => 111
Lugoj => Mehadia => 70
Mehadia => Lugoj => 70
Mehadia => Dobreta => 75
Dobreta => Mehadia => 75
Dobreta => Craiova => 120
Sibiu => Oradea => 151
Sibiu => Arad => 140
Sibiu => Fagaras => 99
Sibiu => Rimnicu Vilcea => 80
Rimnicu Vilcea => Sibiu => 80
Rimnicu Vilcea => Craiova => 146
Rimnicu Vilcea => Pitesti => 97
Craiova => Dobreta => 120
Craiova => Rimnicu Vilcea => 146
Craiova => Pitesti => 138
Fagaras => Sibiu => 99
Fagaras => Bucharest => 211
Pitesti => Rimnicu Vilcea => 97
Pitesti => Craiova => 138
Pitesti => Bucharest => 101
Bucharest => Fagaras => 211
Bucharest => Pitesti => 101
Bucharest => Giurgiu => 90
Bucharest => Urziceni => 85
Giurgiu => Bucharest => 90
Urziceni => Bucharest => 85
Urziceni => Hirsova => 98
Urziceni => Vaslui => 142
Hirsova => Urziceni => 98
Hirsova => Eforie => 86
Eforie => Hirsova => 86
Vaslui => Urziceni => 142
Vaslui => Iasi => 92
Iasi => Vaslui => 92
Iasi => Neamt => 87
Neamt => Iasi => 87

"""



# Note timings for BFS
start_time = time.time()

#for i in range (0,100):

#Call BFS to find route between Arad and Neamt
breadth_first_search(romania,'Arad','Neamt')

print("Performance in seconds of Breadth First Search algorithm's single execution is => %s " % (time.time() - start_time))

"""

*** BFS OUTPUT ***
Route between Arad and Neamt => ['Arad', 'Zerind', 'Timisoara', 'Sibiu', 'Oradea', 'Arad', 'Fagaras', 'Rimnicu Vilcea', 'Sibiu', 'Craiova', 'Pitesti', 'Rimnicu Vilcea', 'Craiova', 'Bucharest', 'Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni', 'Bucharest', 'Hirsova', 'Vaslui', 'Urziceni', 'Iasi', 'Vaslui', 'Neamt']
Visited cities => {'Arad', 'Sibiu', 'Rimnicu Vilcea', 'Bucharest', 'Iasi', 'Urziceni', 'Neamt', 'Pitesti', 'Vaslui'}
Performance in seconds of Breadth First Search algorithm's single execution is => 0.0058765411376953125
"""

# Note timings for DFS
start_time = time.time()

#for i in range (0,100):

#Call DFS to find route between Arad and Neamt
depth_first_search(romania,'Arad','Neamt')

print("Performance in seconds of Depth First Search algorithm's single execution is => %s " % (time.time() - start_time))

"""

*** DFS OUTPUT ***
Route between Arad and Neamt => ['Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest', 'Urziceni', 'Vaslui', 'Iasi', 'Neamt']
Visited cities => {'Arad', 'Sibiu', 'Rimnicu Vilcea', 'Bucharest', 'Iasi', 'Urziceni', 'Neamt', 'Pitesti', 'Vaslui'}
Performance in seconds of Depth First Search algorithm's single execution is => 0.007287263870239258
"""