# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:32:03 2019

@author: 13383861
"""

# TSPLib class takes in a set of nodes and a distance matrix
# provides various methods of calculating tsp solutions.
# In practice, using the CERES solver though google OR tools would probably be the best way of solving the problem

from typing import Dict
import random
from sklearn.cluster import KMeans
from itertools   import permutations, combinations
from functools   import lru_cache as cache

class TSPSolver:
    '''Given a distance matrix and a list of nodes, returns solutions to the TSP problem'''

        
    def __init__(self, nodes, distance_matrix: Dict = None, dist_calculator = None, split_function = None, start_node = None):
        '''Given a set of nodes and a function which returns the distance between them, initialises the distance matrix'''
        self.nodes = nodes
        if distance_matrix:
            self.distance_matrix = distance_matrix
            self.verify_distance_matrix()
        if dist_calculator:
            self._create_dist_matrix(dist_calculator)
        else:
            raise Exception("You need to provide a distance calculator in order to determine ")
            
        self.split_function = split_function
        
        self._verify_distance_matrix()
        self.rep_improve_nn_tsp_multi = self.bind(self.rep_improve_nn_tsp, 10)
        
        
    def _create_dist_matrix(self, dist_calculator):
        self.distance_matrix = {}
        for grid_loc1 in self.nodes:
            #initialise as empty
            self.distance_matrix[grid_loc1] = {}
            for grid_loc2 in self.nodes:
                #if grid_loc2 not in distance_matrix:
                self.distance_matrix[grid_loc1][grid_loc2] = dist_calculator(grid_loc1, grid_loc2)
                
                
    
    def _verify_distance_matrix(self):
        '''Ensures that the distance matrix has been properly constructed'''
        #make sure that can query the distance from any node to any other node
        for node1 in self.nodes:
            for node2 in self.nodes:
                try:
                    res = self.distance_matrix[node1][node2]
                    if not isinstance(res, float) and not isinstance(res, int):
                        raise Exception("distance matrix must contain float or integer distances, not {}".format(type(res)))
                except Exception as e:
                    raise e
                    
    def get_distance(self, node1, node2):
        return self.distance_matrix[node1][node2]
        
    def _all_tours(self, nodes):
        start, *others = nodes
        return [[start] + list(perm) for perm in permutations(others)]
        
    def all_tours(self):
        "Return a list of non-redundant tours (permutations of cities)."
        start, *others = self.nodes
        return [[start] + list(perm) for perm in permutations(others)]
        
    def shortest_tour(self, tours): 
        return self._shortest_tour(tours, self.distance_matrix)
    
    def _shortest_tour(self, tours, distance_matrix): 
        return min(tours, key=lambda tour: self._distancePath(tour, distance_matrix))    
    
    def _exhaustive_tsp(self, vertices, distance_matrix):
        '''Generate all possible tours of the cities and choose the shortest tour.
        O(n!)'''
        if len(vertices) > 12:
            print("computer will freeze with this many cities")
            return
        return self._shortest_tour(self._all_tours(vertices), distance_matrix)
    
    def exhaustive_tsp(self):
        return self._exhaustive_tsp(self.nodes, self.distance_matrix)
    
    #%%
    def _nn_tsp(self, cities: "dict city_name: city", distance_matrix, start=None):
        """Start the tour at the given start city (default: first city); 
        at each step extend the tour by moving from the previous city 
        to its nearest neighbor that has not yet been visited.
        O(n^2)"""
        C = start or self._first(cities)
        tour = [C]
        unvisited = set(set(cities) - {C})
        while unvisited:
            C = self._nearest_neighbor(C, unvisited, distance_matrix)
            tour.append(C)
            unvisited.remove(C)
        return tour   
        #return list(map(lambda x: cities[x],tour))
        
    def nn_tsp(self, start = None):
        return self._nn_tsp(self.nodes, self.distance_matrix, start)
    
    def _first(self, collection): 
        return next(iter(collection))
    
    def _nearest_neighbor(self, A, nodes, distance_matrix):
        "Find the city in cities that is nearest to city A. O(nlogn)"
        return min(nodes, key=lambda x: distance_matrix[A][x])
    
    def nearest_neighbor(self, A, nodes):
        "Find the city in cities that is nearest to city A. O(nlogn)"
        return self._nearest_neighbor(A, nodes, self.distance_matrix)
        
    #%%
    def _rep_nn_tsp(self, nodes, distance_matrix, k=25):
        '''Repeat nn_tsp starting from k different cities; pick the shortest tour.
        O(k * n^2)'''
        return self.shortest_tour((self.nn_tsp(nodes,distance_matrix, start) for start in self.sample(nodes, k)), distance_matrix)

    def rep_nn_tsp(self, nodes= None, k=25):
        '''Repeat nn_tsp starting from k different cities; pick the shortest tour.
        O(k * n^2)'''
        if not nodes:
            return self._rep_nn_tsp(self.nodes, self.distance_matrix, k)
        else:
            return self._rep_nn_tsp(nodes, self.distance_matrix, k)
    
    def sample(self,population, k, seed=42):
        "Return a list of k elements sampled from population. Set random.seed."
        if k >= len(population): 
            return population
        else:
            random.seed((k, seed))
            return random.sample(population, k)
        
    #%%
            
    def _reverse_segment_if_improvement(self, tour, i, j, distance_matrix):
        "If reversing tour[i:j] would make the tour shorter, then do it. Length of  O(j-i)" 
        
        # Given tour [...A,B...C,D...], consider reversing B...C to get [...A,C...B,D...]
        A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
        # Are old links (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
        if distance_matrix[A][B] + distance_matrix[C][D] > distance_matrix[A][C] + distance_matrix[B][D]:
            tour[i:j] = reversed(tour[i:j])
            return True   
        
    def reverse_segment_if_improvement(self, tour, i, j):
        "If reversing tour[i:j] would make the tour shorter, then do it. Length of  O(j-i)" 
        
        # Given tour [...A,B...C,D...], consider reversing B...C to get [...A,C...B,D...]
        A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
        # Are old links (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
        if self.distance_matrix[A][B] + self.distance_matrix[C][D] > self.distance_matrix[A][C] + self.distance_matrix[B][D]:
            tour[i:j] = reversed(tour[i:j])
            return True   
    
    #%%
    def _improve_tour(self, tour, distance_matrix):
        "Try to alter tour for the better by reversing segments. O(N^2 * N^2 * random factor due to fact could be no improvements) = O(N^4) "
        while True:
            improvements = {self._reverse_segment_if_improvement(tour, i, j, distance_matrix)
                            for (i, j) in self.subsegments(len(tour))}
            if improvements == {None}:
                return tour
    
    def improve_tour(self, tour):
        "Try to alter tour for the better by reversing segments. O(N^2 * N^2 * random factor due to fact could be no improvements) = O(N^4) "
        return self._improve_tour(tour, self.distance_matrix)
    #%%
    #@cache()
    #def subsegments(N):
    #    "Return (i, j) pairs denoting tour[i:j] subsegments of a tour of length N. O()"
    #    return [(i, i + length)
    #            for length in reversed(range(2, N))
    #            for i in reversed(range(N - length + 1))]
        
    #%%
    @cache()
    def subsegments(self,N):
        "Return (i, j) pairs denoting tour[i:j] subsegments of a tour of length N. O(N^2)"
        return [(i, i + length)
                for length in range(N-1, 1, -1)
                for i in range(N - length, -1, -1)]
        
    #%%
    def _improve_nn_tsp(self, nodes, distance_matrix): 
        "Improve the tour produced by nn_tsp."
        return self.improve_tour(self.nn_tsp(nodes, distance_matrix))
    
    def improve_nn_tsp(self, nodes = None): 
        "Improve the tour produced by nn_tsp."
        if not nodes:
            return self._improve_nn_tsp(self.nodes, self.distance_matrix)
        else:
            return self._improve_nn_tsp(nodes, self.distance_matrix)
    
    def _rep_improve_nn_tsp(self, nodes, distance_matrix, k=5):
        "Run nn_tsp from k different starts, improve each tour; keep the best."
        return self._shortest_tour((self._improve_tour(self._nn_tsp(nodes, distance_matrix, start), distance_matrix) 
                             for start in self.sample(nodes, k)), distance_matrix)
        
    def rep_improve_nn_tsp(self, k=5):
        return self._rep_improve_nn_tsp(self.nodes, self.distance_matrix, k)

    #%%
    @cache()
    def bind(self, fn, *extra):
        "Bind extra arguments; also assign .__name__"
        newfn = lambda *args: fn(*args, *extra)
        newfn.__name__ = fn.__name__  + ''.join(', ' + str(x) for x in extra)
        return newfn
    
    
    
    #%%
    def join_endpoints(self, endpoints, A, B):
        "Join segments [...,A] + [B,...] into one segment. Maintain `endpoints`. O(len(Aseg) + len(Bseg))"
        Aseg, Bseg = endpoints[A], endpoints[B]
        #reverse to make sure that A is at the end of an ordered segment.
        if Aseg[-1] is not A: Aseg.reverse()
        #reverse to make sure that B is at the start of an ordered segment.
        if Bseg[0]  is not B: Bseg.reverse()
        Aseg += Bseg
        del endpoints[A], endpoints[B] 
        #update dict so that the start of segment A (merged with B) and then end of segment A are both the merged ASeg
        endpoints[Aseg[0]] = endpoints[Aseg[-1]] = Aseg
        return Aseg
    
    def _shortest_links_first(self, nodes, distance_matrix):
        "Return all links between each possible pair of cities, sorted shortest first. O(n^(n/2) * log(n^(n/2)))"
        return sorted(combinations(nodes, 2), key=lambda link: distance_matrix[link[0]][link[1]])
    
    def shortest_links_first(self):
        "Return all links between each possible pair of cities, sorted shortest first. O(n^(n/2) * log(n^(n/2)))"
        return self._shortest_links_first(self.nodes, self.distance_matrix)
           
    
    '''Greedy Algorithm: Maintain a set of segments; intially each city defines its own 1-city segment. 
    Find the shortest possible link that connects two endpoints of two different segments, and join those segments with that link.
    Repeat until we form a single segment that tours all the cities.'''
    
    def _greedy_tsp(self, nodes, distance_matrix):
        """Go through links, shortest first. Use a link to join segments if possible."""
        endpoints = {C: [C] for C in nodes} # A dict of {endpoint: segment}, initialised as 1-length segments
        for (A, B) in self._shortest_links_first(nodes, distance_matrix):
            #check both A and B are still valid endpoints and that they are not endpoints of the same segment
            if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
                new_segment = self.join_endpoints(endpoints, A, B)
                if len(new_segment) == len(nodes):
                    return new_segment
                
    def greedy_tsp(self):
        """Go through links, shortest first. Use a link to join segments if possible."""
        return self._greedy_tsp(self.nodes, self.distance_matrix)
        
    def _improve_greedy_tsp(self, nodes, distance_matrix): 
        return self._improve_tour(self._greedy_tsp(nodes, distance_matrix), distance_matrix)
                
    def improve_greedy_tsp(self): 
        return self._improve_tour(self._greedy_tsp(self.nodes, self.distance_matrix), self.distance_matrix)
    
    #%%
    def _join_tours(self, tour1, tour2, distance_matrix):
        "Consider all ways of joining the two tours together, and pick the shortest."
        segments1, segments2 = self.rotations(tour1), self.rotations(tour2)
        return self._shortest_tour((s1 + s2
                             for s1 in segments1
                             for s  in segments2
                             for s2 in (s, s[::-1])), distance_matrix)

    
    def join_tours(self, tour1, tour2):
        return self._join_tours(tour1, tour2, self.distance_matrix)
    
    def rotations(self, sequence):
        "All possible rotations of a sequence."
        # A rotation is some suffix of the sequence followed by the rest of the sequence.
        return [sequence[i:] + sequence[:i] for i in range(len(sequence))]
    
    #%%
    
    #def cluster(cities: "a dict of type city_name: city", no_clusters):
    #    #distances = [(distance_lat(zero_coord, city[1]), distance_lng(zero_coord, city[1])) for city in cities_values_sorted]
    #    distances = [(city.x_val, city.y_val) for city in cities]
    #    data = np.array(distances)
    #    test = KMeans(n_clusters = no_clusters)
    #    test.fit(data)
    #    labels = test.labels_
    #    #reorder by cluster
    #    clustered_cities = {rav_no:[] for rav_no in range(no_clusters)}
    #    for city, label_index in zip(cities,labels):
    #        clustered_cities[label_index]+=[city]
    #    return clustered_cities
    
    #%%
        
    def _split_cities(self, nodes, split_fn: "a function that takes in a city and return two values that can split the cities into clustered regions"):
        "Split cities vertically if map is wider; horizontally if map is taller."
        width  = self.extent(list(map(lambda node: split_fn(node)[0], nodes)))
        height = self.extent(list(map(lambda node: split_fn(node)[1], nodes)))
        nodes= sorted(nodes, key= lambda city:(city.x_val if (width > height) else city.y_val))
        middle = len(nodes) // 2
        return nodes[:middle], nodes[middle:]
    
    def split_cities(self):
        "Split cities vertically if map is wider; horizontally if map is taller."
        if not self.split_function:
            print("split_cities has no method of splitting")
        else:
            return self._split_cities(self.nodes, self.split_function)
    
    #def split_cities(cities):
    #    '''use k-means'''
    #    return cluster(cities, 2)
    
    def extent(self, numbers):
        return max(numbers) - min(numbers)
    
    def _divide_tsp(self, nodes, distance_matrix, n=10):
        """Find a tour by divide and conquer: if number of cities is n or less, use exhaustive search.
        Otherwise, split the cities in half, solve each half recursively, 
        then join those two tours together."""
        #print('cities: ',cities)
        if len(nodes) <= n:
            return self._exhaustive_tsp(nodes, distance_matrix)
        else:
            half1, half2 = self._split_cities(nodes)
            return self._join_tours(self._divide_tsp({city for city in half1 if city in nodes}, distance_matrix), self._divide_tsp({city for city in half2 if city in nodes}, distance_matrix), distance_matrix)
        
    def divide_tsp(self, n=10):
        return self._divide_tsp(self.nodes, self.distance_matrix)
        
    # TO DO: functions: split_cities, join_tours
    def _improve_divide_tsp(self, cities, distance_matrix): 
        return self._improve_tour(self._divide_tsp(cities, distance_matrix), distance_matrix)

    def improve_divide_tsp(self): 
        return self.improve_tour(self.divide_tsp())


    #%%
    def _mst(self, vertexes, distance_matrix):
        """Given a set of vertexes, build a minimum spanning tree: a dict of the form 
        {parent: [child...]}, spanning all vertexes."""
        tree  = {self._first(vertexes): []} # the first city is the root of the tree.
        links = self._shortest_links_first(vertexes, distance_matrix)
        while len(tree) < len(vertexes):
            (A, B) = self._first((A, B) for (A, B) in links if (A in tree) ^ (B in tree))
            if A not in tree: (A, B) = (B, A)
            tree[A].append(B)
            tree[B] = []
        return tree
    
    def mst(self):
        """Given a set of vertexes, build a minimum spanning tree: a dict of the form 
        {parent: [child...]}, spanning all vertexes."""
        return self._mst(self.nodes, self.distance_matrix)
    #%%
    def _mst_tsp(self, cities, distance_matrix):
        "Create a minimum spanning tree and walk it in pre-order, omitting duplicates."
        return list(self._preorder_traversal(self._mst(cities, distance_matrix), self._first(cities)))
    
    def mst_tsp(self):
        "Create a minimum spanning tree and walk it in pre-order, omitting duplicates."
        return self._mst_tsp(self.nodes, self.distance_matrix)
    
    
    def _preorder_traversal(self, tree, root):
        "Traverse tree in pre-order, starting at root of tree."
        yield root
        for child in tree.get(root, ()):
            yield from self._preorder_traversal(tree, child)
            
    def _improve_mst_tsp(self, cities, distance_matrix): 
        return self._improve_tour(self._mst_tsp(cities, distance_matrix), distance_matrix)
    
    def improve_mst_tsp(self): 
        return self.improve_tour(self.mst_tsp())
    #%%
        
    @cache(None)
    def shortest_segment(self, A, Bs, C):
        "The shortest segment starting at A, going through all Bs, and ending at C."
        if not Bs:
            return [A, C]
        else:
            return min((self.shortest_segment(A, Bs - {B}, B, self.distance_matrix) + [C] for B in Bs),
                       key=lambda x: self.segment_length(x, self.distance_matrix))
                
    def segment_length(self, segment):
        "The total of distances between each pair of consecutive cities in the segment."
        # Same as tour_length, but without distance(tour[0], tour[-1])
        return self._sumDistance(segment, self.distance_matrix)
    
    
    
    def _ensemble_tsp(self, nodes, distance_matrix, algorithms=None): 
        "Apply an ensemble of algorithms to cities and take the shortest resulting tour."
        ensemble = {self._rep_improve_nn_tsp, self._improve_greedy_tsp, self._improve_mst_tsp}
        if self.split_function:
            ensemble.add(self._improve_divide_tsp)
        return self._shortest_tour((tsp(nodes, distance_matrix) for tsp in (algorithms or ensemble)), distance_matrix)
    
    def ensemble_tsp(self, algorithms=None): 
        "Apply an ensemble of algorithms to cities and take the shortest resulting tour."
        ensemble = {self.rep_improve_nn_tsp, self.improve_greedy_tsp, self.improve_mst_tsp}
        if self.split_function:
            ensemble.add(self.improve_divide_tsp)
        return self.shortest_tour((tsp() for tsp in (algorithms or ensemble)))    
    
    def distance(self, a, b):
        """Calculates distance between two latitude-longitude coordinates in meters."""
        return self.distance_matrix[a][b]
    
    def _distancePath(self, l: "list of city", distance_matrix):
        #distance from new york included, which is set as start point
        #dist = distance_matrix['0'][l[0]]
        dist = 0
        for i in range(len(l)):
            dist += distance_matrix[l[i-1]][l[i]]
        return dist
    
    def distancePath(self, l: "list of city"):
        #distance from new york included, which is set as start point
        #dist = distance_matrix['0'][l[0]]
        return self._distancePath(l, self.distance_matrix)
    
    def minDistance(self, state: "list of city"):
        return min([self.distancePath(path) for path in state])
    
    def maxDistance(self, state: "list of city"):
        return max([self.distancePath(path) for path in state])
           
    def sumDistance(self, l: "list of city"):
        return self.distancePath(l) 
    
    
    
#run some assertions as quick test
if __name__ == '__main__':
    
    nodes = ['1','2','4','5','8']
    
    def dist_calculator(a, b):
        return int(a) - int(b)
    
    tsp_solver = TSPSolver(nodes, dist_calculator = dist_calculator)
    print(tsp_solver.distance_matrix)
    res = tsp_solver.ensemble_tsp()
    print(res)
    
    
    
    
    
    
    
    
    