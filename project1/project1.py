import sys

import networkx as nx

import numpy as np
import csv
import time
import random
from scipy.special import gammaln

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{},{}\n".format(idx2names[edge[0]], idx2names[edge[1]]))



def prior(vars, G):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    
    prior_matrix = [np.ones((int(q[i]), r[i])) for i in range(n)]
    return prior_matrix


def sub2ind(siz, x):
    k = np.concatenate([np.array([1]), np.cumprod(siz[:-1])])
    return int(np.dot(k, x - 1) + 1)

def statistics(vars, G, D):
    n = D.shape[0]
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[j] for j in list(G.predecessors(i))]) for i in range(n)]
    M = [np.zeros((int(q[i]), r[i])) for i in range(n)]
    for o in range(D.shape[1]):
        for i in range(n):
            k = D[i, o]
            parents = list(G.predecessors(i))
            j = 1
            if parents:
                j = sub2ind([r[p] for p in parents], np.array([D[parent, o] for parent in parents]))

            M[i][j - 1, k - 1] += 1.0
    
    return M



def bayesian_score_component(M, alpha):
    p = np.sum(gammaln(alpha + M))
    p -= np.sum(gammaln(alpha))
    p += np.sum(gammaln(np.sum(alpha, axis=1)))
    p -= np.sum(gammaln(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)  # Assuming you have a statistics function
    alpha = prior(vars, G)     # Assuming you have a prior function
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])

def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
            
    
    # read the variables and data from csv file
    variables = None
    data = []
    names2idx = {}

    # Open the CSV file
    with open(infile, "r") as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # get the variables and construct names2idx
        variables = next(csv_reader)
        names2idx = {v: idx for idx, v in enumerate(variables)}

        # Read the remaining rows (data) and store them in the 2D matrix
        for row in csv_reader:
            data.append(row)
        
        # now we want to convert [[3,3,2,3,1,3], [1,3,2,3,2,3]] to [[3,1], [3,3], [2,2], [3,3], [1,2], [3,3]]
        # D = [[] for _ in range(len(variables))]
        # for row in data:
        #     for col in range(len(row)):
        #         D[col].append(int(row[col]))
        D = np.array(data).astype(int).T

        
        # build graph from the gph file 

        # create empty graph
        G = nx.DiGraph()
        # read nodes from .gph file and add to graph
        with open(outfile, "r") as file:
            for line in file:
                parent, child = line.strip().split(",")
                G.add_edge(names2idx[parent], names2idx[child])
        

        max_values = np.amax(D, axis=1)
        # adjust variable list
        vars = [{"symbol": key, "r": max_values[idx]} for idx, key in enumerate(variables)]     

        
    # print(G.edges)
    # print(G.nodes)
    # print(vars)
    return bayesian_score(vars, G, D)


class K2Search:
    def __init__(self, ordering=None):
        self.ordering = ordering

def K2fit(method, vars, D):
    G = nx.DiGraph()
    num_vars = len(vars)
    G.add_nodes_from(range(num_vars))
    
    for k in range(1, num_vars):
        i = method.ordering[k]

        y = bayesian_score(vars, G, D)

        while True:
            y_best, j_best = float('-inf'), 0

            for j in method.ordering[:k]:
                if not G.has_edge(j, i):
                    G.add_edge(j, i)
                    y_prime = bayesian_score(vars, G, D)

                    if y_prime > y_best:
                        y_best, j_best = y_prime, j

                    G.remove_edge(j, i)

            if y_best > y:
                y = y_best
                G.add_edge(j_best, i)
            else:
                break

    return G

class LocalDirectedGraphSearch:
    def __init__(self, G=None, k_max=None):
        self.G = G
        self.k_max = k_max

def rand_graph_neighbor(G):
    n = G.number_of_nodes()
    i = random.randint(0, n - 1)
    j = (i + random.randint(1, n - 1)) % n
    G_copy = G.copy()
    if G_copy.has_edge(i, j):
        G_copy.remove_edge(i, j)
    else:
        G_copy.add_edge(i, j)
    return G_copy

def LocalSearch_fit(method, vars, D):
    G = method.G
    y = bayesian_score(vars, G, D)
    for k in range(1, method.k_max + 1):
        G_prime = rand_graph_neighbor(G)
        if nx.is_directed_acyclic_graph(G_prime):
            y_prime = bayesian_score(vars, G_prime, D)
            if y_prime > y:
                y, G = y_prime, G_prime
    return G

def findBestG(method, infile, outfile):

    # build the variable list and read data from infile.csv
    # read the variables and data from csv file
    variables = None
    data = []

    # Open the CSV file
    with open(infile, "r") as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # get the variables and construct names2idx, idx2names
        variables = next(csv_reader)
         
        names2idx = {v: idx for idx, v in enumerate(variables)}
        idx2names = {idx: v for idx, v in enumerate(variables)}

        # Read the remaining rows (data) and store them in the 2D matrix
        for row in csv_reader:
            data.append(row)

        D = np.array(data).astype(int).T
        max_values = np.amax(D, axis=1)
        
        # adjust variable list
        vars = [{"symbol": key, "r": max_values[idx]} for idx, key in enumerate(variables)] 

    # specify the structure algorithm and find the best Graph
    
    # K2 algorithm method
    if isinstance(method, K2Search):
        test_ordering = [i for i in range(len(vars))]
        test_K2Search = K2Search(test_ordering)
        test_G = K2fit(test_K2Search, vars, D)
    
    # LocalSearch algorithm method
    elif isinstance(method, LocalDirectedGraphSearch):
        pre_G_name = input("Which Graph do you want to do the local search on?[Enter file name with .gph]: ")
        pre_G = nx.DiGraph()
        # read nodes from .gph file and add to graph
        with open(pre_G_name, "r") as file:
            for line in file:
                parent, child = line.strip().split(",")
                pre_G.add_edge(names2idx[parent], names2idx[child])
        
        method.G = pre_G
        method.k_max = int(input("Enter the number of iterations for each local node, i.e k_max: "))
        test_G = LocalSearch_fit(method, vars, D)

    # write the best graph to .gph outfile
    write_gph(test_G, idx2names, outfile)
    
    return test_G


    
    

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    # start = time.time()
    # print("The Bayesian score of the given structure and data is: " + str(compute(inputfilename, outputfilename)))
    # end = time.time()
    # print("Execution time: " + str(round(end - start, 2)) + " s")

    # using k2 algorithm

    # print("\nFinding the best graph...\n")
    # start = time.time()
    # findBestG(K2Search(), inputfilename, outputfilename)
    # end = time.time()
    # print("Algorithm completed!\n")
    # print("The Bayesian score of the given structure and data is: " + str(compute(inputfilename, outputfilename)))
    # print("Execution time: " + str(round(end - start, 2)) + " s")

    # using localsearch algorithm
    
    print("\nFinding the best graph...\n")
    start = time.time()
    findBestG(LocalDirectedGraphSearch(), inputfilename, outputfilename)
    end = time.time()
    print("Algorithm completed!\n")
    print("The Bayesian score of the given structure and data is: " + str(compute(inputfilename, outputfilename)))
    print("Execution time: " + str(round(end - start, 2)) + " s")


if __name__ == '__main__':
    main()
