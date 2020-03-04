""" This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_3.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_3_small_tree, q_2_3_medium_tree, q_2_3_large_tree).
    Each tree have 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""

import numpy as np
import math as m
from Tree import Tree
from Tree import Node
from collections import defaultdict

s_collection = defaultdict(dict)
t_collection = defaultdict(dict)

def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):
    """

    # TODO Add your code here
    #Print info about tree
    print("Tree Topology: ", tree_topology)
    print("Values for theta: ", theta)
    print("Beta list: ", beta)

    #Marginalize out parent to make a new categorical distribution
    #Repeat until child node
    #Repeat until all children has a 5x1 categorical distribution
    #When finished, we have a categorical distribution for all children
    #Now simply calculate the joint probability of the child nodes. Remember d-seperation and since we have margi. the
    #parents, these can now be considered independent

    #Calculate sub-problem
    #likelihood = calculateSubproblem(tree_topology, theta, beta, 0, theta[0])

    #Calculate and store s_values
    root = 0
    for category_i, prob_i in enumerate(theta[0]):
         s(root, category_i, beta, theta, tree_topology)

    #Calculate final likelihood
    for leaf_node, category_i in enumerate(beta):
        #print(leaf_node)
        #print(category_i)
        if not np.isnan(category_i):
            #print(leaf_node)
            #print(category_i)
            return t(leaf_node, category_i, tree_topology[leaf_node], theta, tree_topology)*s_collection[leaf_node].get(int(category_i))

    return 0

def t(child, category, parent, theta, tree_topology):
    sibling = find_sibling(child, tree_topology)

    if np.isnan(parent):
        return theta[child][category] #This is the root
    if t_collection[child].get(category) is not None:
        return t_collection[child].get(category)

    parent = int(parent)
    category = int(category)
    likelihood = 0 #likelihood for sub-tree
    length_of_categorical_dist = len(theta[0])
    for i in range(length_of_categorical_dist):
        for j in range(length_of_categorical_dist):
            '''print(child)
            print(category)
            print(i)
            print(sibling)
            print(j)
            print(parent)'''
            likelihood += theta[child][category][i]*theta[sibling][i][j]*s_collection[sibling].get(j)*t(parent, j, tree_topology[parent], theta, tree_topology)

    t_collection[child][category] = likelihood

    return likelihood

def s(parent, category, beta, theta, tree_topology):
    if s_collection[parent].get(category) is not None:
        return s_collection[parent].get(category)
    if not np.isnan(beta[parent]): #Check if leaf node
        #print("Returning!!!")
        s_collection[parent][category] = 1 if beta[parent] == int(category) else 0
        #print("Parent: ", parent)
        #print(s_collection[parent][category])
        return s_collection[parent][category]

    child1, child2 = findChildren(tree_topology, parent)
    #print(child1, child2)

    sub_likelihood_1 = 0
    sub_likelihood_2 = 0

    for category_j, prob_j in enumerate(theta[child1][category]):
        sub_likelihood_1 += (s(child1, category_j, beta, theta, tree_topology) * prob_j)

    for category_j, prob_j in enumerate(theta[child2][category]):
        sub_likelihood_2 += s(child2, category_j, beta, theta, tree_topology) * prob_j
        #print(child2)

    #print(sub_likelihood_1*sub_likelihood_2)
    s_collection[parent][category] = sub_likelihood_1*sub_likelihood_2
    #print(s_collection[parent].get(category))
    return sub_likelihood_1*sub_likelihood_2

def findChildren(tree_topology, parent):
    child1 = findChild(tree_topology, parent, parent)
    print("-----------------------------CHILD1--------------------------------- ::: ", child1)
    child2 = findChild(tree_topology, parent, child1+1)
    print("-----------------------------CHILD---------------------------------- ::: ", child2)
    return child1, child2

def findChild(tree_topology, parent, startNode):
    while startNode < len(tree_topology):
        print(startNode)
        if tree_topology[startNode] == parent:
            return startNode
        startNode += 1

def find_sibling(child, tree_topology):
    for possible_sibling, parent in enumerate(tree_topology):
        if np.isnan(parent) and np.isnan(tree_topology[child]) and child != possible_sibling:
            return possible_sibling
        elif parent == tree_topology[child] and child != possible_sibling:
            return possible_sibling
    return None #if there is no sibling


'''def calculateSubproblem(tree_topology, theta, beta, parent, parentCat):
    """
            This function calculates a sub-problem and returns the likelihood of the the branch values of that sub-tree
            :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
            :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
            :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                        Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
            :return: likelihood: The likelihood of beta. Type: float.
    """

    print("Beta of parent: ", beta[parent])

    if betaIsLeaf(beta[parent]):
        leafValue = int(beta[parent])
        print("Leaf likelihood: ", parentCat[leafValue])
        return parentCat[leafValue]

    child1, child2 = findChildren(tree_topology, parent)


    childCat1 = np.transpose(parentCat).dot(theta[child1])
    childCat2 = np.transpose(parentCat).dot(theta[child2])

    print("ParentCat: ", parentCat)
    print("ChildCat: ", theta[child1])

    print("New Cat distribution ", childCat1)

    return calculateSubproblem(tree_topology, theta, beta, child1, childCat1)*calculateSubproblem(tree_topology, theta, beta, child2, childCat2)

def betaIsLeaf(betaNode):
    return m.isnan(betaNode) is False

def findChildren(tree_topology, parent):
    child1 = findChild(tree_topology, parent, parent)
    #print("-----------------------------CHILD1--------------------------------- ::: ", child1)
    child2 = findChild(tree_topology, parent, child1)

    return child1, child2

def findChild(tree_topology, parent, startNode):
    while startNode < len(tree_topology):
        if tree_topology[startNode] == parent:
            return startNode
        startNode += 1

'''

def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_3_small_tree.pkl"
    #filename = "data/q2_3_medium_tree.pkl"
    #filename = "data/q2_3_large_tree.pkl"

    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    '''
    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)
    '''

    beta = t.filtered_samples[0]
    print("\n\tSample: ", 0, "\tBeta: ", beta)
    sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
    print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
