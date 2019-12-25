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
from Tree import Tree
from Tree import Node


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
    #print("Tree Topology: ", tree_topology)
    #print("Values for theta: ", theta)

    #Marginalize out parent to make a new categorical distribution
    #Repeat until child node
    #Repeat until all children has a 5x1 categorical distribution
    #When finished, we have a categorical distribution for all children
    #Now simply calculate the joint probability of the child nodes. Remember d-seperation and since we have margi. the
    #parents, these can now be considered independent

    #Calculate sub-problem
    likelihood = calculateSubproblem(tree_topology, theta, beta)

    return likelihood

def calculateSubproblem(tree_topology, theta, beta, parent):
    """
            This function calculates a sub-problem and returns the likelihood of the the branch values of that sub-tree
            :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
            :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
            :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                        Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
            :return: likelihood: The likelihood of beta. Type: float.
    """

    if betaIsLeaf(beta[parent]):
        print("Leaf likelihood: ", theta[beta[parent]])
        return theta[beta[parent]]

    parentCat = theta[parent]
    child1, child2 = findChildren(tree_topology, parent)
    child1Cat = theta[child1]
    child2Cat = theta[child2]

    theta[child1] = np.dot(parentCat, child1Cat)
    theta[child2] = np.dot(parentCat, child2Cat)

    print("New Cat distribution ", theta[child1])

    return calculateSubproblem(tree_topology, theta, beta, child1)*calculateSubproblem(tree_topology, theta, beta, child2)

def betaIsLeaf(betaNode):
    if betaNode != np.nan:
        return True
    else:
        return False

def findChildren(tree_topology, nextNode):
    child1 = np.nan
    child2 = np.nan

    child1 = findChild(tree_topology, nextNode, nextNode)
    child2 = findChild(tree_topology, nextNode, child1)

    return child1, child2

def findChild(tree_topology, nextNode, startNode):
    child = np.nan
    while startNode < len(tree_topology):
        if tree_topology[startNode] == nextNode:
            child = startNode
            break
        startNode += startNode + 1
    return child

def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_3_medium_tree.pkl" #"data/q2_3_small_tree.pkl"  , "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
