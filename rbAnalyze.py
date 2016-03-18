from rbGenerate import *
import numpy as np
import random
from os import walk

from sklearn import svm, metrics


def motionGraphToAdjMatrix(motionGraph, isDirected=False):
    n = motionGraph.numVertices
    adjMatrix = np.zeros((n, n), dtype=int)
    edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
    for edge in edges:
        adjMatrix[edge.startID, edge.endID] += 1
        if not isDirected:  # ensure symmetry
            adjMatrix[edge.endID, edge.startID] += 1
    return adjMatrix

#TODO: verify that summing logarithms is a valid technique
def fourGraphletFeatures(mat, multiCount=True, numSamples=10000):
    numVertices = mat.shape[0]
    # ensure that every possible set is included
    # frozensets are hashable
    featureDict = {frozenset([0]): 0,
                   frozenset([1]): 0,
                   frozenset([2]): 0,
                   frozenset([3]): 0,
                   frozenset([0, 1]): 0,
                   frozenset([0, 2]): 0,
                   frozenset([1, 2]): 0,
                   frozenset([1, 3]): 0,
                   frozenset([2, 3]): 0,
                   frozenset([0, 1, 2]): 0,
                   frozenset([1, 2, 3]): 0}

    for sample in range(numSamples):
        degCounts = [0, 0, 0, 0]
        edgeCounts = {}
        sampVerts = random.sample(range(numVertices), 4)
        #for each possibility of four verticecs
        for i in range(4):
            for j in range(4):
                numEdges = mat[sampVerts[i], sampVerts[j]]
                if numEdges > 0:
                    #initialize if not in dictionary
                    if frozenset([i,j]) not in edgeCounts:
                        edgeCounts[frozenset([i,j])] = 0
                    edgeCounts[frozenset([i,j])] += numEdges
                    degCounts[j] += 1
        #Add weight for the number of graphs in a multigraph
        numGraphs = 1
        if multiCount:
            numGraphs = 0
            for val in edgeCounts.values():
                numGraphs += np.log(val)
        degSet = frozenset(degCounts)
        featureDict[degSet] += numGraphs
    featureList = []
    for key in sorted(featureDict):
        featureList += [featureDict[key]]
    #print(featureList)
    return featureList

#----------------------------------------

def analyze():
    f = []
    for(path, dirs, files) in walk("Data\\Pickles"):
        f.extend(files)
        break

    featureVectors = []
    labels = []
    for file in f:
        print("Generating features for file: ", file)
        mg = pickle.load(open("Data\\Pickles\\" + file, "rb"))
        mgMat = motionGraphToAdjMatrix(mg)
        # print(mg.label)
        # generate feature vectors
        featureVectors += [fourGraphletFeatures(mgMat)]
        labels += [mg.label]
    # apply svm
    featureArray = np.array(featureVectors)
    labelArray = np.array(labels)

    classifier = svm.SVC(kernel='linear', gamma=0.001)
    classifier.fit(featureArray[::2], labelArray[::2])
    expected = labelArray[1::2]
    predicted = classifier.predict(featureArray[1::2])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(expected, predicted))

analyze()
