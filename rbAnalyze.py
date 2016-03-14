from rbGenerate import *
import numpy as np
import random
from os import walk


def motionGraphToAdjMatrix(motionGraph, isDirected=False):
    n = motionGraph.numVertices
    adjMatrix = np.zeros((n, n), dtype=int)
    edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
    for edge in edges:
        adjMatrix[edge.startID, edge.endID] += 1
        if not isDirected:  # ensure symmetry
            adjMatrix[edge.endID, edge.startID] += 1
    return adjMatrix


def fourGraphletFeatures(mat, numSamples=10000):
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
        sampVerts = random.sample(range(numVertices), 4)
        for i in range(4):
            for j in range(4):
                if mat[sampVerts[i], sampVerts[j]] > 0:
                    degCounts[j] += 1
        degSet = frozenset(degCounts)
        featureDict[degSet] += 1
    featureList = []
    for key in sorted(featureDict):
        featureList += [featureDict[key] / numSamples]
    return featureList


def flattenList(l):
    return [item for sublist in l for item in sublist]


def chunkList(l, chunkSize=10):
    chunkInds = range(0, len(l), chunkSize * 2)
    firstChunks = [l[i:i + chunkSize] for i in chunkInds]
    secondChunks = [l[i + chunkSize:i + 2 * chunkSize] for i in chunkInds]
    return (flattenList(firstChunks), flattenList(secondChunks))

#----------------------------------------


def analyze():
    f = []
    for(path, dirs, files) in walk("Data\\Graphs"):
        f.extend(files)
        break

    featureVectors = []
    labels = []
    for file in f:
        file = "Data\\Graphs\\" + file
        mg = genGraph(file)
        mgMat = motionGraphToAdjMatrix(mg)
        # print(mg.label)
        # generate feature vectors
        featureVectors += [fourGraphletFeatures(mgMat)]
        labels += [mg.label]
    # apply svm
    featureChunks = chunkList(featureVectors)
    labelChunks = chunkList(labels)
    featureArray = np.array(featureChunks[0])
    labelArray = np.array(labelChunks[0])
    print(np.array(labels))
    print("----------------")
    print(labelArray)
