from rbGenerate import *
import numpy as np
import random
from os import walk

from sklearn import svm, metrics

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
        #for each possibility of four vertices
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


def motionGraphToAdjMatrix(motionGraph, isDirected=False):
    n = motionGraph.numVertices
    adjMatrix = np.zeros((n, n), dtype=int)
    edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
    for edge in edges:
        adjMatrix[edge.startID, edge.endID] += 1
        if not isDirected:  # ensure symmetry
            adjMatrix[edge.endID, edge.startID] += 1
    return adjMatrix


def serializeGraphs():
    f = []
    for(path, dirs, files) in os.walk("Data\\Graphs"):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)
    for fileName in f:
        print("Generating motion graph for file: ", fileName)
        mg = genGraph(fileName)
        pickleFile = fileName.replace("Graphs", "Pickles")
        pickleFile = os.path.splitext(pickleFile)[0] + ".p"
        pickle.dump(mg, open(pickleFile, "wb"))


def serializeAdjMats(dirToPopulate="Data\\Pickles\\AdjMats\\Undirected"):
    sourceDir = "Data\\Pickles\\MotionGraphs"
    f = []
    for(path, dirs, files) in os.walk(sourceDir):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)

    for fileName in f:
        print("Generating adjacency matrix for file: ", fileName)
        mg = pickle.load(open(fileName, "rb"))
        mgMat = motionGraphToAdjMatrix(mg)
        adjMatLocation = fileName.replace(sourceDir, dirToPopulate)
        pickle.dump((mg.label, mgMat), open(adjMatLocation, "wb"))


def serializeFeatVecs(dirToPopulate="Data\\Pickles\\FeatureVectors\\4Graphlet", kernelFunc=fourGraphletFeatures):
    sourceDir = "Data\\Pickles\\AdjMats\\Undirected"
    f = []
    for(path, dirs, files) in os.walk(sourceDir):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)

    for fileName in f:
        print("Generating feature vector for file: ", fileName)
        (label, mgMat) = pickle.load(open(fileName, "rb"))
        featureVector = kernelFunc(mgMat)

        featVecLocation = fileName.replace(sourceDir, dirToPopulate)
        pickle.dump((featureVector, label), open(featVecLocation, "wb"))

#----------------------------------------

def analyze(dirToAnalyze="Data\\Pickles\\FeatureVectors\\4Graphlet"):
    print("Loading feature vectors")
    f = []
    for(path, dirs, files) in walk(dirToAnalyze):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)

    featureVectors = []
    labels= []
    for fileName in f:
        (featureVector, label) = pickle.load(open(fileName, "rb"))
        featureVectors += [featureVector]
        labels += [label]

    # apply svm
    featureArray = np.array(featureVectors)
    labelArray = np.array(labels)
    print("Learning...")
    classifier = svm.SVC(kernel='linear', gamma=0.001)
    classifier.fit(featureArray[::2], labelArray[::2])
    expected = labelArray[1::2]
    predicted = classifier.predict(featureArray[1::2])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(expected, predicted))

#analyze("Data\\Pickles\\FeatureVectors\\4Graphlet\\590Testing")
