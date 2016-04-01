from rbGenerate import *
import numpy as np
import random
from os import walk

from sklearn import svm, metrics
import matplotlib.pyplot as plt

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


def partitionEdges(l):
    if len(l) == 0:
        return l
    retList = []
    curList = []
    curTime = l[0].time
    for edge in l:
        if edge.time == curTime:
            curList.append(edge)
        else:
            curTime = edge.time
            retList.append(curList)
            curList = [edge]
    return retList

# TODO: Actually figure out what you're trying to do
def messagePathFeatures(mg, maxPathLength=10):
    #Each robot counts the number of times a path of length n has ended on it
    numPathList = [0 for val in range(maxPathLength+1)]

    #for each observed robot
    for observedEdges in mg.graph:
        #each observer maintains a list of current sightings and paths
        currentPaths = [[] for observer in range(mg.numVertices)]

        for edge in observedEdges:
            start = edge.startID
            end = edge.endID
            if end in currentPaths[start] or len(currentPaths[start]) == maxPathLength:
                numPathList[len(currentPaths[start])] += 1
                currentPaths[end] = [end]
            elif len(currentPaths[end]) <= len(currentPaths[start]):
                currentPaths[end] = currentPaths[start] + [end]

            currentPaths[start] = []
    #if sum(numPathList) > 0:
    #    numPathList = [val / sum(numPathList) for val in numPathList]
    return numPathList



def tempMW():
    f = []
    for(path, dirs, files) in os.walk("Data\\Pickles\\MotionGraphs\\Robots\\RedVaries"):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)
    for fileName in f:
        mg = pickle.load(open(fileName, "rb"))
        print(messagePathFeatures(mg))
    #mg = pickle.load(open(f[0], "rb"))
    #print(messagePathFeatures(mg))

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

#TODO: Make this do things
#Create and export bar graph of ranges
def genBarGraph():
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

def genConfMatGraphic(cm, labels, title='Confusion matrix', cmap=plt.cm.OrRd):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('Expected Value')
    plt.xlabel('Predicted Value')

def saveConfMat(fileToAnalyze):
    labels = []
    cm = []
    f = open(fileToAnalyze)
    lines = list(f)

    labelStart = 6
    numLabels = 0
    while lines[labelStart + numLabels] != "\n":
        labels.append(lines[labelStart + numLabels].split()[0])
        numLabels += 1

    cmStart = labelStart + numLabels + 5 #magic number defined by file format
    cm = lines[cmStart:cmStart+numLabels]
    cm = [[int(val) for val in line.strip(" []\n").split()] for line in cm]
    cm = np.array(cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fileName = os.path.splitext(fileToAnalyze)[0]
    dirToSave = fileName.replace("Results", "Graphics") + ".png"
    fileName = os.path.basename(fileName)

    plt.figure()
    genConfMatGraphic(cm_normalized, labels, fileName)
    #plt.show()
    plt.savefig(dirToSave,bbox_inches="tight")
    plt.clf()


def generateCFGraphics(rootDir="Results\\4Graphlet"):
    f = []
    for(path, dirs, files) in walk(rootDir):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            if "EP.txt" not in fullName:
                f.append(fullName)
    for fileName in f:
        print("Generating confusion matrix graphic for ", fileName)
        saveConfMat(fileName)


def retrieveFeatures(dirToAnalyze="Data\\Pickles\\FeatureVectors\\4Graphlet"):
    print("Loading feature vectors for", dirToAnalyze)
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
    return (featureArray, labelArray)


def generateClassifier(featureArray, labelArray):
    print("Learning...")
    classifier = svm.SVC(kernel='linear', gamma=0.001)
    classifier.fit(featureArray[::2], labelArray[::2])
    return classifier


def pickleClassifier(classifier, pickleLocation="Data\\Pickles\\Classifiers\\4GAll.txt"):
    pickle.dump(classifier, open(pickleLocation, "wb"))
    print("Classifier serialized")


def analyze(dirToAnalyze="Data\\Pickles\\FeatureVectors\\4Graphlet", classifierLocation=""):
    featureArray, labelArray = retrieveFeatures(dirToAnalyze)
    #Interleaved testing and training
    if classifierLocation == "":
        classifier = generateClassifier(featureArray, labelArray)
    else: #test using generated classifier
        classifier = pickle.load(open(classifierLocation, "rb"))

    expected = labelArray[1::2]
    predicted = classifier.predict(featureArray[1::2])

    #for val in zip(expected, predicted):
    #    print(val)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)


#pickleClassifier(generateClassifier(*retrieveFeatures()))
#analyze("Data\\Pickles\\FeatureVectors\\4Graphlet\\Vision", "Data\\Pickles\\Classifiers\\4GAll.txt")
#generateCFGraphics()
