from rbGenerate import *
import numpy as np
import random
from os import walk
import re
import time

from sklearn import svm, metrics, cross_validation, preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def fourGraphletFeatures(mat, numSamples=1000, multiCount=True):
    #print("Samples: ", numSamples)
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
        # for each possibility of four vertices
        for i in range(4):
            for j in range(4):
                numEdges = mat[sampVerts[i], sampVerts[j]]
                if numEdges > 0:
                    # initialize if not in dictionary
                    if frozenset([i, j]) not in edgeCounts:
                        edgeCounts[frozenset([i, j])] = 0
                    edgeCounts[frozenset([i, j])] += numEdges
                    degCounts[j] += 1
        # Add weight for the number of graphs in a multigraph
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
    # print(featureList)
    return featureList


def messagePathFeatures(mg, maxPathLength=10):
    # Each robot counts the number of times a path of length n has ended on it
    numPathList = [0 for val in range(maxPathLength + 1)]

    # for each observed robot
    for observedEdges in mg.graph:
        # each observer maintains a list of current sightings and paths
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
    # if sum(numPathList) > 0:
    #    numPathList = [val / sum(numPathList) for val in numPathList]
    return numPathList


def motionGraphToAdjMatrix(motionGraph, isDirected=False):
    n = motionGraph.numVertices
    adjMatrix = np.zeros((n, n), dtype=int)
    edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
    for edge in edges:
        adjMatrix[edge.startID, edge.endID] += 1
        if not isDirected:  # ensure symmetry
            adjMatrix[edge.endID, edge.startID] += 1
    return adjMatrix


def showGraphStats():
    sourceDir = "Data\\Pickles\\MotionGraphs"
    f = getFilesInDirectory(sourceDir)

    totalEdges = 0
    maxDegree = 0
    totalSparsity = 0
    for fileName in f:
        print("Extracting data from: ",fileName)
        motionGraph = pickle.load(open(fileName, "rb"))
        edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
        startIDs = [edge.startID for edge in edges]
        totalEdges += len(edges)
        degrees = [startIDs.count(x) for x in set(startIDs)]
        if len(degrees) > 0 and max(degrees) > maxDegree:
            maxDegree = max(degrees)
        edgeIDs = set([(edge.startID,edge.endID) for edge in edges])
        sparsity = len(edgeIDs) / (motionGraph.numVertices**2)
        totalSparsity += sparsity
        print("Edges: ", len(edges))
        print("Sparsity: ", sparsity)
    print("Total: ", totalEdges)
    print("Max degree: ", maxDegree)
    print("Average edges ", totalEdges / len(f))
    print("Average sparsity ", totalSparsity / len(f))


def getFilesInDirectory(rootDir):
    f = []
    for(path, dirs, files) in os.walk(rootDir):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            f.append(fullName)
    return f


def serializeGraphs():
    f = getFilesInDirectory("Data\\Graphs")
    for fileName in f:
        print("Generating motion graph for file: ", fileName)
        mg = genGraph(fileName)
        pickleFile = fileName.replace("Graphs", "Pickles")
        pickleFile = os.path.splitext(pickleFile)[0] + ".p"
        pickle.dump(mg, open(pickleFile, "wb"))


def serializeAdjMats(dirToPopulate="Data\\Pickles\\AdjMats\\Directed"):
    sourceDir = "Data\\Pickles\\MotionGraphs"
    f = getFilesInDirectory(sourceDir)

    for fileName in f:
        print("Generating pickle for file: ", fileName)
        mg = pickle.load(open(fileName, "rb"))
        adjMat = motionGraphToAdjMatrix(mg, True)

        pickleLocation = fileName.replace(sourceDir, dirToPopulate)
        pickle.dump((mg.label, adjMat), open(pickleLocation, "wb"))


def serialize4GraphFeats(dirToPopulate="Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet20000", numSamples=1000):
    sourceDir = "Data\\Pickles\\AdjMats\\Undirected"
    f = getFilesInDirectory(sourceDir)
    start = time.time()
    for fileName in f:
        #print("Generating feature vector for file: ", fileName)
        adjMat = pickle.load(open(fileName, "rb"))
        featureVector = fourGraphletFeatures(adjMat[1], numSamples)

        featVecLocation = fileName.replace(sourceDir, dirToPopulate)
        #pickle.dump((featureVector, adjMat[0]), open(featVecLocation, "wb"))
    end = time.time()
    print(numSamples, end-start)


def serializePathFeats(dirToPopulate="Data\\Pickles\\FeatureVectors\\MessagePaths\\MessagePath1", k=1):
    sourceDir = "Data\\Pickles\\MotionGraphs"
    f = getFilesInDirectory(sourceDir)

    start = time.time()
    for fileName in f:
        #print("Generating feature vector for file: ", fileName)
        mg = pickle.load(open(fileName, "rb"))
        featureVector = messagePathFeatures(mg, k)

        featVecLocation = fileName.replace(sourceDir, dirToPopulate)
        #pickle.dump((featureVector, mg.label), open(featVecLocation, "wb"))
    end = time.time()
    print(k, end-start)

def adjMatPicklesToText():
    fileLocation="Data\\Pickles\\AdjMatsRaw\\Graphs"
    sourceDir = "Data\\Pickles\\AdjMats\\Directed"
    f = getFilesInDirectory(sourceDir)

    curFileNumber = 1
    for fileName in f:
        print("Printing adjacency matrix for file: ", fileName)
        label, adjMat = pickle.load(open(fileName, "rb"))

        curFile = "graph" + str(curFileNumber) + ".txt"
        writeFile = open(fileLocation + "\\" + curFile, "wt")

        writeFile.write(label)
        writeFile.write("\n")

        for row in adjMat:
            for val in row:
                writeFile.write(str(val))
                writeFile.write(" ")
            writeFile.write("\n")
        curFileNumber += 1
    writeFile.close()

#----------------------------------------


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

    cmStart = labelStart + numLabels + 5  # magic number defined by file format
    cm = lines[cmStart:cmStart + numLabels]
    cm = [[int(val) for val in line.strip(" []\n").split()] for line in cm]
    cm = np.array(cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fileName = os.path.splitext(fileToAnalyze)[0]
    dirToSave = fileName.replace("Results", "Graphics") + ".png"
    fileName = os.path.basename(fileName)

    plt.figure()
    genConfMatGraphic(cm_normalized, labels, fileName)
    #plt.show()
    plt.savefig(dirToSave, bbox_inches="tight")
    plt.clf()


def generateCFGraphics(rootDir="Results\\MessagePath"):
    f = []
    for(path, dirs, files) in walk(rootDir):
        for fileName in files:
            fullName = os.path.join(path, fileName)
            if "EP.txt" not in fullName:
                f.append(fullName)
    for fileName in f:
        print("Generating confusion matrix graphic for ", fileName)
        saveConfMat(fileName)


def saveStatGraph(stats):
    keys = sorted(stats[0][0])

    predictedList = []
    for expected in stats:
        transpose = {key:[] for key in keys}
        for chunk in expected:
            for key in keys:
                transpose[key] += [chunk[key]]
        predictedList += [transpose]

    #plots = []
    width = 1.0
    ind = np.linspace(1,width*5,num=5)
    colors = ["#ff9f01", "#dae820", "#0cff65", "#1187e8", "#9423ff"]
    colorMap = dict(zip(keys, colors))
    fig = plt.figure(figsize=(16,8))


    for i in range(len(predictedList)):
        bottomHeights = [1.0] * 5
        for key in keys:
            normalizedList = [val / 10.0 for val in predictedList[i][key]]
            curList = tuple(normalizedList)
            for j in range(len(curList)):
                bottomHeights[j] -= curList[j]
            plt.bar(ind + 6*i*width, curList, width, color=colorMap[key], bottom=tuple(bottomHeights))


    plt.ylabel('Fraction of Guesses')
    plt.title('Varying Number of Blue Robots')
    xpositions = np.linspace(3.5*width,27.5*width,num=5)
    plt.xticks(xpositions, tuple(keys))
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.xlim([0,31*width])
    plt.ylim([0,1.05])

    #patches = [mpatches.Patch(color=col, label=key) for (col,key) in zip(colors,keys)]
    #plt.legend(handles=patches)

    plt.show()


def getVariableStatArray(resFile):
    f = open(resFile)
    lines = []
    for line in f:
        regex = re.compile('[^a-zA-Z ]')
        lines.append(regex.sub('', line).split())

    #Partitioned file into expected values
    expectedList = [lines[i*50:(i+1)*50] for i in range(5)]
    #partition expected values into ranges
    chunkedList = []
    for expected in expectedList:
        chunks = ([expected[i*10:(i+1)*10] for i in range(5)])
        chunkedList += [chunks]

    statList = []
    for expected in chunkedList:
        expectedStat = []
        for chunk in expected:
            curStats = {"contract":0, "disperse":0, "randomStep":0, "resourceCollector":0, "static":0}
            for val in chunk:
                curStats[val[1]] += 1
            expectedStat += [curStats]
        statList += [expectedStat]

    return statList


def generateStackGraphs(rootDir="Results\\MessagePath\\All"):
    fileList = ["BlueRobotsEP.txt"]
    #fileList = ["BlueRobotsEP.txt", "RedRobotsEP.txt", "BlueVisionEP.txt", "RedVisionEP.txt", "TimeEP.txt"]
    for fileName in fileList:
        stats = getVariableStatArray(rootDir + "\\" + fileName)
        print("Generating stack graph for ", fileName)
        saveStatGraph(stats)


def retrieveFeatures(dirToAnalyze="Data\\Pickles\\FeatureVectors\\MessagePath"):
    print("Loading feature vectors for", dirToAnalyze)
    f = getFilesInDirectory(dirToAnalyze)

    featureVectors = []
    labels = []
    for fileName in f:
        (featureVector, label) = pickle.load(open(fileName, "rb"))
        featureVectors += [featureVector]
        labels += [label]

    # apply svm
    featureArray = np.array(featureVectors)
    labelArray = np.array(labels)
    return (featureArray, labelArray)


def generateClassifier(featureArray, labelArray, myKernel='linear'):
    print("Learning...")
    start = time.time()
    classifier = svm.SVC(kernel=myKernel, gamma=0.001)
    classifier.fit(featureArray[::2], labelArray[::2])
    end = time.time()
    print("Elapsed time: ", end - start)
    return classifier


def pickleClassifier(classifier, pickleLocation="Data\\Pickles\\Classifiers\\MPAll.p"):
    pickle.dump(classifier, open(pickleLocation, "wb"))
    print("Classifier serialized")


def crossVal(dirToAnalyze="Data\\Pickles\\FeatureVectors\\4GraphletSmall", classifierLocation=""):
    exampleArray, labelArray = retrieveFeatures(dirToAnalyze)
    # Interleaved testing and training

    clf = svm.SVC(kernel='linear',gamma=0.001)
    scores = cross_validation.cross_val_score(clf,exampleArray,labelArray,cv=10)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def analyze(dirToAnalyze="Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet10000", classifierLocation=""):
    exampleArray, labelArray = retrieveFeatures(dirToAnalyze)

    # Interleaved testing and training
    if classifierLocation == "":
        print("Generating classifier")

        classifier = generateClassifier(exampleArray, labelArray)
        #pickleClassifier(classifier,"Data\\Pickles\\Classifiers\\" + dirToAnalyze.split("\\")[-1] + ".p")
    else:  # test using generated classifier
        classifier = pickle.load(open(classifierLocation, "rb"))

    expected = labelArray[1::2]
    predicted = classifier.predict(exampleArray[1::2])

    #for val in zip(expected, predicted):
    #    print(val)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)


def batchAnalyze():
    #dirsToAnalyze = [1, 5, 10, 25, 50, 100, 500, 1000, 5000, 10000, 20000]
    dirsToAnalyze = [6] #list(range(1,11))
    for d in dirsToAnalyze:
        #serialize4GraphFeats("Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet" + str(d), d)
        serializePathFeats("Data\\Pickles\\FeatureVectors\\MessagePaths\\MessagePath" + str(d), d)
        #analyze("Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet" + str(d))


# pickleClassifier(generateClassifier(*retrieveFeatures()))
# analyze("Data\\Pickles\\FeatureVectors\\MessagePath")
# analyze()
# generateCFGraphics()
