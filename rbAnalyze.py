# This is a helper class to analyze data generated from rbGenerate.py.
# A large part of this program is very ad hoc serializing and deserializing.
# User beware

from rbGenerate import *
import numpy as np
import random
from os import walk
import re
import time

from sklearn import svm, metrics, cross_validation, preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Extracts features using the four graphlet method


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

# Extracts features based on message paths


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
        print("Extracting data from: ", fileName)
        motionGraph = pickle.load(open(fileName, "rb"))
        edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
        startIDs = [edge.startID for edge in edges]
        totalEdges += len(edges)
        degrees = [startIDs.count(x) for x in set(startIDs)]
        if len(degrees) > 0 and max(degrees) > maxDegree:
            maxDegree = max(degrees)
        edgeIDs = set([(edge.startID, edge.endID) for edge in edges])
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

#--------------------------------------------------------
# Serialization methods


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
    print(numSamples, end - start)


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
    print(k, end - start)


def adjMatPicklesToText():
    fileLocation = "Data\\Pickles\\AdjMatsRaw\\Graphs"
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
    # plt.show()
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


def showExampleGraph():
    contractData = (0, 0.5, 0.1, 0.1, 0.2)
    randomData = (0.7, 0.3, 0.4, 0.3,   0)
    resourceData = (0, 0.1, 0.5, 0.6, 0.8)
    disperseData = (0.3, 0.1,   0,   0,   0)

    width = 1.0
    ind = np.linspace(1, width * 5, num=5)
    colors = ("#ff7200", "#e80500", "#1917e8", "#cc00ff")

    fig = plt.figure(figsize=(7, 8))
    globalFont = 16

    p1 = plt.bar(ind, resourceData, width, color=colors[2], bottom=tuple(
        sum(t) for t in zip(randomData, disperseData, contractData)))
    p2 = plt.bar(ind, contractData, width, color=colors[0], bottom=tuple(
        sum(t) for t in zip(randomData, disperseData)))
    p3 = plt.bar(ind, disperseData, width, color=colors[1], bottom=randomData)
    p4 = plt.bar(ind, randomData, width, color=colors[3])

    plt.xlabel('Amount of Time (sec)')
    plt.ylabel('Fraction of Predictions of Each Behavior', fontsize=globalFont)
    plt.title("Accuracy for Random Motion vs. Time", fontsize=globalFont)
    plt.xticks(ind + width / 2., ("5-10", "10-15", "15-20",
                                  "20-25", "25-30"), fontsize=globalFont - 2)
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.tick_params(bottom="off", top="off")
    plt.xlim([0.5, 6.5 * width])
    plt.ylim([0, 1.05])

    #patches = [mpatches.Patch(color=col, label=key) for (col,key) in zip(colors,keys)]
    # plt.legend(handles=patches, loc=7, bbox_to_anchor=(1.225, 0.5),
    # fontsize=globalFont) #center right
    plt.legend((p2[0], p3[0], p1[0], p4[0]), ("Contract", "Dispersal",
                                              "Resource Collection", "Random Motion"), loc=7, bbox_to_anchor=(1.55, 0.5))

    # plt.show()
    plt.savefig(
        "Graphics\\4Graphlet\\All\\Comparisons\\StackExample.png", bbox_inches="tight")
    plt.clf()


def saveStatGraph(stats, title, fileName):
    keys = sorted(stats[0][0])

    predictedList = []
    for expected in stats:
        transpose = {key: [] for key in keys}
        for chunk in expected:
            for key in keys:
                transpose[key] += [chunk[key]]
        predictedList += [transpose]

    #plots = []
    width = 1.0
    ind = np.linspace(1, width * 5, num=5)
    # colors = ["#ff9f01", "#dae820", "#0cff65", "#1187e8", "#9423ff"]
    colors = ["#ff7200", "#e80500", "#cc00ff", "#1917e8", "#00bfff"]
    colorMap = dict(zip(keys, colors))
    fig = plt.figure(figsize=(24, 8))
    globalFont = 22

    for i in range(len(predictedList)):
        bottomHeights = [1.0] * 5
        for key in keys[i + 1:] + keys[:i + 1]:
            normalizedList = [val / 10.0 for val in predictedList[i][key]]
            curList = tuple(normalizedList)
            for j in range(len(curList)):
                bottomHeights[j] -= curList[j]
            plt.bar(ind + 6 * i * width, curList, width,
                    color=colorMap[key], bottom=tuple(bottomHeights))

    xpositions = np.linspace(3.5 * width, 27.5 * width, num=5)

    plt.ylabel('Fraction of Predictions', fontsize=globalFont)
    plt.title(title, fontsize=globalFont)
    xpositions = np.linspace(3.5 * width, 27.5 * width, num=5)
    plt.xticks(xpositions, tuple(keys), fontsize=globalFont - 2)
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.tick_params(bottom="off", top="off")
    plt.xlim([0, 31 * width])
    plt.ylim([0, 1.05])

    patches = [mpatches.Patch(color=col, label=key)
               for (col, key) in zip(colors, keys)]
    plt.legend(handles=patches, loc=7, bbox_to_anchor=(
        1.225, 0.5), fontsize=globalFont)  # center right

    dirToSave = fileName.replace("Results", "Graphics")
    fileName = os.path.basename(fileName)
    dirToSave = dirToSave.replace(
        fileName, "Comparisons\\With Legends\\" + fileName).replace(".txt", ".png")

    # plt.show()
    plt.savefig(dirToSave, bbox_inches="tight")
    plt.clf()


def getVariableStatArray(resFile):
    f = open(resFile)
    lines = []
    for line in f:
        regex = re.compile('[^a-zA-Z ]')
        lines.append(regex.sub('', line).split())

    # Partitioned file into expected values
    expectedList = [lines[i * 50:(i + 1) * 50] for i in range(5)]
    # partition expected values into ranges
    chunkedList = []
    for expected in expectedList:
        chunks = ([expected[i * 10:(i + 1) * 10] for i in range(5)])
        chunkedList += [chunks]

    statList = []
    for expected in chunkedList:
        expectedStat = []
        for chunk in expected:
            curStats = {"contract": 0, "disperse": 0,
                        "randomStep": 0, "resourceCollector": 0, "static": 0}
            for val in chunk:
                curStats[val[1]] += 1
            expectedStat += [curStats]
        statList += [expectedStat]

    return statList


def generateStackGraphs(rootDir="Results\\MessagePath\\All", preTitle="Path Length: "):
    #fileList = ["BlueRobotsEP.txt"]
    fileList = ["BlueRobotsEP.txt", "RedRobotsEP.txt",
                "BlueVisionEP.txt", "RedVisionEP.txt", "TimeEP.txt"]
    titleList = ["Varying Number of Observers (40 - 140)",
                 "Varying Number of Agents (40 - 140)",
                 "Varying Range of Observer Vision (0.15 - 0.25)",
                 "Varying Range of Agent Vision (0.6 - 0.16)",
                 "Varying Amount of Time (5 sec - 30 sec)"]
    titleList = [preTitle + val for val in titleList]

    argumentList = zip(fileList, titleList)
    for fileName, title in argumentList:
        stats = getVariableStatArray(rootDir + "\\" + fileName)
        print("Generating stack graph for ", fileName)
        saveStatGraph(stats, title, rootDir + "\\" + fileName)


def generateLineGraphs():
    # k-graphlet
    gxVals = [1, 5, 10, 25, 50, 100, 500,
              1000, 5000, 10000, 20000]  # k-graphlet
    gAcc = [0.33, 0.55, 0.60, 0.66, 0.71, 0.73,
            0.79, 0.81, 0.84, 0.84, 0.86]  # Accuracy
    gSVM = [0.08, 0.33, 0.71, 2.03, 5.12, 11.77,
            134.32, 199.90, 238.33, 246.89, 240.23]
    gFeats = [0.98, 1.22, 1.57, 2.55, 4.17, 7.40,
              33.26, 65.88, 331.97, 655.03, 1350.31]

    mxVals = list(range(1, 11))  # Message path
    mAcc = [0.73, 0.73, 0.79, 0.86, 0.87, 0.87, 0.86, 0.86, 0.86, 0.85]
    mSVM = [17.88, 2.59, 0.38, 0.15, 0.21, 0.14, 0.25, 0.29, 0.18, 0.18]
    mFeats = [90.51, 90.33, 90.37, 88.35, 88.20,
              89.04, 89.27, 87.50, 88.00, 88.40]

    plt.figure(1)

    plt.subplot(231)
    plt.plot(gxVals, gAcc)
    plt.title("4-Graphlet Accuracy")
    plt.xlabel("Number of Samples")
    plt.ylim([0, 1])
    plt.grid(True)

    plt.subplot(232)
    plt.plot(gxVals, gFeats)
    plt.title("4-Graphlet Time for Feature Vectors (sec)")
    plt.xlabel("Number of Samples")
    plt.grid(True)

    plt.subplot(233)
    plt.plot(gxVals, gSVM)
    plt.title("4-Graphlet Training Time (sec)")
    plt.xlabel("Number of Samples")
    plt.grid(True)

    plt.subplot(234)
    plt.plot(mxVals, mAcc)
    plt.title("Path Length Accuracy")
    plt.xlabel("Max Path Length")
    plt.ylim([0, 1])
    plt.grid(True)

    plt.subplot(235)
    plt.plot(mxVals, mFeats)
    plt.title("Path Length Time for Feature Vectors(sec)")
    plt.xlabel("Max Path Length")
    plt.ylim([0, 1400])
    plt.grid(True)

    plt.subplot(236)
    plt.plot(mxVals, mSVM)
    plt.title("Path Length Training Time (sec)")
    plt.xlabel("Max Path Length")
    plt.ylim([0, 250])
    plt.grid(True)

    plt.show()


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

    print("Classifying...")
    clf = svm.SVC(kernel='linear', gamma=0.001)
    scores = cross_validation.cross_val_score(
        clf, exampleArray, labelArray, cv=10)

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

    # for val in zip(expected, predicted):
    #    print(val)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)


def batchAnalyze():
    #dirsToAnalyze = [1, 5, 10, 25, 50, 100, 500, 1000, 5000, 10000, 20000]
    dirsToAnalyze = [6]  # list(range(1,11))
    for d in dirsToAnalyze:
        #serialize4GraphFeats("Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet" + str(d), d)
        serializePathFeats(
            "Data\\Pickles\\FeatureVectors\\MessagePaths\\MessagePath" + str(d), d)
        #analyze("Data\\Pickles\\FeatureVectors\\4Graphlets\\4Graphlet" + str(d))


# pickleClassifier(generateClassifier(*retrieveFeatures()))
# analyze("Data\\Pickles\\FeatureVectors\\MessagePath")
# analyze()
# generateCFGraphics()
