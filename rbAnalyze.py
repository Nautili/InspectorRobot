import pygame
from pygame.locals import *
from pygame import gfxdraw
import numpy as np
from os import walk
import math
import random

blueCount = 0
redCount = 0

class Edge:
     def __init__(self, startID, endID, point1, point2, time):
          self.startID = startID
          self.endID = endID
          self.startPoint = point1
          self.endPoint = point2
          self.time = time

class MotionGraphs:
    def __init__(self, numObserved, numObservers):
        self.numVertices = numObservers
        self.graph = [[] for i in range(numObserved)]

def genGraph(file):
     f = open(file, 'r')

     metaSplit = f.readline().strip().split(",")
     global blueCount
     global redCount
     blueCount = int(metaSplit[2])
     redCount = int(metaSplit[4])

     motionGraphs = MotionGraphs(redCount, blueCount)
     next(f) #flush blue

     #Assume blue robots remain in the same position
     global bluePositions
     bluePositions = [() for i in range(blueCount)]
     blueNeighbors = [[] for i in range(blueCount)]
     currentSightings = [[] for i in range(redCount)]

     for i in range(blueCount): #initialize blue and red positions
          blueIdentSplit = f.readline().strip().split(",")
          rid = int(blueIdentSplit[0])
          bx = blueIdentSplit[1]
          by = blueIdentSplit[2]
          bluePositions[i] = (float(bx), float(by))

          blueNeighborList = f.readline().strip().split(",")
          for neighbor in blueNeighborList[1:]:
               blueNeighbors[i] += [neighbor]

          redNeighborList = f.readline().strip().split(",")
          if int(redNeighborList[0]) > 0:
               for neighbor in redNeighborList[1:]: #initialize red queue
                    currentSightings[int(neighbor)] += [i]

     next(f) #flush red
     for i in range(redCount*3):
          next(f) #flush red data

     time = 1
     endCheck = f.readline()
     while(endCheck != ""):
          newSightings = [[] for i in range(redCount)]
          for blueRobot in range(blueCount):
               next(f) #skip blue robot data and neighbors
               next(f)
               newSightingsData = f.readline().strip().split(",")

               if newSightingsData[0] == '0':
                    newSightingsData = []
               else:
                    newSightingsData = list(map(int, newSightingsData))

               #update current red neighbors
               for redRobot in newSightingsData[1:]:
                    newSightings[redRobot] += [blueRobot] #this blue robot sees red robot
                    #update edge in motion graph
                    if blueRobot not in currentSightings[redRobot]: #if red robot is not seen by this blue robot, make new observation
                         for lastBlueSighting in currentSightings[redRobot]:
                              motionGraphs.graph[redRobot] += [Edge(lastBlueSighting, blueRobot, bluePositions[lastBlueSighting], bluePositions[blueRobot], time)]

          #add robots back in that are currently not seen
          for redRobot in range(redCount):
              if not newSightings[redRobot]: #if red robot not seen by anybody
                  newSightings[redRobot] = currentSightings[redRobot]

          currentSightings = newSightings #update to next time step
          next(f)
          for i in range(redCount*3): #skip through red data
               next(f)
          time += 1
          endCheck = f.readline()

     return motionGraphs

def displayGraph(motionGraphs, blueRobots, graphsVisible):
     pygame.init()
     screen = pygame.display.set_mode((600, 600),RESIZABLE)
     clock = pygame.time.Clock()

     red = (215, 40, 60)
     aqua = (0, 140, 255)
     egg = (225, 235, 215)
     black = (0, 0, 0)

     robRadius = 4
     lineWidth = 1
     arrowSize = 16
     arrowAngle = math.pi/10
     fps = 30

     while True:
          pressed_keys = pygame.key.get_pressed()

          # Event filtering

          for event in pygame.event.get():
               quit_attempt = False
               if event.type == pygame.QUIT:
                    quit_attempt = True
               elif event.type == pygame.KEYDOWN:
                    alt_pressed = pressed_keys[pygame.K_LALT] or \
                                  pressed_keys[pygame.K_RALT]
                    if event.key == pygame.K_ESCAPE:
                         quit_attempt = True
                    elif event.key == pygame.K_F4 and alt_pressed:
                         quit_attempt = True
               elif event.type==VIDEORESIZE:
                    screen=pygame.display.set_mode(event.dict['size'],RESIZABLE)
                    pygame.display.flip()

               if quit_attempt:
                    pygame.quit()

          screen.fill(egg)

          for graphNum in graphsVisible:
               for edge in motionGraphs.graph[graphNum]:
                    newStart = (edge.startPoint[0] * screen.get_width(), edge.startPoint[1] * screen.get_height())
                    newEnd = (edge.endPoint[0] * screen.get_width(), edge.endPoint[1] * screen.get_height())
                    pygame.draw.aaline(screen, black, newStart, newEnd, lineWidth)

                    edgeVec = tuple((x-y) for x,y in zip(newStart,newEnd))

                    mag = (edgeVec[0]**2 + edgeVec[1]**2)**(1/2.0)
                    arrowVec = tuple(map(lambda x:x/mag*arrowSize, edgeVec))
                    arrowPointVec = tuple(map(lambda x:x/arrowSize*robRadius, arrowVec))
                    arrowPointVec = vecAdd(newEnd, arrowPointVec)

                    arrowPoints = [arrowPointVec]
                    vecRight = rotate(arrowVec, arrowAngle)
                    vecLeft = rotate(arrowVec, -arrowAngle)

                    arrowPoints += [vecAdd(newEnd, vecRight)]
                    arrowPoints += [vecAdd(newEnd, vecLeft)]

                    pygame.gfxdraw.aapolygon(screen, arrowPoints, black)
                    pygame.gfxdraw.filled_polygon(screen, arrowPoints, black)

          for robot in blueRobots:
               drawx = int(robot[0] * screen.get_width())
               drawy = int(robot[1] * screen.get_height())
               pygame.gfxdraw.aacircle(screen, drawx, drawy, robRadius, aqua)
               pygame.gfxdraw.filled_circle(screen, drawx, drawy, robRadius, aqua)

          pygame.display.set_caption("Motion graph")

          pygame.display.flip()
          clock.tick(fps)

def rotate(vec, r):
     newx = vec[0] * math.cos(r) - vec[1] * math.sin(r)
     newy = vec[0] * math.sin(r) + vec[1] * math.cos(r)
     return (newx, newy)

def vecAdd(point, vec):
     return (point[0] + vec[0], point[1] + vec[1])

def motionGraphToAdjMatrix(motionGraph, isDirected = True):
    n = motionGraph.numVertices
    adjMatrix = np.zeros((n,n), dtype=int)
    edges = [edge for subgraph in motionGraph.graph for edge in subgraph]
    for edge in edges:
        adjMatrix[edge.startID,edge.endID] += 1
        if not isDirected: #ensure symmetry
            adjMatrix[edge.endID,edge.startID] += 1
    return adjMatrix

def fourGraphletFeatures(mat, numSamples=10000):
    numVertices = mat.shape[0]
    #ensure that every possible set is included
    #frozensets are hashable
    featureDict = {frozenset([0]):0,
                   frozenset([1]):0,
                   frozenset([2]):0,
                   frozenset([3]):0,
                   frozenset([0,1]):0,
                   frozenset([0,2]):0,
                   frozenset([1,2]):0,
                   frozenset([1,3]):0,
                   frozenset([2,3]):0,
                   frozenset([0,1,2]):0,
                   frozenset([1,2,3]):0}

    for sample in range(numSamples):
        degCounts = [0,0,0,0]
        sampVerts = random.sample(range(numVertices), 4)
        for i in range(4):
            for j in range(4):
                if mat[sampVerts[i],sampVerts[j]] > 0:
                    degCounts[j] += 1
        degSet = frozenset(degCounts)
        featureDict[degSet] += 1
    featureList = []
    for key in sorted(featureDict):
        featureList += [featureDict[key] / numSamples]
    print(np.array(featureList))
    return np.array(featureList)


#kernel takes two matrices and returns a kernel matrix
def fourGraphletKernel(mat1, mat2):
    kernelMat = np.zeros((mat1.shape[0], mat2.shape[0]), dtype=int)

#----------------------------------------

def analyze(draw=True):
    if not draw:
         f = []
         for(path, dirs, files) in walk("Data\\Graphs"):
              f.extend(files)
              break

         stats = []
         for file in f:
              #get label from filename
              label = file.split("_")[0].strip("0123456789")
              file = "Data\\Graphs\\" + file
              mg = genGraph(file)
              #generate feature vectors
              #form kernel matrix
              #apply svm
    else:

         mg = genGraph(r'Data\resourceCollector15_05_18-08_54_20.csv')
         adj = motionGraphToAdjMatrix(mg, False)
         fourGraphletFeatures(adj)
         displayGraph(mg, bluePositions, list(range(redCount)))
         #displayGraph(mg, bluePositions, [0])
