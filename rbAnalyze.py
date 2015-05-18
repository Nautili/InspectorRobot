import pygame
from pygame.locals import *
from pygame import gfxdraw
from os import walk
import math

blueCount = 0
redCount = 0

class Edge:
     def __init__(self, startID, endID, point1, point2, time):
          self.startID = startID
          self.endID = endID
          self.startPoint = point1
          self.endPoint = point2
          self.time = time

def genGraph(file):
     f = open(file, 'r')
     
     metaSplit = f.readline().strip().split(",")
     global blueCount
     global redCount
     blueCount = int(metaSplit[2])
     redCount = int(metaSplit[4])
     
     motionGraphs = [[] for i in range(redCount)]
     next(f) #flush blue
     
     #Assume blue robots remain in the same position
     global bluePositions
     bluePositions = [() for i in range(blueCount)]
     blueNeighbors = [[] for i in range(blueCount)]
     currentSightings = [[] for i in range(redCount)]
     
     for i in range(blueCount):
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
               next(f)
               next(f)
               newSightingsData = f.readline().strip().split(",")
               
               if newSightingsData[0] == '0':
                    newSightingsData = []
               else:
                    newSightingsData = list(map(int, newSightingsData))
               
               #update current red neighbors
               for redRobot in newSightingsData[1:]:
                    newSightings[redRobot] += [blueRobot] #blue robot sees red robot
                    #update edge in motion graph
                    if blueRobot not in currentSightings[redRobot]: #make new observation
                         for lastBlueSighting in currentSightings[redRobot]:
                              motionGraphs[redRobot] += [Edge(lastBlueSighting, blueRobot, bluePositions[lastBlueSighting], bluePositions[blueRobot], time)]
         
          currentSightings = newSightings #update to next time step
          next(f)
          for i in range(redCount*3):
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
               for edge in motionGraphs[graphNum]:
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
          
def getDegreeStats(motionGraphs):
     edges = [edge for subgraph in motionGraphs for edge in subgraph]
     outDegree = [0 for i in range(blueCount)]
     inDegree = [0 for i in range(blueCount)]
     
     maxTime = 1
     times = [edge.time for edge in edges]
     if len(times) > 0:
          maxTime = max(times)
     
     for edge in edges:
          outDegree[edge.startID] += 1
          inDegree[edge.endID] += 1
     
     totalDegree = sum(outDegree)
     avgDegree = totalDegree / len(outDegree)
     
     varOut = 0
     minOut = float("inf")
     maxOut = 0
     for degree in outDegree:
          varOut += (degree - avgDegree)**2
          if degree < minOut:
               minOut = degree
          if degree > maxOut:
               maxOut = degree
     varOut /= len(outDegree)
     
     varIn = 0
     minIn = float("inf")
     maxIn = 0
     for degree in inDegree:
          varIn += (degree - avgDegree)**2
          if degree < minIn:
               minIn = degree
          if degree > maxIn:
               maxIn = degree
     varIn /= len(inDegree)
     
     retStats = (totalDegree, avgDegree, minOut, minIn, maxOut, maxIn, varOut, varIn)
     return tuple(map(lambda x:x/maxTime, retStats))
     
def classify(stats, scaleFactor):
     if stats[0] < 5:
          return "static"
     elif stats[6] / scaleFactor > 0.6:
          return "resource collection"
     elif stats[7] / scaleFactor < 0.3:
          return "dispersion"
     else:
          return "random"
          
#----------------------------------------
draw = True

if not draw:
     f = []
     for(path, dirs, files) in walk("Data\\Training"):
          f.extend(files)
          break

     stats = []
     for file in f:
          file = "Data\\Training\\" + file
          mg = genGraph(file)
          #print(file)
          newStat = getDegreeStats(mg)
          stats += [newStat]

     agg = [stats[i:i+10] for i in range(0,len(stats),10)]

     groupStats = []
     for group in agg:
          groupStats += [tuple([sum(y) / len(y) for y in zip(*group)])]
          
     scaleFactor = max([max(tup) for tup in groupStats])
     if scaleFactor == 0:
          scaleFactor = 1

     for stat in groupStats: 
          print("(", end = "")
          for val in stat:
               print("%0.2f" % val, end = ", ")
          print(")")
          print (stat[6] / scaleFactor, stat[7] / scaleFactor)

     for stat in stats:
          print(classify(stat, scaleFactor))
else:
     
     mg = genGraph(r'Data\disperse15_05_18-08_56_33.csv')
     displayGraph(mg, bluePositions, list(range(redCount)))
     #displayGraph(mg, bluePositions, [1])
