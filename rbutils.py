import random

#contains class information for individual robots
class Robot:
     def __init__(self, rid, x, y, maxSpeed = 0.005, vis = 0.15):
          self.rid = rid
          self.x = x
          self.y = y
          self.vis = vis
          self.maxSpeed = maxSpeed 
          self.nearSame = []
          self.nearOther = []
          self.goalList = []
          
     def addGoal(loc):
          self.goalList += [loc]
          
     def setGoal(loc):
          self.goalList = [loc]
          
     def clearGoals():
          self.goalList = []
          
     def randomizePosition(self):
          self.x = random.random()
          self.y = random.random()

#return distance between two robots
def dist(r1, r2):
     return ((r1.x - r2.x)**2 + (r1.y - r2.y)**2)**(1.0/2)

#returns list of "num" random robots     
def initRobots(num):
     rlist = []
     for i in range(num):
          rlist += [Robot(i, random.random(), random.random())]
     return rlist
     
#randomizes position of each robot in list  
def randomizeRobots(rlist):
     for rob in rlist:
          rob.randomizePosition()
          
#increases or decreases visibility     
def changeVis(rlist, d):
     newList = []
     for rob in rlist:
          rob.vis += d
          newList += [rob]
     return newList
     
#maintains list of nearest neighbors
def updateNearestNeighbors(blueRobots, redRobots):
     for blue in blueRobots:
          blue.nearSame = []
          for other in blueRobots:
               if other != blue and dist(blue, other) < blue.vis:
                    blue.nearSame += [other]
     for blue in blueRobots:
          blue.nearOther = []
          for other in redRobots:
               if dist(blue, other) < blue.vis:
                    blue.nearOther += [other]

#moves robots in rlist uniformly randomly within bounded box
def randomStep(rlist, xlo, xhi, ylo, yhi):
     for rob in rlist:
          dx = (2*random.random() - 1)
          dy = (2*random.random() - 1)
          dist = (2)**(1.0/2)
          dx /= dist
          dy /= dist
          rob.x += dx * rob.maxSpeed
          rob.y += dy * rob.maxSpeed
          
          if rob.x < xlo: rob.x = xlo
          if rob.x > xhi: rob.x = xhi
          if rob.y < ylo: rob.y = ylo
          if rob.y > yhi: rob.y = yhi
