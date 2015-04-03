import random

def dist(r1, r2):
     return ((r1.x - r2.x)**2 + (r1.y - r2.y)**2)**(1.0/2)
     
def randomStep(rlist, xlo, xhi, ylo, yhi):
     for rob in rlist:
          dx = (2*random.random() - 1)
          dy = (2*random.random() - 1)
          dist = (dx**2 + dy**2)**(1.0/2)
          dx /= dist
          dy /= dist
          rob.x += dx * rob.maxSpeed
          rob.y += dy * rob.maxSpeed
          
          if rob.x < xlo: rob.x = xlo
          if rob.x > xhi: rob.x = xhi
          if rob.y < ylo: rob.y = ylo
          if rob.y > yhi: rob.y = yhi
