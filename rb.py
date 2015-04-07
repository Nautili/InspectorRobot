# Scene code from http://www.nerdparadise.com/tech/python/pygame/basics/part7/
# The first half is just boiler-plate stuff...

import pygame
from pygame.locals import *
from pygame import gfxdraw
import random
import datetime
import rbutils

class SceneBase:
     def __init__(self):
          self.next = self
    
     def ProcessInput(self, events, pressed_keys):
          print("uh-oh, you didn't override this in the child class")

     def Update(self):
          print("uh-oh, you didn't override this in the child class")

     def Render(self, screen):
          print("uh-oh, you didn't override this in the child class")

     def SwitchToScene(self, next_scene):
          self.next = next_scene
    
     def Terminate(self):
          self.SwitchToScene(None)

def run_game(width, height, fps, starting_scene):
     pygame.init()
     screen = pygame.display.set_mode((width, height),RESIZABLE)
     clock = pygame.time.Clock()

     active_scene = starting_scene
     paused = False

     while active_scene != None:
          pressed_keys = pygame.key.get_pressed()
        
          # Event filtering
          filtered_events = []
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
                    if event.key == pygame.K_SPACE:
                         paused = not paused
               elif event.type==VIDEORESIZE:
                    screen=pygame.display.set_mode(event.dict['size'],RESIZABLE)
                    pygame.display.flip()
            
               if quit_attempt:
                    active_scene.Terminate()
               else:
                    filtered_events.append(event)
        
          active_scene.ProcessInput(filtered_events, pressed_keys)
          if not paused:
               active_scene.Update()
          active_scene.Render(screen)
             
          active_scene = active_scene.next
             
          pygame.display.flip()
          clock.tick(fps)

class RobotScene(SceneBase):
     def __init__(self, numRed, numBlue, blueVision, updater, isGoal = False):
          SceneBase.__init__(self)
          
          self.showBlue = False
          self.showRed = False
          self.isGoal = isGoal
          self.redRobots = rbutils.initRobots(numRed)
          self.blueRobots = rbutils.initRobots(numBlue)
          rbutils.setGoals(self.redRobots)
          self.updater = updater
          self.isPrinting = False
          self.curFile = ''
    
     def ProcessInput(self, events, pressed_keys):
          for event in events:
               if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                         self.showRed = not self.showRed
                    if event.key == pygame.K_b:
                         self.showBlue = not self.showBlue
                    if event.key == pygame.K_o:
                         self.isPrinting = not self.isPrinting
                         if self.isPrinting:
                              updaterName = self.updater.__name__
                              fname = updaterName + datetime.datetime.now().strftime('%y_%m_%d-%H_%M_%S')
                              fname += '.csv'
                              self.curFile = fname
                              rbutils.printStartState(fname, updaterName, self.redRobots, self.blueRobots)
                    if event.key == pygame.K_RETURN:
                         rbutils.randomizeRobots(self.redRobots)
                         rbutils.randomizeRobots(self.blueRobots)
                         rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_UP:
                         self.blueRobots = rbutils.changeVis(self.blueRobots, 0.01)
                         rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_DOWN:
                         self.blueRobots = rbutils.changeVis(self.blueRobots, -0.01)
                         rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_RIGHT:
                         self.redRobots = rbutils.changeVis(self.redRobots, 0.01)
                         rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_LEFT:
                         self.redRobots = rbutils.changeVis(self.redRobots, -0.01)
                         rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
               
                    if event.key == pygame.K_1:
                         self.updater = rbutils.randomStep
                         self.isPrinting = False
                         self.isGoal = False
                    if event.key == pygame.K_2:
                         self.updater = rbutils.resourceCollector
                         self.isPrinting = False
                         self.isGoal = True
                    if event.key == pygame.K_3:
                         self.updater = rbutils.disperse
                         self.isPrinting = False
                         self.isGoal = False
        
     def Update(self):
          rbutils.updateNearestNeighbors(self.blueRobots, self.redRobots)
          self.updater(self.redRobots, 0, 1, 0, 1)
          if self.isPrinting:
               rbutils.printState(self.curFile,self.blueRobots,self.redRobots)
    
     def Render(self, screen):
          red = (215, 40, 60)
          aqua = (0, 140, 255)
          mint = (120, 210, 170)
          egg = (225, 235, 215)
          black = (0, 0, 0)
          
          robRadius = 4
          goalRadius = 6
          lineWidth = 1
          
          screen.fill(egg)

          #draw lines
          for br in self.blueRobots:
               newx = int(br.x * screen.get_width())
               newy = int(br.y * screen.get_height())    
               if self.showBlue:
                    for nn in br.nearSame:
                         ox = int(nn.x * screen.get_width())
                         oy = int(nn.y * screen.get_height())
                         pygame.draw.aaline(screen, black, (newx, newy), (ox, oy), lineWidth)
               if self.showRed:
                    for nn in br.nearOther:
                         ox = int(nn.x * screen.get_width())
                         oy = int(nn.y * screen.get_height())
                         pygame.draw.aaline(screen, black, (newx, newy), (ox, oy), lineWidth)
          #draw goals
          #assumes goals are same for all robots
          if self.isGoal:
               for g in self.redRobots[0].getGoalList():
                    newx = int(g.x * screen.get_width())
                    newy = int(g.y * screen.get_height())       
                    pygame.gfxdraw.aacircle(screen, newx, newy, goalRadius, mint)               
                    pygame.gfxdraw.filled_circle(screen, newx, newy, goalRadius, mint)
                               
          #draw robots
          for br in self.blueRobots:               
               newx = int(br.x * screen.get_width())
               newy = int(br.y * screen.get_height())          
               pygame.gfxdraw.aacircle(screen, newx, newy, robRadius, aqua)               
               pygame.gfxdraw.filled_circle(screen, newx, newy, robRadius, aqua)
               
          for rr in self.redRobots:
               newx = int(rr.x * screen.get_width())
               newy = int(rr.y * screen.get_height())       
               pygame.gfxdraw.aacircle(screen, newx, newy, robRadius, red)               
               pygame.gfxdraw.filled_circle(screen, newx, newy, robRadius, red)
          
          caption = 'Inspectors'
          if self.isPrinting:
               caption += ' (Printing to file)'
          pygame.display.set_caption(caption)
          
          

run_game(600, 600, 60, RobotScene(80, 80, 0.2, rbutils.randomStep))
#run_game(600, 600, 60, RobotScene(80, 80, 0.2, rbutils.resourceCollector, True))
#run_game(600, 600, 60, RobotScene(80, 80, 0.2, rbutils.disperse))
