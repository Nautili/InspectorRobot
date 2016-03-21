# Scene code from http://www.nerdparadise.com/tech/python/pygame/basics/part7/
# The first half is just boiler-plate stuff...

import pygame
from pygame.locals import *
from pygame import gfxdraw
import random
import os
import datetime
import rbUtils

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
          self.redRobots = rbUtils.initRobots(numRed)
          self.blueRobots = rbUtils.initRobots(numBlue)
          rbUtils.setGoals(self.redRobots)
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
                         if os.path.exists("Data\\Graphs"):
                              self.isPrinting = not self.isPrinting
                              if self.isPrinting:
                                   updaterName = self.updater.__name__
                                   fname = updaterName + datetime.datetime.now().strftime('%y_%m_%d-%H_%M_%S')
                                   fname += '.csv'
                                   self.curFile = fname
                                   rbUtils.printStartState(fname, updaterName, self.blueRobots, self.redRobots)
                         else:
                              print("Directory \"Data\\Graphs\" is missing. Create a folder in this directory called \"Data\\Graphs\" to save results.")
                    if event.key == pygame.K_RETURN:
                         self.isPrinting = False
                         rbUtils.randomizeRobots(self.redRobots)
                         rbUtils.randomizeRobots(self.blueRobots)
                         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_UP:
                         self.blueRobots = rbUtils.changeVis(self.blueRobots, 0.01)
                         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_DOWN:
                         self.blueRobots = rbUtils.changeVis(self.blueRobots, -0.01)
                         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_RIGHT:
                         self.redRobots = rbUtils.changeVis(self.redRobots, 0.01)
                         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
                    if event.key == pygame.K_LEFT:
                         self.redRobots = rbUtils.changeVis(self.redRobots, -0.01)
                         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)

                    if event.key == pygame.K_1:
                         self.updater = rbUtils.randomStep
                         self.isPrinting = False
                         self.isGoal = False
                    if event.key == pygame.K_2:
                         self.updater = rbUtils.resourceCollector
                         self.isPrinting = False
                         self.isGoal = True
                    if event.key == pygame.K_3:
                         self.updater = rbUtils.disperse
                         self.isPrinting = False
                         self.isGoal = False
                    if event.key == pygame.K_4:
                         self.updater = rbUtils.static
                         self.isPrinting = False
                         self.isGoal = False

     def Update(self):
          rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
          self.updater(self.redRobots, 0, 1, 0, 1)
          if self.isPrinting:
               rbUtils.printState(self.curFile,self.blueRobots,self.redRobots)

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

class GeneratorScene(SceneBase):
     def __init__(self, numRed, numBlue, blueVision, updater, numSamples, isGoal = False):
          SceneBase.__init__(self)

          self.showBlue = False
          self.showRed = False
          self.isGoal = isGoal
          self.redRobots = rbUtils.initRobots(numRed)
          self.blueRobots = rbUtils.initRobots(numBlue)
          rbUtils.setGoals(self.redRobots)
          self.updater = updater
          self.numSamples = numSamples
          self.curFile = ''

     def ProcessInput(self, events, pressed_keys):
         pass #no-op

     def Update(self):
         rbUtils.updateNearestNeighbors(self.blueRobots, self.redRobots)
         self.updater(self.redRobots, 0, 1, 0, 1)
         rbUtils.printState(self.curFile,self.blueRobots,self.redRobots)

     def Render(self, screen):
         pass #no-op


#randomStep is run by default
def runRB():
    run_game(600, 600, 60, RobotScene(80, 80, 0.2, rbUtils.randomStep))

def generateData():

    run_game(600, 600, 60, GeneratorScene(80, 80, 0.2, rbUtils.randomStep))
