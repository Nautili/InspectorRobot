# Scene code from http://www.nerdparadise.com/tech/python/pygame/basics/part7/
# The first half is just boiler-plate stuff...

import pygame
import random
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
     screen = pygame.display.set_mode((width, height))
     clock = pygame.time.Clock()

     active_scene = starting_scene

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
            
               if quit_attempt:
                    active_scene.Terminate()
               else:
                    filtered_events.append(event)
        
          active_scene.ProcessInput(filtered_events, pressed_keys)
          active_scene.Update()
          active_scene.Render(screen)
             
          active_scene = active_scene.next
             
          pygame.display.flip()
          clock.tick(fps)

class RobotScene(SceneBase):
     def __init__(self, numRed, numBlue, blueVision, updater):
          SceneBase.__init__(self)
          
          self.showBlue = False
          self.showRed = False
          self.redRobots = rbutils.randRobots(numRed)
          self.blueRobots = rbutils.randRobots(numBlue)
          self.updater = updater
    
     def ProcessInput(self, events, pressed_keys):
          for event in events:
               if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.showRed = not self.showRed
               if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    self.showBlue = not self.showBlue
               if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.redRobots = rbutils.randRobots(len(self.redRobots))
                    self.blueRobots = rbutils.randRobots(len(self.blueRobots))
               if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.blueRobots = rbutils.changeVis(self.blueRobots, 0.01)
               if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    self.blueRobots = rbutils.changeVis(self.blueRobots, -0.01)
        
     def Update(self):
          for blue in self.blueRobots:
               blue.nearSame = []
               for other in self.blueRobots:
                    if other != blue and rbutils.dist(blue, other) < blue.vis:
                         blue.nearSame += [other]
          for blue in self.blueRobots:
               blue.nearOther = []
               for other in self.redRobots:
                    if rbutils.dist(blue, other) < blue.vis:
                         blue.nearOther += [other]
          self.updater(self.redRobots, 0, 1, 0, 1)
    
     def Render(self, screen):
          red = (215, 40, 60)
          aqua = (0, 140, 255)
          egg = (235, 235, 211)
          black = (0, 0, 0)
          
          screen.fill(egg)
          radius = 4
          lineWidth = 1

          for rr in self.redRobots:
               newx = int(rr.x * screen.get_width())
               newy = int(rr.y * screen.get_height())               
               pygame.draw.circle(screen, red, (newx, newy), radius)
          for br in self.blueRobots:
               newx = int(br.x * screen.get_width())
               newy = int(br.y * screen.get_height())
               
               if self.showBlue:
                    for nn in br.nearSame:
                         ox = int(nn.x * screen.get_width())
                         oy = int(nn.y * screen.get_height())
                         pygame.draw.line(screen, black, (newx, newy), (ox, oy), lineWidth)
               if self.showRed:
                    for nn in br.nearOther:
                         ox = int(nn.x * screen.get_width())
                         oy = int(nn.y * screen.get_height())
                         pygame.draw.line(screen, black, (newx, newy), (ox, oy), lineWidth)
               pygame.draw.circle(screen, aqua, (newx, newy), radius)

run_game(400, 300, 60, RobotScene(80, 80, 0.2, rbutils.randomStep))
