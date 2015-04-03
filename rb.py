# Scene code from http://www.nerdparadise.com/tech/python/pygame/basics/part7/
# The first half is just boiler-plate stuff...

import pygame
import random

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


class Robot:
     def __init__(self, x, y):
          self.x = x
          self.y = y
          self.nn = []

class RobotScene(SceneBase):
     def __init__(self, numRed, numBlue):
          SceneBase.__init__(self)
          self.redRobots = []
          for i in range(numRed):
               self.redRobots += [Robot(random.random(), random.random())]
          self.blueRobots = []
          for i in range(numBlue):
               self.blueRobots += [Robot(random.random(), random.random())]
          
    
     def ProcessInput(self, events, pressed_keys):
          pass
        
     def Update(self):
          pass
    
     def Render(self, screen):
          screen.fill((255, 255, 255))
          radius = 4
          red = (215, 40, 60)
          aqua = (0, 140, 255)

          for rr in self.redRobots:
               newx = int(rr.x * screen.get_width())
               newy = int(rr.y * screen.get_height())
               pygame.draw.circle(screen, red, (newx, newy), radius)
          for br in self.blueRobots:
               newx = int(br.x * screen.get_width())
               newy = int(br.y * screen.get_height())
               pygame.draw.circle(screen, aqua, (newx, newy), radius)

run_game(400, 300, 60, RobotScene(50, 50))
