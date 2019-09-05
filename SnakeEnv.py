import numpy as np
import random
from gym import spaces
import cv2

class SnakeEnv(object):
    def __init__(self, gridSize=32, initialSnakeLength=4, render=False, renderID=0, renderWait=100):
        self.initialLength = initialSnakeLength
        self.gridSize = gridSize
        self.state = np.zeros((self.gridSize, self.gridSize), float)
        self.appleReward = 50
        self.collisionReward = -100
        self.appleCount = 15
        self.info = None
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.gridSize, self.gridSize))
        self.action_space = spaces.Discrete(3)
        self.render = render
        self.renderID = renderID
        self.renderWait = renderWait

    def step(self, action):
        if self.done:
            return self.state, 0,  self.done, self.info

        history = []
        history.append(self.head)
        history.extend(self.tail[:-1])
        tailend = self.tail[-1]
        if(action == 0):
            if(self.dir==0):
                head = [self.head[0]-1, self.head[1]]
            elif(self.dir==1):
                head = [self.head[0], self.head[1]+1]
            elif(self.dir==2):
                head = [self.head[0]+1, self.head[1]]
            elif(self.dir==3):
                head = [self.head[0], self.head[1]-1]
        
        elif(action == 1):
            if(self.dir==0):
                head = [self.head[0], self.head[1]+1]
                self.dir = 1
            elif(self.dir==1):
                head = [self.head[0]+1, self.head[1]]
                self.dir = 2
            elif(self.dir==2):
                head = [self.head[0], self.head[1]-1]
                self.dir=3
            elif(self.dir==3):
                head = [self.head[0]-1, self.head[1]]
                self.dir=0
        
        elif(action == 2):
            if(self.dir==0):
                head = [self.head[0], self.head[1]-1]
                self.dir=3
            elif(self.dir==1):
                head = [self.head[0]-1, self.head[1]]
                self.dir=0
            elif(self.dir==2):
                head = [self.head[0], self.head[1]+1]
                self.dir=1
            elif(self.dir==3):
                head = [self.head[0]+1, self.head[1]]
                self.dir=2

        self.tail = history
        self.head = head
        reward=0
        
        if self.checkApple():
            reward = self.appleReward
            self.tail.append(tailend)
            self.length+=1

        if self.checkCollision():
            reward = self.collisionReward

        if not self.done:
            self.updateState()

        if self.render:
            self.renderFrame()

        return self.state, reward, self.done, self.info
    
    def reset(self):
        self.length = self.initialLength
        self.head = [self.gridSize//2 + 1, self.gridSize//2 + 1]
        self.tail = [[self.gridSize//2 + 2 + i, self.gridSize//2 + 1] for i in range(self.initialLength-1)]
        self.dir = 0

        self.apples = []
        for i in range(self.appleCount):
            self.apples.append(self.randomApple())

        self.done = False
        self.updateState()

        return self.state

    def updateState(self):
        self.state = np.zeros((self.gridSize, self.gridSize), float)
        for apple in self.apples:
            self.state[apple[0]-1, apple[1]-1] = 1
        self.state[self.head[0]-1, self.head[1]-1] = 0.75
        for coor in self.tail:
            self.state[coor[0]-1, coor[1]-1] = 0.5

    def randomApple(self):
        apple = [random.randint(1, self.gridSize), random.randint(1, self.gridSize)]
        if apple == self.head:
            apple = self.randomApple()
        for coor in self.tail:
            if apple == coor:
                apple = self.randomApple()
        return apple

    def checkApple(self):
        for apple in self.apples:
            if self.head == apple:
                self.apples.remove(apple)
                newApple = self.randomApple()
                self.apples.append(newApple)
                return True
        else:
            return False

    def checkCollision(self):
        if self.head[0] < 1 or self.head[1] < 1 or self.head[0] > self.gridSize or self.head[1] > self.gridSize:
            self.done = True
            return True

        for coor in self.tail:
            if self.head == coor:
                self.done = True
                return True

    def renderFrame(self):
        image = cv2.resize(self.state, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow(f"Worker{self.renderID}", image)
        cv2.waitKey(self.renderWait)
