import numpy as np
import random
from gym import spaces
import cv2

class SnakeEnv(object):
    def __init__(self, env_parameters, renderID=0, renderWait=100, channel_first=False):
        self.initialLength = int(env_parameters['INITIAL_LENGTH'])
        self.gridSize = int(env_parameters['GRIDSIZE'])
        self.visionRadius = int(env_parameters['VISION_RADIUS'])
        self.appleReward = 1
        self.collisionReward = -1
        self.appleCount = int(env_parameters['APPLE_COUNT'])
        self.score = 0
        self.info = None
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.visionRadius*2 + 1, self.visionRadius*2 + 1, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.renderID = renderID
        self.renderWait = renderWait
        self.channel_first = channel_first

    def step(self, action):
        if self.done:
            return self.get_state(), 0,  self.done, self.info

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
            self.score+=1

        if self.checkCollision():
            reward = self.collisionReward

        if not self.done:
            self.updateState()

        return self.get_state(), reward, self.done, self.info
    
    def get_state(self):
        if self.channel_first:
            return np.moveaxis(self.vision, source=-1, destination=0)
        else:
            return self.vision

    def reset(self):
        self.length = self.initialLength
        self.head = [self.gridSize//2 + 1 + self.visionRadius, self.gridSize//2 + 1 + self.visionRadius]
        self.tail = [[self.gridSize//2 + 2 + self.visionRadius + i, self.gridSize//2 + 1 + self.visionRadius] for i in range(self.initialLength-1)]
        self.dir = 0
        self.score = 0

        self.apples = []
        for i in range(self.appleCount):
            self.apples.append(self.randomApple())

        self.done = False
        self.updateState()

        return self.get_state()

    def updateState(self):
        self.state = np.ones((self.gridSize + self.visionRadius*2, self.gridSize + self.visionRadius*2, 3), float)
        self.state[self.visionRadius:-self.visionRadius, self.visionRadius:-self.visionRadius, :] = [0, 0, 0]
        for apple in self.apples:
            self.state[apple[0]-1, apple[1]-1, :] = [0, 1, 0]
        self.state[self.head[0]-1, self.head[1]-1, :] = [0, 0, 1]
        for coor in self.tail:
            self.state[coor[0]-1, coor[1]-1, :] = [1, 0, 0]
        self.vision = self.state[self.head[0] - self.visionRadius-1:self.head[0]+self.visionRadius, self.head[1]-self.visionRadius-1:self.head[1]+self.visionRadius, :]

    def randomApple(self):
        apple = [random.randint(1+self.visionRadius, self.gridSize+self.visionRadius), random.randint(1+self.visionRadius, self.gridSize+self.visionRadius)]
        if apple == self.head:
            apple = self.randomApple()
        for coor in self.tail:
            if apple == coor:
                apple = self.randomApple()
        for coor in self.apples:
            if apple == coor:
                apple = self.randomApple()
        return apple

    def checkApple(self):
        for apple in self.apples:
            if self.head == apple:
                self.apples.remove(apple)
                self.apples.append(self.randomApple())
                return True
        else:
            return False

    def checkCollision(self):
        if self.head[0] < 1+self.visionRadius or self.head[1] < 1 + self.visionRadius or self.head[0] > self.gridSize+self.visionRadius or self.head[1] > self.gridSize+self.visionRadius:
            self.done = True
            return True

        for coor in self.tail:
            if self.head == coor:
                self.done = True
                return True

    def render(self, wait=False):
        image = cv2.resize(self.state, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.putText(image, f'Score:{self.score}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
        cv2.imshow(f"Worker: {self.renderID}", image)
        if wait:
            cv2.waitKey(self.renderWait)

    def renderVision(self):
        image = cv2.resize(self.vision, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow(f"POV Worker: {self.renderID}", image)
        cv2.waitKey(self.renderWait)
