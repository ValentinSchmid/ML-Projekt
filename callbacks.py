import numpy as np
import random
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s
from settings import e

def setup(self):
    np.random.seed()
    
    arena = self.game_state['arena']
    accessible = []
    for d in arena:
        if (arena[d] != 1):
            accessible.append(d)
    # Q matrix
    Q = np.matrix(np.zeros([len(accessible),5]))  # [][0] = up ; [][1] = down ; [][2] = left ; [][3] = right ; [][4] = wait


def act(self):
    self.logger.info('Pick action at random')
    x, y, _, bombs_left, score = self.game_state['self']
    state = (x,y)
    
    epsilon = 0.7    
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT' , 'WAIT']
    shuffle(action_ideas)
        
    if np.random.rand(1) <= epsilon:
        self.next_action = action_ideas.pop()
    else:
        q_state = Q[(x,y)]
        act = np.argmax(q_state)
        if act == 0: action_ideas.append('UP')
        if act == 1: action_ideas.append('DOWN')
        if act == 2: action_ideas.append('LEFT')
        if act == 3: action_ideas.append('RIGHT')
        if act == 4: action_ideas.append('WAIT')
        
        self.next_action = action_ideas.pop()

def reward_update(self):
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    
    reward[state][self.next_action] = 0
    if self.events == e.MOVED_LEFT : reward[state][self.next_action] =  reward[state][self.next_action] - 1
    if self.events == e.MOVED_RIGHT : reward[state][self.next_action] =  reward[state][self.next_action] - 1
    if self.events == e.MOVED_UP : reward[state][self.next_action] =  reward[state][self.next_action] - 1
    if self.events == e.MOVED_DOWN : reward[state][self.next_action] =  reward[state][self.next_action] - 1
    if self.events == e.WAITED : reward[state][self.next_action] =  reward[state][self.next_action] - 5
    if self.events == e.COIN_COLLECTED : reward[state][self.next_action] =  reward[state][self.next_action] + 100
    if self.events == e.INVALID_ACTION : reward[state][self.next_action] =  reward[state][self.next_action] - 100
    print('State:',state,'Reward:',reward[state][self.next_action])

def end_of_episode(self):
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    alpha = 1
    gamma = 0.9
    
    for state in accessible:
        if self.next_action == 'UP': next_state = (x,y+1)
        if self.next_action == 'DOWN': next_state =  (x,y-1)
        if self.next_action == 'LEFT': next_state =  (x-1,y)
        if self.next_action == 'RIGHT': next_state = (x+1,y)
        if self.next_action == 'WAIT': next_state = (x,y)
        max_Q_next = np.argmax(Q[next_state])
        Q[state][self.next_action] = Q[state][self.next_action] + alpha * (reward[state][self.next_action] + gamma * max_Q_next - Q[state][self.next_action])