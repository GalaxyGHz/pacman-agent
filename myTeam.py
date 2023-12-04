# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint, Stack

from queue import PriorityQueue
from itertools import product

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ActionInfo():
    def __init__(self, prio, game_state, dist, previous=None, action=None):
        self.priority = prio
        self.game_state = game_state
        self.dist = dist
        self.previous = previous
        self.action = action

    def __lt__(self, other):
        return self.priority < other.priority

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.001):
        super().__init__(index, time_for_computing)
        self.start = None
        self.returning_home = False

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                next_game_state = self.get_next_game_state(game_state, action)
                pos2 = next_game_state.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
        
    def get_next_game_state(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        next_game_state = game_state.generate_successor(self.index, action)
        pos = next_game_state.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return next_game_state.generate_successor(self.index, action)
        else:
            return next_game_state

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_game_state = self.get_next_game_state(game_state, action)
        features['next_game_state_score'] = self.get_score(next_game_state)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'next_game_state_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def score(self, my_pos, target, walked):
        my_score = self.get_maze_distance(my_pos, target) + walked
        return my_score

    @staticmethod
    def attacker_Astar(agent, game_state, target):
        pq = PriorityQueue()
        visited = set()
        my_pos = game_state.get_agent_state(agent.index).get_position()
        pq.put(ActionInfo(agent.score(my_pos, target, 0), game_state, 0))
        
        while not pq.empty():
            pack = pq.get()
            current_game_state, distance_traveled = pack.game_state, pack.dist
            my_pos = current_game_state.get_agent_state(agent.index).get_position()
            if my_pos in visited:
                continue
            visited.add(my_pos)

            if my_pos == target:
                break

            actions = current_game_state.get_legal_actions(agent.index)

            for action in actions:
                next_game_state = current_game_state.generate_successor(agent.index, action)
                my_pos = next_game_state.get_agent_state(agent.index).get_position()

                if next_game_state.get_agent_state(agent.index).get_position() == agent.start:
                    continue

                new_score = agent.score(my_pos, target, distance_traveled + 1)
                new_state = ActionInfo(new_score, next_game_state, distance_traveled + 1)
                new_state.previous = pack
                new_state.action = action
                pq.put(new_state)

        while pack.previous.previous:
            pack = pack.previous
        return pack.action
    
    def get_edibles(self, my_pos, game_state):
        food_list = self.get_food(game_state).as_list()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invader_pos = [a.get_position() for a in enemies if a.is_pacman and a.get_position() is not None]
        if game_state.get_agent_state(self.index).scared_timer == 0:
            food_list += invader_pos

        food_distances = [(self.get_maze_distance(my_pos, food), food) for food in food_list]
        food_sorted = [f[1] for f in sorted(food_distances)]

        return food_sorted

    def get_home_pos(self, my_pos, game_state):
        walls = game_state.get_walls()
        if self.red:
            w = walls.width // 2 - 2
        else:
            w = walls.width // 2 + 1
        home_list = []
        for h in range(1, walls.height):
            if not walls[w][h]:
                home_list.append((w, h))

        home_distances = [(self.get_maze_distance(my_pos, home), home) for home in home_list]
        home_sorted = [h[1] for h in sorted(home_distances)]

        return home_sorted[:1]

    def get_ghosts(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared = [a for a in enemies if not a.is_pacman and a.scared_timer > 5 and a.get_position() is not None]
        not_scared = [a for a in enemies if not a.is_pacman and a.scared_timer <= 5 and a.get_position() is not None]
        return scared, not_scared

    def choose_target(self, game_state):
        food_left = len(self.get_food(game_state).as_list())
        carrying = game_state.get_agent_state(self.index).num_carrying
        my_pos = game_state.get_agent_state(self.index).get_position()
        f = self.get_edibles(my_pos, game_state)
        h = self.get_home_pos(my_pos, game_state)
        c = None if not self.get_capsules(game_state) else self.get_capsules(game_state)
        scared, not_scared = self.get_ghosts(game_state)
        # print(f, c, h)
        if food_left <= 2:
            return h
        if not carrying:
            self.returning_home = False
        if self.returning_home:
            return h
        
        if not scared and not not_scared:
            return f
        elif not_scared and c:
            return c
        elif not_scared:
            if carrying:
                self.returning_home = True
                return h
            else: 
                return f

        return f
    
    def is_hole(self, wall_map, pos):
        wall_cnt = 0        
        for dx, dy in product(range(-1, 2), range(-1, 2)):
            if wall_map[int(pos[0]) + dx][int(pos[1]) + dy]:
                wall_cnt += 1
        return wall_cnt >= 2 

    def nearest_open_space(self, game_state, pos):
        wall_map = game_state.get_walls()
        
        stack = Stack()
        stack.push(pos)
        
        vis = set()
        vis.add(pos)

        while not stack.isEmpty():
            curr_pos = stack.pop()
            if not self.is_hole(wall_map, curr_pos):
                return curr_pos
            
            for dx, dy in product(range(-1, 2), range(-1, 2)):
                new_pos = tuple([curr_pos[0] + dx, curr_pos[1] + dy])
                if not wall_map[int(new_pos[0])][int(new_pos[1])] and new_pos not in vis:
                    stack.push(new_pos)
                    vis.add(new_pos)


    def attacker_Astar_food(self, game_state, targets):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_pos = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
        my_pos = game_state.get_agent_state(self.index).get_position()

        for target in targets:
            open_space = self.nearest_open_space(game_state, target)
            ghost_dists = [self.get_maze_distance(open_space, ghost) for ghost in ghost_pos]
            if ghost_dists:
                ghost_dist = min(ghost_dists)

                my_dist = self.get_maze_distance(my_pos, open_space)
                margin = self.get_maze_distance(open_space, target)
                
                if my_dist + margin < ghost_dist:
                    # print("1")
                    return OffensiveReflexAgent.attacker_Astar(self, game_state, target)
                    
            else:
                # print("2")
                return OffensiveReflexAgent.attacker_Astar(self, game_state, targets[0])
        # print("3")
        # print(self.start)
        tmp = list(self.start)
        tmp[1] += 1
        tmp = tuple(tmp)
        return OffensiveReflexAgent.attacker_Astar(self, game_state, random.choice(targets)) #TODO: fix


    
    def choose_action(self, game_state):
        # start = time.time()
        walls = game_state.get_walls()
        if self.red:
            w = walls.width - 2
        else:
            w = 1
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos[0] == w:
            return "Stop"



        target = self.choose_target(game_state)

        if len(target) == 1:
            best_action = OffensiveReflexAgent.attacker_Astar(self, game_state, target[0])
            print("h")
        else:
            best_action = self.attacker_Astar_food(game_state, target)
            print("f")
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        next_game_state = self.get_next_game_state(game_state, action)

        my_state = next_game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_game_state.get_agent_state(i) for i in self.get_opponents(next_game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
