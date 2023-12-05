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
from itertools import product, count

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
    
def nearest_open_space(game_state, food):
    stack = Stack()
    stack.push(food)
    visited = set()
    visited.add(food)

    while not stack.isEmpty():
        curr_cell = stack.pop()
        if not surrounded_by_walls(game_state, curr_cell):
            return curr_cell
        
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            new_cell = tuple([curr_cell[0] + dx, curr_cell[1] + dy])
            if is_not_wall(game_state, new_cell) and new_cell not in visited:
                stack.push(new_cell)
                visited.add(new_cell)
    
def surrounded_by_walls(game_state, cell):
    wall_cnt = 0
    wall_map = game_state.get_walls()
    for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
        if wall_map[int(cell[0]) + dx][int(cell[1]) + dy]:
            wall_cnt += 1
    return wall_cnt > 1

def is_wall(game_state, cell):
    wall_map = game_state.get_walls()
    return wall_map[int(cell[0])][int(cell[1])]

def is_not_wall(game_state, cell):
    wall_map = game_state.get_walls()
    return not wall_map[int(cell[0])][int(cell[1])]

def arena_width(game_state):
    return game_state.get_walls().width

def arena_height(game_state):
    return game_state.get_walls().height

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.001):
        super().__init__(index, time_for_computing)
        self.start = None
        self.returning_home = False
        self.escape_deadlock = False

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
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
        
    def num_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying
    
    def not_scared(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer == 0
    
    def get_my_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()
    
    def get_food_positions(self, game_state):
        return self.get_food(game_state).as_list()
    
    def get_food_count(self, game_state):
        return len(self.get_food(game_state).as_list())
    
    def get_enemy_pacmen_positions(self, game_state):
        enemie_states = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [a.get_position() for a in enemie_states if a.is_pacman and a.get_position() is not None]
    
    def get_enemy_ghost_positions(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared_ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer > 5 and a.get_position() is not None]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer <= 5 and a.get_position() is not None]
        return ghosts, scared_ghosts
    
    def calculate_distances_to(self, game_state, positions):
        return [self.get_maze_distance(self.get_my_position(game_state), pos) for pos in positions]
    
    def calculate_distances_with_positions(self, game_state, positions):
        return [(self.get_maze_distance(self.get_my_position(game_state), pos), pos) for pos in positions]
    
    def sort_distances(self, distances):
        return [dist[1] for dist in sorted(distances)]
    
    def get_edibles(self, game_state):
        food_list = self.get_food_positions(game_state)

        # If you are closer to an enemy that you can eat than to any food, eat them
        pacmen_list = self.get_enemy_pacmen_positions(game_state)

        # If you are scared, ignore attacking enemies
        if self.not_scared(game_state):
            food_list += pacmen_list

        food = self.sort_distances(self.calculate_distances_with_positions(game_state, food_list))

        return food
    
    def get_edge_home_cells(self, game_state):
        w = arena_width(game_state) // 2 - 1 if self.red else arena_width(game_state) // 2
        home = []
        for h in range(1, arena_height(game_state) - 1):
            if is_not_wall(game_state, (w, h)):
                home.append((w, h))
        return home
    
    def get_closest_home_cell_position(self, game_state):
        home_list = self.get_edge_home_cells(game_state)
        home = self.sort_distances(self.calculate_distances_with_positions(game_state, home_list))
        return home[:1]
    
    def panic(self, game_state):
        my_pos = self.get_my_position(game_state)
        enemy_start = arena_width(game_state) - 2 if self.red else 1
        return my_pos[0] == enemy_start
    
    def score(self, my_pos, target, distance_traveled):
        my_score = self.get_maze_distance(my_pos, target) + distance_traveled
        return my_score

    def Astar(self, game_state, target):
        pq = PriorityQueue()
        visited = set()
        # Needed to resolve order in the priority queue if two states have the same score
        counter = count()
        self.add_to_prio_queue(pq, game_state, 0, None, None, target, counter)
        
        while not pq.empty():
            previous, current_game_state, distance_traveled = self.get_from_prio_queue(pq)
            my_pos = self.get_my_position(current_game_state)

            if my_pos in visited:
                continue
            visited.add(my_pos)
            if my_pos == target:
                break

            actions = current_game_state.get_legal_actions(self.index)
            for action in actions:
                next_game_state = self.get_next_game_state(current_game_state, action)
                my_pos = self.get_my_position(next_game_state)
                # If you get eaten, discard state
                if my_pos == self.start:
                    continue
                self.add_to_prio_queue(pq, next_game_state, distance_traveled + 1, previous, action, target, counter)

        return self.first_action(previous)
    
    def add_to_prio_queue(self, pq, game_state, distance_traveled, previous, action, target, counter):
        next_state = {
            "game_state" : game_state,
            "distance_traveled" : distance_traveled,
            "previous" : previous,
            "action" : action
        }
        pq.put((self.score(self.get_my_position(game_state), target, 0), next(counter), next_state))

    def get_from_prio_queue(self, pq):
        score, order, previous = pq.get()
        current_game_state, distance_traveled = previous["game_state"], previous["distance_traveled"]
        return previous, current_game_state, distance_traveled
    
    def first_action(self, previous):
        while previous["previous"]["previous"]:
            previous = previous["previous"]
        return previous["action"]
        
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    
    def choose_action(self, game_state):
        # start = time.time()
        # If you end up on the far side of the arena, give up
        if self.panic(game_state): return "Stop"

        targets = self.choose_targets(game_state)

        best_action = self.attacker_Astar_food(game_state, targets)
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action
    
    def choose_targets(self, game_state):
        food_left = self.get_food_count(game_state)
        carrying = self.num_carrying(game_state)

        f = self.get_edibles(game_state)
        h = self.get_closest_home_cell_position(game_state)
        c = self.get_capsules(game_state)
        
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        
        # If you have enough food to win, go home
        if food_left <= 2:
            return h
        
        # If you decided to play safe and go home, go home (if you returned all of the food you were carrying stop going home)
        if not carrying:
            self.returning_home = False
        if self.returning_home:
            return h
        
        # If you dont see anyone, go get food
        if not scared_ghosts and not ghosts:
            return f
        # If you see enemy ghosts and there is a capsule, go get the capsule
        elif ghosts and c:
            return c
        # If you see a ghost and there is no capsules
        elif ghosts:
            # If a ghost is near you and you have food, play safe and start going home
            if carrying and min(self.calculate_distances_to(game_state, ghosts)) < 5:
                self.returning_home = True
                return h
            # If you dont have food on you, you have nothing to lose so try getting food
            else: 
                return f

        return f
    
    def attacker_Astar_food(self, game_state, targets):
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        my_pos = self.get_my_position(game_state)
        ghost_dists = self.calculate_distances_to(game_state, ghosts)

        if len(targets) == 1 or not ghost_dists:
            return self.Astar(game_state, targets[0])
        elif ghost_dists:
            for target in targets:
                open_space = nearest_open_space(game_state, target)
                ghost_dist = min(ghost_dists)

                my_dist = self.get_maze_distance(my_pos, open_space)
                margin = self.get_maze_distance(open_space, target)
                
                if my_dist + margin*2 + 1 < ghost_dist:
                    print("1")
                    return self.Astar(game_state, target)

                    
            print("3")
            # print(self.start)
            tmp = list(self.start)
            tmp[1] += 1
            tmp = tuple(tmp)
            return self.Astar(game_state, tmp) #TODO: fix

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

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
    
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

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
