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
from contest.util import nearestPoint, Stack, Queue

from queue import PriorityQueue
from itertools import product, count

# Constants
STATE_HOME = 1
STATE_CAPSULE = 2
STATE_ATTACK = 3
STATE_DEFEND = 4
STATE_INVESTIGATE = 5
STATE_PANIC = 42

state_to_str = { 1: 'home', 2: 'capsule', 3: 'attack', 4: 'defend', 5: 'investigate', 42: 'panic' }

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
        self.state = None
        self.my_food = set()
        self.last_food_eaten = set()
        self.missing_food_cell = [0, 0]
        self.last_visited_cells = Queue()

        # add something to make Queue non empty
        for i in range(10):
            self.last_visited_cells.push([i, i + 1])

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
        
    def in_lead(self, game_state):
        return game_state.data.score if self.red else -game_state.data.score
        
    def num_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying
    
    def not_scared(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer == 0
    
    def scared(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer != 0
    
    def get_my_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()
    
    def get_food_positions(self, game_state):
        return self.get_food(game_state).as_list()
    
    def get_food_count(self, game_state):
        return len(self.get_food(game_state).as_list())
    
    def update_my_food_status(self, game_state):
        my_food_new = game_state.get_red_food() if self.red else game_state.get_blue_food()
        my_food_new = set(my_food_new.as_list())
        self.last_food_eaten = self.my_food.difference(my_food_new)
        self.my_food = my_food_new

    def get_enemy_pacmen_positions(self, game_state):
        enemy_states = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [a.get_position() for a in enemy_states if a.is_pacman and a.get_position() is not None]
    
    def get_enemy_ghost_positions(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared_ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer > 5 and a.get_position() is not None]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer <= 5 and a.get_position() is not None]
        return ghosts, scared_ghosts
    
    def get_closest_ghost(self, game_state):
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        ghosts = self.sort_distances(self.calculate_distances_with_positions(game_state, ghosts))
        return None if not ghosts else ghosts[0]

    def get_scared_time(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer

    def no_different_cells_visited(self):
        return len(set(self.last_visited_cells.list))

    def update_last_visited(self, game_state):
        self.last_visited_cells.pop()
        self.last_visited_cells.push(self.get_my_position(game_state))

    def calculate_distances_to(self, game_state, positions):
        return [self.get_maze_distance(self.get_my_position(game_state), pos) for pos in positions]
    
    def calculate_distances_with_positions(self, game_state, positions):
        return [(self.get_maze_distance(self.get_my_position(game_state), pos), pos) for pos in positions]
    
    def sort_distances(self, distances):
        return [dist[1] for dist in sorted(distances)]
    
    def is_pacman(self, game_state):
        state = game_state.get_agent_state(self.index)
        return state.is_pacman

    def get_edibles(self, game_state):
        food_list = self.get_food_positions(game_state)

        # If you are closer to an enemy that you can eat than to any food, eat them
        pacmen_list = []

        # If you are scared, ignore attacking enemies
        if self.not_scared(game_state):
            food_list += pacmen_list

        # Sort food positions according to distances
        food = self.sort_distances(self.calculate_distances_with_positions(game_state, food_list))

        return food
    
    def get_edge_home_cells(self, game_state, enemy_home=False):
        if enemy_home:
            w = arena_width(game_state) // 2 - (0 if self.red else 1)
        else:
            w = arena_width(game_state) // 2 - (1 if self.red else 0)
        home = []
        for h in range(1, arena_height(game_state) - 1):
            if is_not_wall(game_state, (w, h)):
                home.append((w, h))
        return home

    def get_furthest_escape_position(self, game_state):
        home_list = self.get_edge_home_cells(game_state)
        home = self.sort_distances(self.calculate_distances_with_positions(game_state, home_list))
        return home[-1]

    def get_closest_home_cell_position(self, game_state):
        home_list = self.get_edge_home_cells(game_state)
        home = self.sort_distances(self.calculate_distances_with_positions(game_state, home_list))
        return home
    
    def near_center(self, game_state, my_x):
        w = arena_width(game_state)
        if self.red:
            x_left, x_right = w // 2 - 5, w // 2 - 1
        else:
            x_left, x_right = w // 2, w // 2 + 4 
        return x_left <= my_x <= x_right
        
    def panic(self, game_state):
        # Give up if you find out that you are trapped
        queue = Queue()
        queue.push((game_state, 0))

        visited = set()
        visited.add(game_state)

        while not queue.isEmpty():
            state, dist = queue.pop()
            # We are not trapped
            if dist > 5:
                return False

            for action in state.get_legal_actions(self.index):
                next_state = self.get_next_game_state(state, action)
                if next_state not in visited:
                    queue.push((next_state, dist + 1))
                    visited.add(next_state)

        print('we are about to panic!')

        # We are trapped time to panic
        return True
    
    def score(self, my_pos, target, distance_traveled):
        my_score = self.get_maze_distance(my_pos, target) + distance_traveled
        return my_score

    def Astar(self, game_state, target, excluded_positions=None):
        pq = PriorityQueue()
        # Needed to resolve order in the priority queue if two states have the same score
        counter = count()
        self.add_to_prio_queue(pq, game_state, 0, None, None, target, counter)
        print(target, self.get_my_position(game_state))
        visited = set()
        visited.add(self.start)

        if excluded_positions:
            # Positions that we don't want to visit are treated as visited
            visited.update(excluded_positions)

        # Edge case - we are already at target
        if self.get_my_position(game_state) == target:
            return "Stop"

        while not pq.empty():
            previous, current_game_state, distance_traveled = self.get_from_prio_queue(pq)
            my_pos = self.get_my_position(current_game_state)

            if my_pos == target:
                break

            actions = current_game_state.get_legal_actions(self.index)
            
            for action in actions:
                next_game_state = self.get_next_game_state(current_game_state, action)
                my_pos = self.get_my_position(next_game_state)
                
                if my_pos in visited:
                    continue
                
                # Check if next_game_state is next to ghost
                closest_ghost = self.get_closest_ghost(game_state)
                if closest_ghost and self.get_maze_distance(my_pos, closest_ghost) < 2:
                    continue

                visited.add(my_pos)
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
        while previous["previous"] and previous["previous"]["previous"]:
            previous = previous["previous"]
        return previous["action"]

    def closest_reachable_food(self, game_state, foods):
        foods = self.sort_distances(self.calculate_distances_with_positions(game_state, foods))
        my_pos = self.get_my_position(game_state)

        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        ghost_dists = self.calculate_distances_to(game_state, ghosts)
        ghost_dist = min(ghost_dists) if ghost_dists else float("inf")

        for food in foods:
            open_space = nearest_open_space(game_state, food)
            my_dist = self.get_maze_distance(my_pos, open_space)
            margin = self.get_maze_distance(open_space, food)
            
            if my_dist + margin * 2 + 1 < ghost_dist:
                return food
        
        return None

    def print_state(self, is_attacker):
        print(f'{"attacker" if is_attacker else "defender"}: {state_to_str[self.state]}')

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Cool (very cool) Attacker
    """
    
    def choose_action(self, game_state):
        # start = time.time()

        self.update_last_visited(game_state)
        target, exclude = self.choose_targets(game_state)

        self.print_state(True)

        best_action = self.Astar(game_state, target, exclude)

        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action
    
    def choose_targets(self, game_state):
        food_left = self.get_food_count(game_state)
        carrying = self.num_carrying(game_state)

        f = self.get_edibles(game_state)
        h = self.get_closest_home_cell_position(game_state)
        c = self.get_capsules(game_state)
        p = self.get_enemy_pacmen_positions(game_state)
        
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        closest_reachable_food = self.closest_reachable_food(game_state, f)

        my_pos = self.get_my_position(game_state)

        # If you end up being trapped, give up
        if self.panic(game_state): 
            self.state = STATE_PANIC
            return my_pos, []

        # If you have enough food to win, go home
        if food_left <= 2:
            self.state = STATE_HOME
            return h[0], []

        # If we carry food and time is running out then it's better to go home 
        if carrying and game_state.data.timeleft < 100:
            self.state = STATE_HOME
            return h[0], []

        # If we are stuck at getting home, try different route
        if self.state == STATE_HOME and self.no_different_cells_visited() < 5:
            return random.choice(h), []

        # If you decided to play safe and go home, go home (if you returned all of the food you were carrying stop going home)
        if self.state == STATE_HOME and carrying:
            return h[0], []
                
        # If you dont see anyone, go get the closest food
        if not scared_ghosts and not ghosts:
            self.state = STATE_ATTACK
            return f[0], []

        if self.no_different_cells_visited() < 5:
            self.state = STATE_HOME
            return random.choice(h), []
        
        # If you see enemy ghosts and there is a capsule, go get the capsule
        if ghosts and c:
            self.state = STATE_CAPSULE
            return c[0], []

        # If you see a ghost nearby and you have food, play safe and go home 
        if ghosts and carrying and min(self.calculate_distances_to(game_state, ghosts)) < 3:
            self.state = STATE_HOME
            return h[0], []

        # If you see ghost check if there is any reachable food
        if closest_reachable_food:
            self.state = STATE_ATTACK
            return closest_reachable_food, []

        # If none of the above conditions happened then go home
        self.state = STATE_HOME
        return random.choice(f), []

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Cool (very cool) Defender
    """

    def choose_action(self, game_state):        
        # start = time.time()
    
        self.update_my_food_status(game_state)
        target, exclude = self.choose_targets(game_state)

        self.print_state(False)

        best_action = self.Astar(game_state, target, exclude)

        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action
    
    def choose_targets(self, game_state):
        food_left = self.get_food_count(game_state)
        carrying = self.num_carrying(game_state)

        f = self.get_edibles(game_state)
        h = self.get_closest_home_cell_position(game_state)
        c = self.get_capsules(game_state)
        p = self.get_enemy_pacmen_positions(game_state)
        
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        closest_reachable_food = self.closest_reachable_food(game_state, f)
        defender_food = self.defender_food(game_state, f)
        food_eaten = list(self.last_food_eaten)

        my_pos = self.get_my_position(game_state)
        scared_time = self.get_scared_time(game_state)

        pacmens = self.get_enemy_pacmen_positions(game_state)
        pacmens = self.sort_distances(self.calculate_distances_with_positions(game_state, pacmens))

        # If you end up being trapped, give up
        if self.panic(game_state): 
            self.state = STATE_PANIC
            return my_pos, []

        if carrying and (game_state.data.timeleft <= 100 or food_left <= 2):
            self.state = STATE_HOME
            return h[0], []

        if scared_time >= 20 and defender_food:
            self.state = STATE_ATTACK
            return defender_food[0], []
        
        if scared_time >= 20 and closest_reachable_food:
            self.state = STATE_ATTACK
            return closest_reachable_food, []
        
        if carrying:
            self.state = STATE_HOME
            return h[0], []

        # Scared ghosts also allowed to defend
        if pacmens:
            self.state = STATE_DEFEND
            return pacmens[0], []

        if food_eaten:
            self.state = STATE_INVESTIGATE
            self.missing_food_cell = food_eaten[0]
            return self.missing_food_cell, []

        if self.state == STATE_INVESTIGATE and my_pos != self.missing_food_cell:
            return self.missing_food_cell, []
    
        if f:
            self.state = STATE_ATTACK
            return f[0], []

        self.state = STATE_HOME
        return random.choice(h), self.get_edge_home_cells(game_state, enemy_home=True)

    def defender_food(self, game_state, closest_reachable_food):
        my_team = game_state.get_red_team_indices() if self.red else game_state.get_blue_team_indices()
        my_teammate = my_team[1] if my_team[0] == self.index else my_team[0]
        my_teammate_pos = game_state.get_agent_position(my_teammate)

        foods = []
        for food in closest_reachable_food:
            my_dist = self.get_maze_distance(self.get_my_position(game_state), food)
            teammate_dist = self.get_maze_distance(my_teammate_pos, food)   
            if my_dist < teammate_dist:
                foods.append(food)

        return foods