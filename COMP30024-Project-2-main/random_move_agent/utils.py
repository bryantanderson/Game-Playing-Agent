
import math
import random
from referee.game import \
    Board, PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir
from queue import PriorityQueue
from copy import deepcopy

# possibly remove these and use the ones in constants.py

MAX_CELL_POWER = 6
SIDE_WIDTH = 7
# (dr, dq) must be one of: (0, 1), (−1, 1), (−1, 0), (0, −1), (1, −1), or, (1, 0)
VALID_DIRECTIONS = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
RED_CELL = "r"
BLUE_CELL = "b"
MAX_TOTAL_POWER = MAX_CELL_POWER * SIDE_WIDTH * SIDE_WIDTH
MINIMAX_LEVEL = 2



# a wrapper class for the Board class, to mainly provide an evaluation function for the state and compute actions
class board_state:
    total_explored = 0
    total_pruned = 0   
    def __init__(self, parent=None, board: Board = Board(), action_taken: Action = None):
        self.parent = parent
        self.board = board
        self.action_taken = action_taken


    def render_board_state(self):
        print("================================\n")
        print("turn:" + str(self.board.turn_color))
        print("blue power: " + str(self.board._color_power(PlayerColor.BLUE)))
        print("red power: " + str(self.board._color_power(PlayerColor.RED)))
        #print("g value: " + str(self.g_value))
        print("action taken: " + str(self.action_taken))
        #print("f value: " + str(self.compute_f_value()))
        print(self.board.render())
        print("================================\n")



    """

    def get_blue_cells(self):
        # we can also just record this as an attribute and update it only when we elimate a blue cell
        blue_cells = []
        for coords, cell in self.board.items():
            if cell[0] == "b":
                blue_cells.append(coords)
        return blue_cells

    def compute_f_value(self):
        return self.g_value + self.least_total_cost()

    """

    def update_board(self, action: Action):
        self.board.apply_action(action)
        return
    
    def next_action(self, minimax_level) -> Action:
        # returns action_taken of the child state with maximum minimax value
        if(minimax_level == 0):
            return self.random_move()

        best_state = None
        color = self.board.turn_color
        score_dir = self.score_direction(color)
        max_val = -1 *(MAX_TOTAL_POWER+1) * score_dir
        maximums = {PlayerColor.RED: -1 *MAX_TOTAL_POWER, PlayerColor.BLUE:MAX_TOTAL_POWER}
        children = self.generate_children()
        while not children.empty():
            entry = children.get()
            #print(entry)
            state = entry[-1]
            #state.render_board_state()
            minimax_val = state.minimax_value(minimax_level-1, maximums.copy())
       
            #print(maximums)
            if minimax_val * score_dir  > max_val * score_dir:
                max_val = minimax_val
                maximums[color] = max_val
                best_state = state
        
        print("EXPLORED:")
        print(board_state.total_explored)
        print("PRUNED:")
        print(board_state.total_pruned)

        return best_state.action_taken

    def evaluation_function(self) -> int:
        if(self.board.game_over):
            #print("game should end! don't pick this")
            return MAX_TOTAL_POWER * self.score_direction(self.board.turn_color.opponent)

        # dummy evaluation function
        evaluation = self.board._color_power(PlayerColor.RED) - self.board._color_power(PlayerColor.BLUE)
        return evaluation

    def minimax_value(self, level, vals: dict[PlayerColor, int]):
        # terminal test
        if level == 0 or self.board.game_over:
            return self.evaluation_function()

        # implement algorithm here:
        children = self.generate_children()
        color = self.board._turn_color
        opponent_color = self.board._turn_color.opponent

        explored = 0 # track explored nodes to measure performance

        while not children.empty():
            # get max / min value of the children states ...
            tuple = children.get()
            explored += 1
            #print(tuple)
            state = tuple[-1]

            #state.render_board_state()
            minimax_val = state.minimax_value(level - 1, vals.copy())
            # print("  "* level + str(minimax_val))
            vals[color] = max(vals[color] * self.score_direction(color),
                               minimax_val * self.score_direction(color)) * self.score_direction(color)
            if vals[color] * self.score_direction(color) >= vals[opponent_color] * self.score_direction(color):
                """
                print("explored:")
                print(explored)
                print("pruned:")
                print(children.qsize())
                """
                board_state.total_explored += explored
                # print(" "*level + str(explored))
                board_state.total_pruned += children.qsize()
                return vals[opponent_color]

        # return alpha/ beta based on turn color
        return vals[color]

    def random_move(self) -> Action:
        while (True):
            x = random.randint(0, SIDE_WIDTH-1)
            y = random.randint(0, SIDE_WIDTH-1)
            pos = HexPos(x, y)
            if (self.board.__getitem__(pos).player == None):
                action = SpawnAction(pos)
                return action



    def score_direction(self, color: PlayerColor) -> int:
        if (color == PlayerColor.RED):
            return 1
        return -1
    

    # need to rewrite this to include spread actions, and taken player color into account.
    def generate_children(self) -> PriorityQueue:
         # generate every possible actions and get the resulting board states
        turn_color = self.board.turn_color
        score_dir = self.score_direction(turn_color)
        children = PriorityQueue()  # pq in case we need to sort it by other measures
        insert_order = 0 
        board_cpy = deepcopy(self.board) # have to use a copy to check cells so that the original isn't changed
        # idea: prioritise "relevant" moves to maximise pruning?
        relevance = 0 # variable that attempts to estimate the relevance of a move
        #self.render_board_state()
        # spread first 
        for coordinates, cell in self.board._state.items():
            if cell.player != self.board.turn_color:
                continue  

            #neighbours_power = self.get_neighbours_power(coordinates)
            #relevance = neighbours_power[PlayerColor.RED] + neighbours_power[PlayerColor.BLUE] # more total power, more relevant?

            for direction in HexDir:
                # have to use deepcopy since it is impossible to create a copy of Board through its constructor, and turn_color is not mutable
                action = SpreadAction(coordinates, direction) 
                spread_power = self.get_powers_in_spread(board_cpy, action)
                relevance = (spread_power[turn_color] + spread_power[turn_color.opponent]*2) # prioritise capturing
                # not consider spreads that captures no cells
                if relevance == 0:
                  
                    continue
                
                children.put((-1 *relevance,insert_order, self.create_child(action)))
                
                insert_order += 1
                
        # spawn, from closest to furthest possible cells - closer = more relevant?

        
            

        if (False):
            # spawn on empty cells
            for r in range(0,SIDE_WIDTH ):
                for q in range(0, SIDE_WIDTH):  
                    spawn_pos = HexPos(r, q)       
                        # checks if cell valid
                    if (board_cpy.__getitem__(spawn_pos).player != None):
                        # print(spawn_pos)
                        continue
                
                    neighbours_power = self.get_neighbours_power(spawn_pos)
                    relevance = relevance = neighbours_power[PlayerColor.RED] + neighbours_power[PlayerColor.BLUE] # more total power, more relevant?
                        
                    action = SpawnAction(spawn_pos)
                    children.put(( -1 *relevance,insert_order, self.create_child(action)))
                    insert_order +=1

        else:
            # spawn based on relative location of existing cells
            spawned = set()
            for distance in range(1, SIDE_WIDTH//2):
                for coordinates, cell in self.board._state.items():   
                    # empty cell
                    if(not cell.player):
                        continue      
                    for direction in HexDir:
                        spawn_r = coordinates.r + direction.r * distance
                        spawn_q = coordinates.q + direction.q *distance
                        spawn_pos = HexPos(spawn_r % SIDE_WIDTH, spawn_q % SIDE_WIDTH)
                        
                        #print(spawn_pos)    
                        # checks if cell valid
                        if (board_cpy.__getitem__(spawn_pos).player != None or spawn_pos in spawned):
                            # print(spawn_pos)
                            continue
                        spawned.add(spawn_pos)
                        neighbours_power = self.get_neighbours_power(board_cpy,spawn_pos)
                        relevance = relevance =(neighbours_power[PlayerColor.RED] - neighbours_power[PlayerColor.BLUE]) * score_dir # more total power, more relevant?
                        
                        action = SpawnAction(spawn_pos)
                        children.put(( -1 *relevance,insert_order, self.create_child(action)))
                        insert_order +=1
        
        # and then consider the cells that are not in a same line as any of the existing cells?

        if children.empty():
            # board must be empty? spawn at 0,0
         
            action = SpawnAction(HexPos(0,0))
            children.put((0, insert_order, self.create_child(action)))
        

        # print(len(spawned))   
        return children
    

    def create_child(self, action: Action):
        # creates a new board state from the current state, given an action
        board_copy = deepcopy(self.board)
        board_copy.apply_action(action)
        new_state = board_state(self, board_copy, action)
        # new_state.render_board_state()
        return new_state
       
    def get_neighbours_power(self, board_cpy, coordinates: HexPos)-> dict[PlayerColor, int]:
        # counts the total power of each color in the neighbouring cells of a cell
        counts = {PlayerColor.RED: 0, PlayerColor.BLUE:0}
        # board_cpy = deepcopy(self.board)
        for direction in HexDir:
            r = coordinates.r + direction.r 
            q = coordinates.q + direction.q
            pos = HexPos(r % SIDE_WIDTH, q % SIDE_WIDTH)
            cell = board_cpy.__getitem__(pos)
            if (cell.player != None):
                counts[cell.player] += cell.power

        return counts

    def get_powers_in_spread(self,board_cpy, action:Action)-> dict[PlayerColor, int]:
        counts = {PlayerColor.RED: 0, PlayerColor.BLUE:0}
        # board_cpy = deepcopy(self.board)
        for distance in range(1, board_cpy.__getitem__(action.cell).power + 1):
            r = action.cell.r + action.direction.r * distance
            q = action.cell.q + action.direction.q * distance
            pos = HexPos(r % SIDE_WIDTH, q % SIDE_WIDTH)
            cell = board_cpy.__getitem__(pos)
            if (cell.player != None):
                counts[cell.player] += cell.power
        return counts

    # trace back to root node and find all actions taken
    def get_all_actions(self) -> list[Action]:
        actions = []
        curr_node = self
        while (curr_node.parent):
            actions.insert(0, curr_node.action_taken)
            curr_node = curr_node.parent
        return actions

   

    # computes the minimum cost for a red cell to spread to a blue cell
    @staticmethod
    def least_cost_from_cell(from_x: int, from_y: int, cell: HexPos, power: int) -> int:
        x_diff = abs(from_x - cell.r)
        y_diff = abs(from_y - cell.q)
        min_distance = min(x_diff, y_diff) + abs(x_diff - y_diff)
        return max(0, min_distance - power) + 1


    # no longer needed?
    """
    # the heuristic: the sum of the least costs to spread to each of the blue cells
    def least_total_cost(self):
        # key: coords of the cell from which the least cost is achieved; value: cost
        least_costs = {}
        for blue_cell in self.get_blue_cells():
            # initialise least_cost with a very big number
            least_cost = float('inf')
            from_cell = None

            for coords, red_cell in self.board.items():
                if red_cell[0] == 'r':
                    cost = self.least_cost_from_cell(coords[0], coords[1], blue_cell, red_cell[1])

                    if cost < least_cost:
                        least_cost = cost
                        from_cell = coords

            # can be ignored if there is already a larger least_cost from this cell
            if from_cell in least_costs.keys() and least_cost < least_costs[from_cell]:
                continue

            least_costs[from_cell] = least_cost

        return sum(least_costs.values())
    """