# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent


from .utils import board_state
from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir
from queue import PriorityQueue


actions_tracker = [0,0,0,0,0,0,0]

# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """

        self.current_state = board_state() # initialise empty board state
       

        # generate two levels of childre to test generate_children()
        
        #self.current_state.render_board_state() 
        """
        next_states = self.current_state.generate_children()
        while not next_states.empty():
            state = next_states.get()[-1]
            state.render_board_state()
            next_next_states = state.generate_children()
            while not next_next_states.empty():
                (next_next_states.get()[-1]).render_board_state()
        """

        

        self._color = color
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        
        minimax_level = 2
        if referee["time_remaining"]:
            if referee["time_remaining"] < 10:
                minimax_level = 0
            elif referee["time_remaining"] < 30:
                minimax_level = 1
            elif referee["time_remaining"] < 100:
                minimax_level = 2
            elif referee["time_remaining"] < 160:
                minimax_level = 3
            elif referee["time_remaining"] < 179:
                minimax_level = 4
        print(referee)
        #print(referee["time_remaining"])
        #print(referee["space_remaining"])
        #print(referee["space_limit"])
        
        actions_tracker[minimax_level] += 1
        print(actions_tracker)
        return self.current_state.next_action(2)
        match self._color:
            case PlayerColor.RED:
                return SpawnAction(HexPos(3, 3))
            case PlayerColor.BLUE:
                # This is going to be invalid... BLUE never spawned!
                return SpreadAction(HexPos(3, 3), HexDir.Up)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        self.current_state.update_board(action)

        match action:
            case SpawnAction(cell):
                print(f"Testing: {color} SPAWN at {cell}")
                pass
            case SpreadAction(cell, direction):
                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass

    # I moved the function of initialise_board() into the board_state class - Kevin
