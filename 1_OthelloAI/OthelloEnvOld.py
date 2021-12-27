import gym
from gym import spaces
import numpy as np
import string
import logging

class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(OthelloEnv, self).__init__()
        self.name = 'othello'
        self.manual = manual
        self.grid_length = 8
        self.n_players = 2
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)
        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        self.verbose = verbose

    @property
    def observation(self):
        if self.players[self.current_player_num].token.number == 1:
            position = np.array([x.number for x in self.board]).reshape(self.grid_shape)
        else:
            position = np.array([-x.number for x in self.board]).reshape(self.grid_shape)
            
        la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
        out = np.stack([position,la_grid], axis = -1)
        return out
            
    @property
    def legal_actions(self):
        legal_actions = [0 for _ in range(self.num_squares)]
        player_number = self.current_player.token.number
        board_matrix = np.matrix([
            [t.number for t in self.board[i:i+self.grid_length]]
            for i in range(0,self.num_squares,self.grid_length)
        ])
        
        for row in range(self.grid_length):
            for col in range(self.grid_length):
                if board_matrix[row,col] != 0: # Check if Token already placed
                    continue
                else: # Check Validity of Moves    
                    args = [board_matrix, legal_actions, player_number, row, col]
                    self.check_legality(*args, 0, -1) # Left
                    self.check_legality(*args, 0, 1) # Right
                    self.check_legality(*args, 1, 0) # Up
                    self.check_legality(*args, -1, 0) # Down
                    self.check_legality(*args, 1, -1) # Up Right
                    self.check_legality(*args, 1, 1) # Up Left
                    self.check_legality(*args, -1, -1) # Down Right 
                    self.check_legality(*args, -1, 1)# Down Left
                
        return np.array(legal_actions)
    
    @property
    def current_player(self):
        return self.players[self.current_player_num]
    
    def check_legality(self, board_matrix, legal_actions, player_number, row, col, sign1, sign2):
        for d in range(1, self.grid_length):
            if (row+(d*sign1))>(self.grid_length-1) or (col+(d*sign2))>(self.grid_length-1):
                break # If outside bounds break
            if (board_matrix[row+(d*sign1), col+(d*sign2)] not in (player_number, 0)):
                continue # If first step is legal continue
            elif d==1: 
                break # If first step is illegal break
            elif (d>1) and (board_matrix[row+(d*sign1), col+(d*sign2)] == 0):
                break # If any subsequent step is illegal break
            elif (board_matrix[row+(d*sign1), col+(d*sign2)] == player_number):
                legal_actions[row*self.grid_length + col] = 1 # If first step legal and final step legal indicate
                
    def check_moves_exist(self, board_matrix, player_number, row, col, sign1, sign2):
        for d in range(1, self.grid_length):
            if (row+(d*sign1))>(self.grid_length-1) or (col+(d*sign2))>(self.grid_length-1):
                break # If outside bounds break
            if (board_matrix[row+(d*sign1), col+(d*sign2)] not in (player_number, 0)):
                continue # If first step is legal continue
            elif d==1: 
                break # If first step is illegal break
            elif (d>1) and (board_matrix[row+(d*sign1), col+(d*sign2)] == 0):
                break # If any subsequent step is illegal break
            elif (board_matrix[row+(d*sign1), col+(d*sign2)] == player_number):
                return True
        return False
                
    def check_flips(self, board_matrix, player_number, action, sign1, sign2):
        flip_cells = []
        row, col = action//8, action%8
        for d in range(1, self.grid_length):
            if (row+(d*sign1))>(self.grid_length-1) or (col+(d*sign2))>(self.grid_length-1):
                break # If outside bounds break
            if (board_matrix[row+(d*sign1), col+(d*sign2)] not in (player_number, 0)):
                flip_cells.append((row+(d*sign1))*self.grid_length + col + (d*sign2))
                continue # If first step is legal continue
            elif d==1:
                break # If first step is illegal break
            elif (d>1) and (board_matrix[row+(d*sign1), col+(d*sign2)] == 0):
                break # If any subsequent step is illegal break
            elif (board_matrix[row+(d*sign1), col+(d*sign2)] == player_number):
                return set(flip_cells)
        return set([])

    def check_game_over(self):
        board_matrix = np.matrix([
            [t.number for t in self.board[i:i+self.grid_length]]
            for i in range(0,self.num_squares,self.grid_length)
        ])
        
        # Check if next player has move, if so game continues
        player_number = self.players[(self.current_player_num + 1) % 2].token.number
        next_player_has_move = False
        for row in range(self.grid_length):
            for col in range(self.grid_length):
                if board_matrix[row,col] != 0: # Check if Token already placed
                    continue
                else: # Check Validity of Moves    
                    args = [board_matrix, player_number, row, col]
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, 0, -1)) # Left
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, 0, 1)) # Right
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, 1, 0)) # Up
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, -1, 0)) # Down
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, 1, -1)) # Up Right
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, 1, 1)) # Up Left
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, -1, -1)) # Down Right 
                    next_player_has_move = max(next_player_has_move, self.check_moves_exist(*args, -1, 1)) # Down Left
        if next_player_has_move:
            return 0, False
        
        # If next player does not have move, check if current player would, if so game continues
        player_number = self.players[self.current_player_num].token.number
        current_player_has_move = False
        for row in range(self.grid_length):
            for col in range(self.grid_length):
                if board_matrix[row,col] != 0: # Check if Token already placed
                    continue
                else: # Check Validity of Moves    
                    args = [board_matrix, player_number, row, col]
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, 0, -1)) # Left
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, 0, 1)) # Right
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, 1, 0)) # Up
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, -1, 0)) # Down
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, 1, -1)) # Up Right
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, 1, 1)) # Up Left
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, -1, -1)) # Down Right 
                    current_player_has_move = max(current_player_has_move, self.check_moves_exist(*args, -1, 1)) # Down Left
        if current_player_has_move:
            return 0, False
        
        # If no possible moves exist, add up tokens and declare winner
        current_player_score = (board_matrix.flatten()==player_number).sum()
        other_player_score = (board_matrix.flatten()==self.players[(self.current_player_num + 1) % 2].token.number).sum()
        if current_player_score>other_player_score:
            logging.debug(f'\nWinner: {self.players[self.current_player_num].token.symbol} with {current_player_score} to {other_player_score}')
            return 1, True
        else:
            logging.debug(f'\nWinner: {self.players[(self.current_player_num + 1) % 2].token.symbol} with {other_player_score} to {current_player_score}')
            return 0, True        
    
    def step(self, action):
        reward = [0., 0.]
        board = self.board
        
        if type(action)==str: # Translate Coordinate to Integer
            action = (int(action[1])-1)*8 + string.ascii_uppercase.index(action[0])
            
        board_matrix = np.matrix([
            [t.number for t in board[i:i+self.grid_length]]
            for i in range(0,self.num_squares,self.grid_length)
        ])
        player_number = self.current_player.token.number
        
        args = [board_matrix, player_number, action]
        flips = self.check_flips(*args, 0, 1) # Left
        flips |= self.check_flips(*args, 0, -1) # Right
        flips |= self.check_flips(*args, 1, 0) # Up
        flips |= self.check_flips(*args, -1, 0) # Down
        flips |= self.check_flips(*args, 1, -1) # Up Right
        flips |= self.check_flips(*args, 1, 1) # Up Left
        flips |= self.check_flips(*args, -1, -1) # Down Right 
        flips |= self.check_flips(*args, -1, 1) # Down Left
        flips |= {action}
        
        for flip in flips:
            board[flip] = self.current_player.token
        self.turns_taken += 1
        
        r, done = self.check_game_over()
        reward = [-r,-r]
        reward[self.current_player_num] = r
        
        self.done = done
        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward[self.current_player_num], done, {}

    def reset(self):
        self.board = [Token('□', 0)] * self.num_squares
        self.board[27] = Token('○', -1)
        self.board[28] = Token('●', 1)
        self.board[36] = Token('○', -1)
        self.board[35] = Token('●', 1)
        self.players = [Player('Black', Token('●', 1)), Player('White', Token('○', -1))]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logging.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False, verbose = True):
        logging.debug('')
        if close:
            return
        if self.done:
            logging.debug(f'Game over.')
        else:
            logging.debug(f"It is Player {self.current_player.token.symbol}'s turn to move")
        
        logging.debug('\nBoard:')
        logging.debug('  A B C D E F G H')
        logging.debug('1 '+' '.join([x.symbol for x in self.board[:self.grid_length]]))
        logging.debug('2 '+' '.join([x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        logging.debug('3 '+' '.join([x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))
        logging.debug('4 '+' '.join([x.symbol for x in self.board[(self.grid_length*3):(self.grid_length*4)]]))
        logging.debug('5 '+' '.join([x.symbol for x in self.board[(self.grid_length*4):(self.grid_length*5)]]))
        logging.debug('6 '+' '.join([x.symbol for x in self.board[(self.grid_length*5):(self.grid_length*6)]]))
        logging.debug('7 '+' '.join([x.symbol for x in self.board[(self.grid_length*6):(self.grid_length*7)]]))
        logging.debug('8 '+' '.join([x.symbol for x in self.board[(self.grid_length*7):(self.grid_length*8)]]))

        if self.verbose:
            observations = [sub[1] for item in self.observation for sub in item]
            logging.debug('\nMoves:')
            logging.debug('  A B C D E F G H')
            logging.debug('1 '+' '.join(['□' if x!=0 else '☒' for x in observations[:self.grid_length]]))
            logging.debug('2 '+' '.join(['□' if x!=0 else '☒' for x in observations[self.grid_length:self.grid_length*2]]))
            logging.debug('3 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*2):(self.grid_length*3)]]))
            logging.debug('4 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*3):(self.grid_length*4)]]))
            logging.debug('5 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*4):(self.grid_length*5)]]))
            logging.debug('6 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*5):(self.grid_length*6)]]))
            logging.debug('7 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*6):(self.grid_length*7)]]))
            logging.debug('8 '+' '.join(['□' if x!=0 else '☒' for x in observations[(self.grid_length*7):(self.grid_length*8)]]))
            
        
        if not self.done:
            logging.debug(f"\nLegal actions: {['ABCDEFGH'[i%8]+str(i//8+1) for i,o in enumerate(self.legal_actions) if o != 0]}")
    
class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol