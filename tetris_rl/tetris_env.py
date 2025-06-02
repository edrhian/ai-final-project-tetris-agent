"""
Entorno de Gym
"""
import numpy as np
import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from enum import Enum
import time

#region CONST
# PIECE COLORS for coloring current, hold and queue pieces
SHAPE_STR_COLORS = {
    'None':(0, 0, 0),
    'I': (0, 255, 255),     # Cyan
    'O': (255, 255, 0),     # Yellow
    'T': (128, 0, 128),     # Purple
    'S': (0, 255, 0),       # Green
    'Z': (255, 0, 0),       # Red
    'J': (0, 0, 255),       # Blue
    'L': (255, 165, 0),     # Orange
}

SHAPES_IDS = {
    'None':0,
    'I':1,
    'O':2,
    'T':3,
    'S':4,
    'Z':5,
    'J':6,
    'L':7
}

# PIECE COLOR for coloring locked in board
SHAPE_ID_COLOR_ARR = [
    (0, 0, 0),        # None
    (0, 255, 255),    # I
    (255, 255, 0),    # O
    (128, 0, 128),    # T
    (0, 255, 0),      # S
    (255, 0, 0),      # Z
    (0, 0, 255),      # J
    (255, 165, 0)     # L
]

# PIECE DEFAULT SHAPES
# Using Super Rotation System (SRS)
# https://cdn.harddrop.com/thumb/3/3d/SRS-pieces.png/300px-SRS-pieces.png
SHAPES = {
    'I': np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    
    'T': np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]),
    
    'O': np.array([
        [1, 1],
        [1, 1]
    ]),
    
    'L': np.array([
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ]),
    
    'J': np.array([
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]),
    
    'S': np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ]),
    
    'Z': np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ])
}

#endregion

#region ENV
#####
#ENV#
#####

class Actions(Enum):
    left = 0
    right = 1
    rotate_clock = 2
    softdrop = 3
    hardrop = 4
    hold_piece = 5
    rotate_counter_clock = 6
    rotate_180 = 7

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30,"obs_types":["dict","rgb_array"]}
    def __init__(self, board_width=10, board_height=22, time_limit_s=120, obs_type="dict",render_mode=None, internal_config=None):
        if internal_config == None:
            print("Using Default internal config")
            self.internal_config={
                'r_finesse': 0.1,
                'p_finesse': 0.01,
                'p_holes': 1,
                'p_cheese': 1,
                'r_max_layer_stack_height': 6,
                'r_stack_height': 1,
                'p_exp_stack_height': 0.4,
                'p_const_stack_height': 0.5,
                'p_mult_stack_height': 0.1,
                'r_flatness': 1,
                'p_mult_bumpiness': 1,
                'r_const_clear_lines': 0,
                'r_clear_lines': [0, 2, 6, 12, 18],
                'r_survive_step':0,
                'r_const_hardrop':0.01,
                'r_softdrop':0,
                'p_game_over':100,
                'p_bug':10000,
            }
        else:
            print("Using Custom internal config")
            self.internal_config = internal_config
        
        self.window_size = 512  # The size of the PyGame window
        
        # TODO add Time Limit
        self.time_limit_s = time_limit_s

        self.board_width = board_width
        self.board_height = board_height
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)

        self.internal_config['p_stack_height_values'] = self.make_stack_height_values()

        self.bag = []

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(660,300,3), dtype=np.uint8)
        
        self.current_piece = None

        self.auto_softdrop_delay = 4

        self.piece_n_action_limit = self.board_height * 4 * self.board_width

        self._action_to_movement = {
            Actions.left.value: lambda: self.current_piece.move(-1, 0, self.board),
            Actions.right.value: lambda: self.current_piece.move(1, 0, self.board),
            Actions.rotate_clock.value: lambda: self.current_piece.rotate(self.board),
            Actions.softdrop.value: lambda: self._softdrop_in_board(),
            Actions.hardrop.value: lambda: self._hardrop_in_board(),
            Actions.hold_piece.value: lambda: self.hold_current_piece(),
            Actions.rotate_counter_clock.value: lambda: self.current_piece.rotate_counter(self.board),
            Actions.rotate_180.value: lambda: self.current_piece.rotate_180(self.board)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert obs_type is None or obs_type in self.metadata["obs_types"]
        self.obs_type = obs_type

        self.window = None
        self.clock = None
    
    def _get_obs(self):
        if self.obs_type == 'rgb_array':
            return self.get_screen_rgb()
            
        next_pieces_keys = [piece.shape_key for piece in self.next_piece[:5]]
        hold_piece = self.hold_piece.shape_key if self.hold_piece else 'None'
        piece_position = (self.current_piece.x + 2, self.current_piece.y) # +2 porque hay offset en la izquierda
        
        obs =  {
            "board":self.board,
            "queue":next_pieces_keys,
            "piece_type":self.current_piece.shape_key,
            "piece_shape":self.current_piece.shape,
            "hold_piece":hold_piece,
            "piece_position":piece_position,
            "current_piece_n_action":self.current_piece_n_action,
            "current_piece_rotation":self.current_piece.rotation,
            "last_piece_lock_height":self.last_piece_lock_height
            }
        
        return obs

    def _get_info(self):
        return {
            'lines_cleared': self.total_lines_cleared,
            'score': self.score,
            'pieces_placed': self.total_pieces_placed
        }
        

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.terminated = False
        self.truncated = False
        
        self.board.fill(0)

        self.bag = self.get_bag()
        self.current_piece = self._next_piece()
        self.next_piece = [self._next_piece() for _ in range(5)]
        self.hold_piece = None
        
        self.can_hold = True
        
        self.score = 0
        
        self.current_piece_n_action = 0
        self.remaining_moves = 6

        self.total_lines_cleared = 0
        self.total_pieces_placed = 0

        self.last_piece_lock_height = 0
        
        observation = self._get_obs()
        info = self._get_info()    

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        self.reward = 0
        self.current_piece_last_y = self.current_piece.y
        action = self._action_to_movement[action]()
        
        self.current_piece_n_action += 1
        
        # Non-timed-gravity, force piece to go down after x moves of the current piece
        if self.current_piece_n_action % self.auto_softdrop_delay == 0 and self.current_piece_n_action > 0 :
            self.current_piece.move(0, 1, self.board)

        # Force piece to lock if
        # has touched ground and has made 4 non-lock actions
        # Is bugging the game (Infinite rotations)
        self._check_lock_delay()
                
        observation = self._get_obs()
        info = self._get_info()
        
        # BUG: Hardcoded fix, if every action is a rotation that prevents
        # the piece from touching the ground, the piece never locks, send truncated if to many moves
        if self.current_piece_n_action > self.piece_n_action_limit:
            self.truncated = True
            self.reward -= self.internal_config['p_bug']
        
        if self.terminated:
            self.reward -= self.internal_config['p_game_over']

        if self.render_mode == "human":
            self._render_frame()

        if not self.terminated and not self.truncated:
            self.reward += self.internal_config['r_survive_step']

        return observation, self.reward, self.terminated, self.truncated, info

    #region BOARD_REWARDS
    #####################
    #BOARD REWARDS LOGIC#
    #####################
    def calc_lock_piece_rewards(self):
        self.masked_board = np.where(self.board > 0, 1, self.board)
        sh = self._calc_stack_height()
        h = self._calc_holes()
        c = self._calc_cheese()
        b = self._calc_bumpiness()
        f = self._calc_finesse()
        
        return sh, h, c, b, f
    
    # Finesse: Mechanical Skill -> Less inputs per piece = more rewards
    def _calc_finesse(self):
        if self.current_piece_n_action >= 20:
            return self.current_piece_n_action * -self.internal_config["p_finesse"]
        else:
            return self.internal_config["r_finesse"]

    # Number of holes
    def _calc_holes(self):
        holes = 0
        for col in self.masked_board.T:
            if not np.any(col == 1):
                continue
    
            first_block = np.argmax(col == 1)
            holes += np.sum(col[first_block+1:] == 0)
    
        return -holes / ((self.board_width - 1) * (self.board_height - 1))
        
    # Cheese: Number of layers to clear in order to clean all the rows above a hole
    def _calc_cheese(self):
        cheese_board = np.flip(self.masked_board.T)
        n_needed_clears = 0
        
        for col in cheese_board:
            possible_cheese = False
            for sqr in col:
                if sqr == 0 and not possible_cheese:
                    possible_cheese = True
                elif sqr == 1 and possible_cheese:
                    n_needed_clears += 1
        return -n_needed_clears / ((self.board_width - 1) * (self.board_height - 1))

    # How higher is the stack
    # Stacking below 'r_max_layer_stack_height' rewards the agents
    # Stacking above 'r_max_layer_stack_height' gives a exponential penalty
    def _calc_stack_height(self):
        layer_sums = np.sum(self.masked_board, axis=1)
        weights = self.internal_config['p_stack_height_values']
        
        mask = weights <= 0
        result = np.where(mask & (layer_sums > 0), weights / layer_sums * (self.board_width - 1), weights * layer_sums / (self.board_width - 1))
    
        return np.sum(result)

    # Height difference between adjacent columns
    def _calc_bumpiness(self):
        if not np.any(self.masked_board):
            return 0
        total_reward = 0
        transposed_board = self.masked_board.T
        for x in range(len(transposed_board)-1):
            col_heights = np.argmax(self.masked_board == 1, axis=0)
            col_empty = ~np.any(self.masked_board == 1, axis=0)
            col_heights[col_empty] = self.masked_board.shape[0]
                
            height_diffs = np.abs(np.diff(col_heights))
        
            flatness_reward = np.sum(height_diffs == 0) / (self.board_width - 1)
            
            
            bumpiness_penalty = np.sum(height_diffs[height_diffs > 0] / (self.board_height * (self.board_width - 1)))
            
        return flatness_reward - bumpiness_penalty
    #endregion
    
    #region GAME_LOGIC
    ############
    #GAME LOGIC#
    ############
    # Generate 7-Bag
    def get_bag(self):
        keys = list(SHAPES.keys())
        sampled_keys = self.np_random.choice(keys, size=7, replace=False)
        return list(sampled_keys)

    def _check_game_over(self):
        if self.current_piece._collides(self.board, self.current_piece.x, self.current_piece.y, self.current_piece.shape):
            return True
        return False

    def _next_piece(self):
        if not self.bag:
            self.bag = self.get_bag()
        return Piece(self.bag.pop(),self.board_width)

    def spawn_new_piece(self):
        self.current_piece = self.next_piece.pop(0)
        self.next_piece.append(self._next_piece())
        self.current_piece.reset_position(self.board_width)

        self.terminated = self._check_game_over()

    def hold_current_piece(self):
        if self.can_hold:
            self.current_piece.reset_position(self.board_width)
            
            if self.hold_piece is None:
                self.hold_piece = self.current_piece
                self.current_piece = self.next_piece.pop(0)
                self.next_piece.append(self._next_piece())
            else:
                self.hold_piece, self.current_piece = self.current_piece, self.hold_piece

            self.current_piece_n_action = 0
            self.terminated = self._check_game_over()
            self.remaining_moves = 6
            self.can_hold = False

    def _lock_in_board(self):
        self.current_piece.lock(self.board)
        self.last_piece_lock_height = self.current_piece.y
        lines_cleared = self._clear_lines()
        
        self.total_lines_cleared += lines_cleared
        self.total_pieces_placed += 1
        line_reward = self.internal_config['r_clear_lines'][lines_cleared]
        self.reward += self.internal_config['r_const_clear_lines'] + line_reward
        
        self.score += [0, 100, 300, 500, 800][lines_cleared]
        
        self.spawn_new_piece()
        self.current_piece_n_action = 0
        self.can_hold = True
        self.remaining_moves = 6
        
        sh, h, c, b, f = self.calc_lock_piece_rewards()
        self.reward += sh + h + c + b + f
        
        return lines_cleared

    def _clear_lines(self):
        lines_cleared = 0
        full_lines = np.all(self.board > 0, axis=1)  # Detect if all cells of a line are > 0
    
        if np.any(full_lines):
            lines_cleared = np.sum(full_lines)
            
            self.board[lines_cleared:] = self.board[~full_lines]
            self.board[:lines_cleared] = np.zeros_like(self.board[:lines_cleared])
    
        return lines_cleared

    def _check_lock_delay(self):
        # BUGFIX (Eternal Rotation)
        if self.current_piece_last_y == self.current_piece.y:
            self.remaining_moves -= 1
        else:
            self.remaining_moves = 6
            
        if self.remaining_moves <= 0:
            return self._hardrop_in_board()
        return 0

    # Rewarding speed
    def _hardrop_in_board(self):
        drops = self.current_piece.hard_drop(self.board)
        lines_cleared = self._lock_in_board() 
        self.reward += self.internal_config['r_const_hardrop'] * drops

    def _softdrop_in_board(self):
        self.current_piece.move(0, 1, self.board)
        self.reward += self.internal_config['r_softdrop']

    #endregion
    
    #region VISUAL_RENDER
    ##################
    #VISUAL RENDERING#
    ##################
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        self.cell_size = 30
        self.width = self.board_width * self.cell_size + 200 # +200 Hold Piece and Queue region
        self.height = self.board_height * self.cell_size
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode(
                (self.width, self.height)
            )
            
            pygame.display.set_caption("Tetris AI")
            
            self.font = pygame.font.SysFont('Arial', 24)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((0, 0, 0))

        # Board rendering
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x]:
                    pygame.draw.rect(
                        self.canvas,
                        SHAPE_ID_COLOR_ARR[self.board[y][x]],  # Color de las piezas en el tablero
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1  # borde
                    )

        # Current Piece rendering
        for dy, row in enumerate(self.current_piece.shape):
            for dx, cell in enumerate(row):
                if cell:
                    px = (self.current_piece.x + dx) * self.cell_size
                    py = (self.current_piece.y + dy) * self.cell_size
                    pygame.draw.rect(
                        self.canvas,
                        SHAPE_STR_COLORS[self.current_piece.shape_key],  # Color para la pieza actual
                        pygame.Rect(px, py, self.cell_size, self.cell_size)
                    )
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        pygame.Rect(px, py, self.cell_size, self.cell_size),
                        1
                    )

        # Hold Piece rendering
        if self.hold_piece:
            self._draw_piece(self.hold_piece.shape, 0, 0, offset_x=self.board_width * self.cell_size + 20, offset_y=50, color=SHAPE_STR_COLORS[self.hold_piece.shape_key])

        # Queue rendering
        for idx, piece in enumerate(self.next_piece[:5]):
            self._draw_piece(piece.shape, 0, 0, offset_x=self.board_width * self.cell_size + 20, offset_y=150 + idx * 100, color=SHAPE_STR_COLORS[piece.shape_key])

        # Score rendering
        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            
            score_surface = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
            self.window.blit(score_surface, (self.board_width * self.cell_size + 20, 10))

            pygame.event.pump()
            pygame.display.update()
        
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def _draw_piece(self, shape, x, y, offset_x=0, offset_y=0, color=(255, 255, 255)):
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell:
                    px = offset_x + (x + dx) * self.cell_size
                    py = offset_y + (y + dy) * self.cell_size
                    pygame.draw.rect(
                        self.canvas,
                        color,
                        pygame.Rect(px, py, self.cell_size, self.cell_size)
                    )
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        pygame.Rect(px, py, self.cell_size, self.cell_size),
                        1
                    )

    def get_screen_rgb(self):
        self.cell_size = 30
        self.width = self.board_width * self.cell_size + 200  # espacio extra para hold y next pieces
        self.height = self.board_height * self.cell_size

        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((0, 0, 0))

        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x]:
                    pygame.draw.rect(
                        self.canvas,
                        SHAPE_ID_COLOR_ARR[self.board[y][x]],  # Color de las piezas en el tablero
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1  # borde
                    )

        # Current Piece
        for dy, row in enumerate(self.current_piece.shape):
            for dx, cell in enumerate(row):
                if cell:
                    px = (self.current_piece.x + dx) * self.cell_size
                    py = (self.current_piece.y + dy) * self.cell_size
                    pygame.draw.rect(
                        self.canvas,
                        SHAPE_STR_COLORS[self.current_piece.shape_key],  # Color para la pieza actual
                        pygame.Rect(px, py, self.cell_size, self.cell_size)
                    )
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        pygame.Rect(px, py, self.cell_size, self.cell_size),
                        1
                    )

        # Hold Piece
        if self.hold_piece:
            self._draw_piece(self.hold_piece.shape, 0, 0, offset_x=self.board_width * self.cell_size + 20, offset_y=50, color=SHAPE_STR_COLORS[self.hold_piece.shape_key])

        # Queue
        for idx, piece in enumerate(self.next_piece[:5]):
            self._draw_piece(piece.shape, 0, 0, offset_x=self.board_width * self.cell_size + 20, offset_y=150 + idx * 100, color=SHAPE_STR_COLORS[piece.shape_key])

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    #endregion
    
    #region AUX
    #####
    #AUX#
    #####
    def make_stack_height_values(self):
        max_layer = self.internal_config["r_max_layer_stack_height"]
        board_height = self.board_height
        
        # Parte 1: Recompensas para capas bajas
        reward_part = self.internal_config['r_stack_height'] / (np.arange(max_layer) + 1)
        
        # Parte 2: Penalización exponencial para capas altas
        a = self.internal_config['p_mult_stack_height']
        b = self.internal_config['p_exp_stack_height']
        num_high_layers = board_height - max_layer
        high_layer_indices = np.arange(num_high_layers)
        penalty_part = -a * np.exp(b * high_layer_indices)
        
        return np.concatenate((reward_part, penalty_part))[::-1]

    def __str__(self):
        return (
            f"TetrisEnv(\n"
            f"  board_width={self.board_width},\n"
            f"  board_height={self.board_height},\n"
            f"  render_mode={self.render_mode},\n"
            f"  obs_type={self.obs_type},\n"
            f"  internal_config={self.internal_config},\n"
            f")"
        )
    #endregion

#endregion

#region PIECE
#######
#PIECE#
#######
class Piece:
    def __init__(self, shape_key, board_width):
        self.shape_key = shape_key
        self.shape = np.array(SHAPES[shape_key])
        self.rotation = 0
        self.x = board_width // 2 - self.shape.shape[1] // 2
        self.y = 0

    # Moves the piece in a direction, if its only checking just returns if the piece can move or not (eg. collides with piece)
    def move(self, dx, dy, board, only_checking=False):
        if self.can_move(dx, dy, board):
            if not only_checking:
                self.x += dx
                self.y += dy
            return True
        return False

    def can_move(self, dx, dy, board):
        return not self._collides(board, self.x + dx, self.y + dy, self.shape)
    
    def rotate(self, board):
        new_shape = np.rot90(self.shape, k=-1) 
        if not self._collides(board, self.x, self.y, new_shape):
            self.shape = new_shape
            self.rotation = (self.rotation + 1) % 4
            return True
        return False

    def rotate_counter(self, board):
        new_shape = np.rot90(self.shape, k=1)  # Rotación antihoraria
        if not self._collides(board, self.x, self.y, new_shape):
            self.shape = new_shape
            self.rotation = (self.rotation - 1) % 4
            return True
        return False

    def rotate_180(self, board):
        new_shape = np.rot90(self.shape, k=2)  # Rotación 180 grados
        if not self._collides(board, self.x, self.y, new_shape):
            self.shape = new_shape
            self.rotation = (self.rotation + 2) % 4
            return True
        return False

    def _collides(self, board, x, y, shape):
        # Iterates every cell in current piece shape
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:  # If cell is occupied
                    new_x = x + j
                    new_y = y + i
                    # Checks Collisions
                    # - Horizontal 
                    # - Floor (check if the piece reaches the bottom of the board)
                    # - Vertical
                    if (new_x < 0 or new_x >= board.shape[1] or
                        new_y >= board.shape[0] or
                        (new_y >= 0 and board[new_y][new_x] != 0)):
                        return True
        return False

    # Softdrops until it can't move, returns the height from where it falls
    def hard_drop(self, board):
        drops = 0
        while self.move(0, 1, board):
            drops += 1
        return drops

    # Locks the piece into the board, used in TetrisEnv._lock_in_board
    def lock(self, board):
        # Iterates every cell in current piece shape
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                # If cell is part of the piece (>0)
                if cell:
                    x, y = self.x + j, self.y + i # Piece Cell Coords
                    # If piece inside board limits
                    if 0 <= y < board.shape[0] and 0 <= x < board.shape[1]:
                        board[y][x] = SHAPES_IDS[self.shape_key]

    # When holding or spawning the piece, sets the position and rotation
    def reset_position(self, board_width):
        self.shape = SHAPES[self.shape_key]
        self.x = board_width // 2 - self.shape.shape[1] // 2
        self.y = 0
        self.rotation = 0
#endregion