# Tetris RL Agent - Final AI Project

My final project of AI

## Introduction

This is the final project for my Artificial Intelligence course. It involves building and training a reinforcement learning agent to play Tetris using a custom environment built with the Gymnasium library.

## About the environment

The environment was not registered via `gymnasium.envs.registration`

|                   |                                  |
|-------------------|----------------------------------|
| Action Space      | Discrete(8)                      |
| Observation Space | Box(0, 255, (660, 300, 3),uint8) |
| Import              | from tetris_env import TetrisEnv |

### Description

The main objective was to add the same behaviour as modern version of Tetris (like Jstris or Tetr.io), however, due to my lack of time, knowledge about gymnasium and game-logic in Python, I wasn't able to recreate 100% of the features. The environment was created using the gym.Env from gymnasium library.

### Features

#### Implemented

- SRS Rotation
- 7-Bag
- Forced gravity (the piece moves 1 cell down after 6 non-drop actions)

#### Not implemented

- Piece Spins and Kick Table
- T-Spins detection
- Full clear detection
- Natural gravity (timed)
- Ghost Piece (not implemented in the observation)

### Actions

|VALUE|MEANING|
|-|-|
|0|MOVE LEFT|
|1|MOVE RIGHT|
|2|ROTATE CLOCKWISE|
|3|SOFTDROP|
|4|HARDROP|
|5|HOLD PIECE|
|6|ROTATE COUNTER-CLOCWISE|
|7|ROTATE 180 DEGREES|

### Observations

TetrisEnv has 2 types of observation

- `obs_type="dict"` -> a dictionary with information of the board
- `obs_type="rgb_array"` -> `observation_space=Box(0, 255, (660, 300, 3), np.uint8)`

'dict' observation returns the following information:

```json
obs =  {
    "board": board,
    "queue": next_pieces_keys,
    "piece_type": current_piece.shape_key,
    "piece_shape": current_piece.shape,
    "hold_piece": hold_piece,
    "piece_position": piece_position,
    "current_piece_n_action": current_piece_n_action,
    "current_piece_rotation": current_piece.rotation,
    "last_piece_lock_height": last_piece_lock_height
}
```

### Rewards

Rewards are calculated after each piece is placed, based on the following factors:

- **Height (Low/High)**:
  - Positive reward if the stack is below the 7th layer.
  - Negative reward if above. This reward scales exponentially based on the number of filled cells per layer.
  
- **Holes**:
  - Penalty for each empty cell with filled cells above it.

- **Cheese Layers**:
  - Penalty for each filled cell above a hole.

- **Bumpiness & Flatness**:
  - Penalty for uneven column heights.
  - Bonus for flat columns (height difference = 0).

- **Finesse**:
  - Bonus if the piece is placed within 20 actions.
  - Penalty if more than 20.

### Render modes

- `render_mode=human`
- `render_mode=rgb_array`

## Installation

Tested with **Python 3.12.1**

```bash
pip install -r requirements.txt
```

- To use the notebooks create the folders: 'plots', 'models'

## Example Usage

```python
from tetris_env import TetrisEnv
env = TetrisEnv(board_width=10,
                board_height=22,
                time_limit_s=120,
                render_mode='rgb_array')
obs, info = env.reset()
```

## Playing the game

play.py is a python script that let's you play the game manually and it will create a dataset with transitions of the game inside the folder 'datasets' (created automatically). The main objective was to use the dataset to experiment with Imitation Learning.

Main has four arguments

```python
main(
    game_mode="opener", 
    time_limit=300,
    fill_cell_prob=0.8,
    gen_tab_height=7,
    mode_interval=7
)
```

It has 3 gamemodes, each one has a timer that ends the game if certain time has passed (defined by time_limit, in seconds)

- **clean**: Only has a timer
- **cheese**: Start with a random set of locked pieces in the board (defined by gen_tab_height), and every **mode_interval** (in seconds) it will generate a random layer at the bottom of the board, the probability of each cell to be a filled is defined by 'fill_cell_prob'
- **opener**: The game will reset your board, queue and current piece every **mode_interval**, it will show a warning at the screen 2 seconds before each reset

### WARNING

If the environment is modified all the datasets created from the previous environment could become deprecated, meaning that using datasets from different environment versions could make the agent learn a non-optimal policy.

## About the notebooks

The training notebooks use a modified version of DQN algorithm used in the [Pytorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
