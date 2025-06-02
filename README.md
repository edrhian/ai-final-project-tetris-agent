# ia-final-project-tetris-agent

My final project of AI course

## Introduction

I've created an environment to train an agent with reinforcement learning to play Tetris

## About the environment

|                   |                                  |
|-------------------|----------------------------------|
| Action Space      | Discrete(8)                      |
| Observation Space | Box(0, 255, (660, 300, 3),uint8) |
| Make              | from tetris_env import TetrisEnv |

### Description

The main objective was to add the same behaviour as modern version of Tetris (like Jstris or Tetr.io), however, due my lack of time and knowledge about gymnasium, I've only acomplished. The environment was created using the gym.Env from gymnasium library.

### Actions

|VALUE|MEANING|
|-|-|
|0|LEFT|
|1|RIGHT|
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
