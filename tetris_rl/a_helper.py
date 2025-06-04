"""
Aux functions for notebooks
"""
import torch
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import glob
import os

#region BOARD_EVALUATION
def eval_holes(board):
    holes = 0
    for col in board.T:
        if not np.any(col == 1):
            continue

        first_block = np.argmax(col == 1)
        holes += np.sum(col[first_block+1:] == 0)

    return holes

def eval_cheese(board):
    cheese_board = np.flip(board.T)
    n_needed_clears = 0
    
    for col in cheese_board:
        possible_cheese = False
        for sqr in col:
            if sqr == 0 and not possible_cheese:
                possible_cheese = True
            elif sqr == 1 and possible_cheese:
                n_needed_clears += 1
    
    return np.array([n_needed_clears], dtype=np.float32)

def eval_sum_heights(board):
    board = np.array(board)
    heights = board.shape[0] - np.argmax(board[::-1], axis=0)
    heights[np.all(board == 0, axis=0)] = 0
    aggregate_height = np.sum(heights)
    return np.array([aggregate_height], dtype=np.float32)

def eval_bumpiness(board):
    board = np.array(board)
    heights = board.shape[0] - np.argmax(board[::-1], axis=0)
    heights[np.all(board == 0, axis=0)] = 0
    bumpiness = np.sum(np.abs(np.diff(heights)))
    return np.array([bumpiness], dtype=np.float32)

'''
Function that reduces the vision range of the board to the six highest layers
'''
def get_six_layer_view(board):
    has_1 = np.any(board,axis=1)
    has_1_idxs = np.where(has_1)[0]

    if len(has_1_idxs) == 0:
        return board[-6:], board.shape[0]
    
    has_1_first_idx = has_1_idxs[0]

    end_idx = has_1_idxs[0] + 6

    is_offset = end_idx > board.shape[0]
    
    if not is_offset:
        board = board[has_1_first_idx:end_idx,:]
    else:
        board = board[board.shape[0]-6:,:]
    return board, has_1_first_idx

def get_upper_outline_idx(board):
    mask = board == 1
    first_ones = np.argmax(mask, axis=0)
    no_ones = ~mask.any(axis=0)
    first_ones[no_ones] = board.shape[0]

    return first_ones

def filter_board_upper_outline(board):
    result = np.zeros_like(board)
    for col in range(board.shape[1]):
        col_data = board[:, col]
        idx = np.argmax(col_data)
        if col_data[idx] == 1:
            result[idx, col] = 1

    return result
#endregion

#region MODEL_UTILS
def save_dqn_model(i_episode,policy_net,target_net,optimizer,memory,env,checkpoint_path,metadata):
    torch.save({
        'episode': i_episode,
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'memory':memory,
        'env_internal_config':env.internal_config,
        'metadata': metadata
    }, checkpoint_path)
    print(f"Checkpoint saved at episode {i_episode}")
#endregion

#region PLOTTING
def plot_all(episode_durations, episode_rewards, episode_lines, episode_pieces, show_result=False, save=False, filename="plot_x.png"):
    # Ensure inputs are not empty
    if not episode_durations or not episode_rewards or not episode_lines or not episode_pieces:
        raise ValueError("Inputs cannot be empty")

    # Convert lists to tensors
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    lines_t = torch.tensor(episode_lines, dtype=torch.float)
    pieces_t = torch.tensor(episode_pieces, dtype=torch.float)

    # Create the figure with 2x2 layout
    plt.figure(figsize=(14, 10))

    # Plot 1: Durations
    plt.subplot(2, 2, 1)
    plt.title('Durations Result' if show_result else 'Training Durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot 2: Rewards
    plt.subplot(2, 2, 2)
    plt.title('Rewards Result' if show_result else 'Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot 3: Lines
    plt.subplot(2, 2, 3)
    plt.title('Lines Result' if show_result else 'Training Lines')
    plt.xlabel('Episode')
    plt.ylabel('Lines')
    plt.plot(lines_t.numpy())
    if len(lines_t) >= 100:
        means = lines_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot 4: Pieces
    plt.subplot(2, 2, 4)
    plt.title('Pieces Result' if show_result else 'Training Pieces')
    plt.xlabel('Episode')
    plt.ylabel('Pieces')
    plt.plot(pieces_t.numpy())
    if len(pieces_t) >= 100:
        means = pieces_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Layout, save, show, clear
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    if show_result:
        plt.show()
    if hasattr(display, 'display') and not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    plt.close()
#endregion
