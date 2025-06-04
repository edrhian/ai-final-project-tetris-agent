"""
Play Tetris and create a dataset of transitions
Has 3 
- Clean: Normal game
- Cheese: Each mode_interval, the game will generate a random layer at the bottom of the board
- Opener: Each mode_interval, the game will reset the state of the environment, it will give a warning 2 seconds before each reset
"""

import pygame
from tetris_env import TetrisEnv, Piece
import csv
import json
import os
from datetime import datetime
import numpy as np
import random

# Tamaño de cada celda
CELL_SIZE = 30
PANEL_WIDTH = 6 * CELL_SIZE
BOARD_WIDTH = 10
BOARD_HEIGHT = 22
WIDTH = (BOARD_WIDTH * CELL_SIZE) + PANEL_WIDTH
HEIGHT = BOARD_HEIGHT * CELL_SIZE
FPS = 60

PIECES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'Z': [[1, 1, 0],
          [0, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]]
}

ID_TO_SHAPE = {0: None, 1: 'I', 2: 'O', 3: 'T', 4: 'S', 5: 'Z', 6: 'J', 7: 'L'}

# Colores
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
GOLD = (255, 215, 0)

PIECE_COLORS = {
    'I': (0, 255, 255),     # Cyan
    'O': (255, 255, 0),     # Yellow
    'T': (128, 0, 128),     # Purple
    'S': (0, 255, 0),       # Green
    'Z': (255, 0, 0),       # Red
    'J': (0, 0, 255),       # Blue
    'L': (255, 165, 0),     # Orange
}


def draw_piece(screen, piece, offset_x, offset_y, ghost=False):
    """Dibuja una pieza (o un fantasma de pieza) en posición absoluta."""
    if ghost:
        color = (piece.shape_key and PIECE_COLORS.get(piece.shape_key, CYAN))
        color = tuple(min(255, int(c * 0.4)) for c in color)  # Color más clarito para fantasma
    else:
        color = PIECE_COLORS.get(piece.shape_key, CYAN)

    for i, row in enumerate(piece.shape):
        for j, cell in enumerate(row):
            if cell:
                pygame.draw.rect(
                    screen, color,
                    (offset_x + j * CELL_SIZE, offset_y + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

def draw_board(screen, board, current_piece, hold_piece, next_piece, score, time_left, alert_opener=False):
    screen.fill(BLACK)
    
    # Draw board with colors
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            cell_id = board[y][x]
            if cell_id != 0:
                shape_key = ID_TO_SHAPE.get(cell_id, None)
                color = PIECE_COLORS.get(shape_key, GRAY)
                pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Dibujar la pieza fantasma
    ghost_piece = get_ghost_piece(current_piece, board)
    draw_piece(screen, ghost_piece, ghost_piece.x * CELL_SIZE, ghost_piece.y * CELL_SIZE, ghost=True)

    # Draw current piece
    draw_piece(screen, current_piece, current_piece.x * CELL_SIZE, current_piece.y * CELL_SIZE)

    # Draw grid
    for x in range(0, BOARD_WIDTH * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (BOARD_WIDTH * CELL_SIZE, y))

    # Info Panel
    panel_x = BOARD_WIDTH * CELL_SIZE + 10
    font = pygame.font.SysFont("consolas", 20)

    # Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (panel_x, 20))  # Asegúrate de renderizar la puntuación aquí

    # Time
    time_text = font.render(f"Time: {int(time_left)}s", True, WHITE)
    screen.blit(time_text, (panel_x, 50))

    # Hold
    hold_text = font.render("Hold:", True, GOLD)
    screen.blit(hold_text, (panel_x, 100))
    if hold_piece:
        draw_piece(screen, hold_piece, panel_x, 130)

    # Next
    next_text = font.render("Next:", True, GOLD)
    screen.blit(next_text, (panel_x, 220))
    for i, p in enumerate(next_piece[:5]):
        draw_piece(screen, p, panel_x, 250 + i * 80)

    # Aviso de reinicio en modo "opener"
    if alert_opener:
        alert_font = pygame.font.SysFont("consolas", 30, bold=True)
        alert_text = alert_font.render("RESTARTING", True, (255, 0, 0))
        screen.blit(alert_text, (BOARD_WIDTH * CELL_SIZE // 2 - 100, HEIGHT // 2 - 20))

    pygame.display.flip()

def get_ghost_piece(current_piece, board):
    """Devuelve una copia de la pieza en su posición de caída."""
    ghost = Piece(current_piece.shape_key, BOARD_WIDTH)
    ghost.shape = current_piece.shape
    ghost.x = current_piece.x
    ghost.y = current_piece.y

    # Bajar hasta colisión
    while ghost.can_move(0, 1, board):
        ghost.y += 1
    return ghost

def add_random_layer(env, fill_cell_prob=0.7):
    fila = np.random.choice([0, 1], size=(1, env.board.shape[1]), p=[1 - fill_cell_prob, fill_cell_prob])

    # Si la fila está completamente llena, eliminar un 1 aleatorio
    if np.sum(fila) == env.board.shape[1]:
        idx = random.randint(0, env.board.shape[1] - 1)
        fila[0][idx] = 0

    env.board = np.vstack((env.board[1:], fila))

def generar_tablero_con_piezas(max_altura):
    alto, ancho = 22, 10
    tablero = np.zeros((alto, ancho), dtype=int)
    fila_actual = alto - 1  # Comenzar desde abajo

    while fila_actual >= alto - max_altura:
        intentos = 0
        colocada = False
        while not colocada and intentos < 100:
            pieza_key = random.choice(list(PIECES.keys()))
            forma = np.array(PIECES[pieza_key])
            pieza_alto, pieza_ancho = forma.shape

            # Elegir posición horizontal aleatoria
            if pieza_ancho > ancho:
                continue  # No cabe
            x = random.randint(0, ancho - pieza_ancho)
            y = fila_actual - pieza_alto + 1
            if y < 0:
                break  # No cabe verticalmente

            # Verificar colisión
            sub_tablero = tablero[y:y+pieza_alto, x:x+pieza_ancho]
            if np.any((sub_tablero + forma) > 1):
                intentos += 1
                continue

            # Colocar pieza
            tablero[y:y+pieza_alto, x:x+pieza_ancho] += forma
            colocada = True

        # Verificar si la fila quedó completa, y si es así, eliminar un bloque aleatorio
        for y in range(alto - max_altura, alto):
            if np.sum(tablero[y]) == ancho:
                idx = random.randint(0, ancho - 1)
                tablero[y][idx] = 0  # Quitar un bloque para evitar fila completa

        fila_actual -= 1

    return tablero

def save_transition_to_csv(obs, action, next_obs, reward, done, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, delimiter=';', fieldnames=[
            'board', 'queue', 'piece_type', 'piece_shape',
            'hold_piece', 'piece_position', 'current_piece_n_action',
            'current_piece_rotation', 'action',
            'next_board', 'next_queue', 'next_piece_type', 'next_piece_shape',
            'next_hold_piece', 'next_piece_position', 'next_current_piece_n_action',
            'next_current_piece_rotation',
            'reward', 'done'
        ])
        
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            # Estado actual
            'board': json.dumps(obs['board'].tolist()),
            'queue': json.dumps(obs['queue']),
            'piece_type': obs['piece_type'],
            'piece_shape': json.dumps(obs['piece_shape'].tolist()),
            'hold_piece': obs['hold_piece'],
            'piece_position': json.dumps(obs['piece_position']),
            'current_piece_n_action': obs['current_piece_n_action'],
            'current_piece_rotation': obs['current_piece_rotation'],

            # Acción
            'action': action,

            # Siguiente estado
            'next_board': json.dumps(next_obs['board'].tolist()),
            'next_queue': json.dumps(next_obs['queue']),
            'next_piece_type': next_obs['piece_type'],
            'next_piece_shape': json.dumps(next_obs['piece_shape'].tolist()),
            'next_hold_piece': next_obs['hold_piece'],
            'next_piece_position': json.dumps(next_obs['piece_position']),
            'next_current_piece_n_action': next_obs['current_piece_n_action'],
            'next_current_piece_rotation': next_obs['current_piece_rotation'],

            # Recompensa y si terminó
            'reward': reward,
            'done': done
        })



def main(game_mode="clean", time_limit=120, fill_cell_prob=0.75, gen_tab_height=6, mode_interval=10):
    game_modes = ["clean", "cheese", "opener"]
    mode_interval = mode_interval * 1000
    assert game_mode in game_modes, f"Modo de juego no valido: '{game_mode}'. Debe ser uno de estos {game_modes}"
    # Obtener fecha y hora actual
    now = datetime.now()
    
    # Formatear como string para el nombre de archivo
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    # Crear el nombre del archivo
    filename = f"datasets/tetris_{game_mode}_{time_limit}s_dataset_{timestamp}.csv"
        
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris RL")
    clock = pygame.time.Clock()

    env = TetrisEnv(obs_type="dict")
    obs = env.reset()
    
    if game_mode == "cheese":
        env.board = generar_tablero_con_piezas(gen_tab_height)
        
    running = True
    start_ticks = pygame.time.get_ticks()
    time_limit = time_limit * 1000

    game_mode_timer = pygame.time.get_ticks()
    
    while running:
        clock.tick(FPS)
        
        elapsed_time = pygame.time.get_ticks() - start_ticks

        # GAMEMODES
        if game_mode == "cheese" and pygame.time.get_ticks() - game_mode_timer > mode_interval:
            add_random_layer(env, fill_cell_prob)
            game_mode_timer = pygame.time.get_ticks()

        if game_mode == "opener" and pygame.time.get_ticks() - game_mode_timer > mode_interval:
            env.current_piece = None
            env.reset()
            game_mode_timer = pygame.time.get_ticks()
        
        if elapsed_time > time_limit:
            running = False
        
        time_left = max(0, (time_limit - elapsed_time) / 1000)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                elif event.key == pygame.K_SPACE:
                    action = 4
                elif event.key == pygame.K_c:
                    action = 5
                elif event.key == pygame.K_x:
                    action = 6
                elif event.key == pygame.K_z:
                    action = 7

                if action is not None:
                    obs = env._get_obs()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
            
                    save_transition_to_csv(obs, action, next_obs, reward, done, filename)
            
                    if done:
                        running = False

        
        # Mostrar aviso si faltan menos de 2 segundos para reinicio en modo "opener"
        alert_opener = (game_mode == "opener") and (pygame.time.get_ticks() - game_mode_timer > mode_interval - 2000)
        draw_board(screen, env.board, env.current_piece, env.hold_piece, env.next_piece, env.score, time_left, alert_opener)

        
    pygame.quit()

if __name__ == "__main__":
    path = 'datasets/'
    contador = 0
    for root, dirs, files in os.walk(path):
        contador += len(files)
    cantidad = contador
    print(f"Cantidad de datasets en {path}: {cantidad}")
    main(
        game_mode="cheese", 
        time_limit=60,
        fill_cell_prob=0.8,
        gen_tab_height=7,
        mode_interval=7
    )

