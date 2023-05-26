import pygame
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

pygame.init()

# Velocity of the players
VELOCITY = 4

# Ball velocity
BALLVELOCITY = [3, 3]

# dimensions of board
WIDTH = 1024
HEIGHT = 616

# dimensions of player
PLAYERHEIGHT = 36
PLAYERWIDTH = 36

# screen refresh rate
FPS = 90

# goal sizes
GOALWIDTH = 53
GOALHEIGHT = 120

# Font Sizes
smallfont = pygame.font.SysFont("comicsansms", 25)
medfont = pygame.font.SysFont("comicsansms", 45)
largefont = pygame.font.SysFont("comicsansms", 65)

# Initialization of pitch
pitch_image = pygame.image.load(os.path.join('Assets', 'pitch.jpg'))
pitch = pygame.transform.scale(pitch_image, (WIDTH, HEIGHT))

# Initialization of player 1
poland_image = pygame.image.load(os.path.join('Assets', 'polishteam.png'))
poland = pygame.transform.scale(poland_image, (PLAYERWIDTH, PLAYERHEIGHT))

# Initialization of player 2
germany_image = pygame.image.load(os.path.join('Assets', 'gremanteam.png'))
germany = pygame.transform.scale(germany_image, (PLAYERWIDTH, PLAYERHEIGHT))

# Initialization of ball
ball_image = pygame.image.load(os.path.join('Assets', 'Ball.png'))
ball = pygame.transform.scale(ball_image, (PLAYERWIDTH, PLAYERHEIGHT))

# Initialization of window
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Soccer Game")

white = (255, 255, 255)

# Objects initialization 
p1 = pygame.Rect(WIDTH / 4 + PLAYERWIDTH / 2, HEIGHT / 2 - PLAYERHEIGHT / 2, PLAYERWIDTH, PLAYERHEIGHT)
p2 = pygame.Rect(WIDTH - WIDTH / 4 + PLAYERWIDTH / 2, HEIGHT / 2 - PLAYERHEIGHT / 2, PLAYERWIDTH, PLAYERHEIGHT)
b = pygame.Rect(WIDTH / 2 + PLAYERWIDTH / 2, HEIGHT / 2 + PLAYERHEIGHT / 2, PLAYERWIDTH, PLAYERHEIGHT)

# Score
score1 = 0
score2 = 0
direction = 1
reward = 0

ACTIONS = ['left', 'right', 'up', 'down']


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = torch.softmax(self.model(state), dim=-1)
        action = torch.multinomial(action_probs, num_samples=1)
        return action.item()

    def update_policy(self, states, actions, rewards):
        self.optimizer.zero_grad()
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        action_probs = torch.softmax(self.model(states), dim=-1)
        selected_action_probs = action_probs.gather(dim=-1, index=actions.unsqueeze(-1))

        loss = -torch.log(selected_action_probs) * rewards
        loss = loss.mean()

        loss.backward()
        self.optimizer.step()


def get_state():
    # Retrieve the positions of p1, p2, and the ball
    p1_position = (p1.x, p1.y)
    p2_position = (p2.x, p2.y)
    ball_position = (b.x, b.y)

    # Retrieve the scores of p1 and p2
    scores = (score1, score2)

    # Return the state as a tuple
    return p1_position, p2_position, ball_position, scores


def perform_action(action):
    if action == 'left' and p1.x - VELOCITY > 0:
        p1.x -= VELOCITY
    elif action == 'right' and p1.x + VELOCITY + PLAYERWIDTH < WIDTH:
        p1.x += VELOCITY
    elif action == 'up' and p1.y - VELOCITY > 0:
        p1.y -= VELOCITY
    elif action == 'down' and p1.y + VELOCITY + PLAYERHEIGHT < HEIGHT - 15:
        p1.y += VELOCITY


def text_objects(text, color, size):
    if size == "small":
        text_surface = smallfont.render(text, True, color)
    elif size == "medium":
        text_surface = medfont.render(text, True, color)
    elif size == "large":
        text_surface = largefont.render(text, True, color)
    return text_surface, text_surface.get_rect()


def p1_handle_movement(keys_pressed):
    if keys_pressed[pygame.K_a] and p1.x - VELOCITY > 0:  # LEFT
        p1.x -= VELOCITY
    if keys_pressed[pygame.K_d] and p1.x + VELOCITY + PLAYERWIDTH < WIDTH:  # RIGHT
        p1.x += VELOCITY
    if keys_pressed[pygame.K_w] and p1.y - VELOCITY > 0:  # UP
        p1.y -= VELOCITY
    if keys_pressed[pygame.K_s] and p1.y + VELOCITY + PLAYERHEIGHT < HEIGHT - 15:  # DOWN
        p1.y += VELOCITY


def p2_handle_movement(keys_pressed):
    if keys_pressed[pygame.K_LEFT] and p2.x - VELOCITY > 0:  # LEFT
        p2.x -= VELOCITY
    if keys_pressed[pygame.K_RIGHT] and p2.x + VELOCITY + PLAYERWIDTH < WIDTH:  # RIGHT
        p2.x += VELOCITY
    if keys_pressed[pygame.K_UP] and p2.y - VELOCITY > 0:  # UP
        p2.y -= VELOCITY
    if keys_pressed[pygame.K_DOWN] and p2.y + VELOCITY + PLAYERHEIGHT < HEIGHT - 15:  # DOWN
        p2.y += VELOCITY


def after_goal():
    global direction
    BALLVELOCITY[0] = BALLVELOCITY[0] * direction
    BALLVELOCITY[1] = BALLVELOCITY[1] * direction
    b.x = win.get_width() / 2
    b.y = win.get_height() / 2


def p1_handle_random():
    move = random.randint(1, 4)
    if move == 1 and p1.x + VELOCITY + PLAYERWIDTH < WIDTH:
        p1.x += VELOCITY
    elif move == 2 and p1.x - VELOCITY > 0:
        p1.x -= VELOCITY
    elif move == 3 and p1.y - VELOCITY > 0:
        p1.y -= VELOCITY
    elif move == 4 and p1.y + VELOCITY + PLAYERHEIGHT < HEIGHT - 15:  # DOWN
        p1.y += VELOCITY


def p2_handle_random():
    move = random.randint(1, 4)
    if move == 1 and p2.x + VELOCITY + PLAYERWIDTH < WIDTH:
        p2.x += VELOCITY
    elif move == 2 and p2.x - VELOCITY > 0:
        p2.x -= VELOCITY
    elif move == 3 and p2.y - VELOCITY > 0:
        p2.y -= VELOCITY
    elif move == 4 and p2.y + VELOCITY + PLAYERHEIGHT < HEIGHT - 15:  # DOWN
        p2.y += VELOCITY


def b_handle_movement():
    global score1, score2, direction, reward
    b.x += BALLVELOCITY[0]
    b.y += BALLVELOCITY[1]

    if b.x <= 0 + GOALWIDTH - PLAYERWIDTH and (HEIGHT / 2 - GOALHEIGHT <= b.y <= HEIGHT / 2 + GOALHEIGHT):
        score1 += 1
        direction = -1
        reward = 10
        after_goal()
    elif (b.x >= WIDTH - GOALWIDTH - b.width) and (HEIGHT / 2 - GOALHEIGHT <= b.y <= HEIGHT / 2 + GOALHEIGHT):
        score2 += 1
        direction = 1
        reward = -10
        after_goal()
    elif b.x < 0 or b.x > win.get_width():
        BALLVELOCITY[0] *= -1
    if b.y - 10 < 0 or b.y + 10 > win.get_height() - b.height:
        BALLVELOCITY[1] *= -1

    if b.colliderect(p1) and p1.y > b.y:
        BALLVELOCITY[0] = abs(BALLVELOCITY[0])  # Change the x-direction of the ball
    elif b.colliderect(p2):
        BALLVELOCITY[0] = -abs(BALLVELOCITY[0])  # Change the x-direction of the ball


def draw_window():
    win.fill(white)
    win.blit(pitch, (0, 0))
    win.blit(ball, (b.x, b.y))
    win.blit(poland, (p1.x, p1.y))
    win.blit(germany, (p2.x, p2.y))
    message_to_screen("Germany", white, -250, -150, "small")
    message_to_screen(str(score1), white, -200, -150, "small")
    message_to_screen("Poland", white, -250, 150, "small")
    message_to_screen(str(score2), white, -200, 150, "small")
    pygame.display.update()


def message_to_screen(msg, color, y_displace=0, x_displace=0, size="small"):
    text_surf, text_rect = text_objects(msg, color, size)
    text_rect.center = (win.get_width() / 2 + x_displace), ((win.get_height() / 2) + y_displace)
    win.blit(text_surf, text_rect)


def main():
    # state_size = 8  # Modify this based on the number of state features
    # action_size = 4  # Modify this based on the number of available actions
    # agent = Agent(state_size, action_size)
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        # RL Agent chooses action
        # state = get_state()
        # action = agent.choose_action(state)
        # Perform action and update RL Agent's policy
        # perform_action(ACTIONS[action])
        # agent.update_policy([state], [action], [reward])
        keys_pressed = pygame.key.get_pressed()
        p1_handle_movement(keys_pressed)
        p2_handle_movement(keys_pressed)
        # p1_handle_random()
        # p2_handle_random()
        b_handle_movement()
        draw_window()
    pygame.quit()


if __name__ == "__main__":
    main()
