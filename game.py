import os

import pygame
import random
import neat

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define game constants
GRAVITY = 0.65
JUMP_STRENGTH = -13
GROUND_LEVEL = HEIGHT - 100


class Player:
    def __init__(self, color):
        self.x, self.y = 50, GROUND_LEVEL
        self.vel_y = 0
        self.alive = True
        self.color = color

    def update(self):
        self.vel_y += GRAVITY
        self.y += self.vel_y

        # Check if player is below the ground
        if self.y >= GROUND_LEVEL:
            self.y = GROUND_LEVEL
            self.vel_y = 0

    def jump(self):
        if self.y == GROUND_LEVEL:
            self.vel_y = JUMP_STRENGTH

    def check_collision(self, obstacles):
        player_rect = pygame.Rect(self.x, self.y - 30, 30, 30)
        for obstacle in obstacles:
            obstacle_rect = pygame.Rect(
                obstacle.x,
                obstacle.y - obstacle.height,
                obstacle.width,
                GROUND_LEVEL,
            )
            if player_rect.colliderect(obstacle_rect):
                self.alive = False
                return True
        return False


class Obstacle:
    def __init__(self, offset=0):
        self.x = WIDTH + offset
        self.y = GROUND_LEVEL
        self.width = 30
        self.height = random.randint(40, 80)  # Random height for the obstacle

    def update(self):
        self.x -= 5  # Move to the left

    def off_screen(self):
        return self.x < -20


def eval_genomes(genomes, config):
    players = []
    nets = []
    ge = []
    obstacles = [Obstacle(), Obstacle(200), Obstacle(400)]

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initial fitness
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(
            Player(
                (
                    random.randint(0, 75),
                    random.randint(0, 125),
                    random.randint(150, 255),
                    random.uniform(0.5, 1.0),
                )
            )
        )
        ge.append(genome)

    clock = pygame.time.Clock()
    # Game loop
    while len(players) > 0:
        screen.fill((255, 255, 255))

        # Draw the floor
        pygame.draw.rect(
            screen, (0, 255, 0), (0, GROUND_LEVEL, WIDTH, HEIGHT - GROUND_LEVEL)
        )
        # Draw the obstacles
        for obstacle in obstacles:
            if obstacle.off_screen():
                obstacles.remove(obstacle)
                obstacles.append(Obstacle(random.randint(100, 300)))
            obstacle.update()
            pygame.draw.rect(
                screen,
                (255, 0, 0),
                (
                    obstacle.x,
                    obstacle.y - obstacle.height,
                    obstacle.width,
                    GROUND_LEVEL,
                ),
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Manage players
        for i, player in enumerate(players):
            output = nets[i].activate(
                (
                    player.y,
                    obstacles[0].x - player.x if obstacles[0] else WIDTH,
                    obstacles[0].height if obstacles[0] else 0,
                    obstacles[1].x - player.x if obstacles[1] else WIDTH,
                    obstacles[1].height if obstacles[1] else 0,
                    obstacles[2].x - player.x if obstacles[2] else WIDTH,
                    obstacles[2].height if obstacles[2] else 0,
                )
            )
            output = output[0]
            if output > 0.5:
                player.jump()
            player.update()
            ge[i].fitness += 0.1
            if player.check_collision(obstacles):
                ge[i].fitness -= 1
                players.pop(i)
                nets.pop(i)
                ge.pop(i)
            pygame.draw.rect(
                screen,
                player.color[:3] + (int(player.color[3] * 255),),
                (player.x, player.y - 30, 30, 30),
            )

        # If there is only one player left, end the game early and increase its fitness to make it the winner
        if len(players) == 1:
            players[0].alive = False
            ge[0].fitness += 10
            players.pop(0)
            nets.pop(0)
            ge.pop(0)
        pygame.display.update()
        clock.tick(30)


def run_neat(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.run(eval_genomes, 150)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-game")
    run_neat(config_path)
