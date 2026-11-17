import pygame
import random
import heapq
import asyncio

pygame.init()
pygame.font.init()

WIDTH = 600
HEIGHT = 600
GRID_SIZE = 29
CELL_SIZE = WIDTH // GRID_SIZE

COLOR_BG = (20, 20, 30)
COLOR_WALL = (235, 100, 20)
COLOR_PATH = (0, 100, 200) 
COLOR_PLAYER = (46, 196, 182)
COLOR_GHOST = (100, 100, 100)
COLOR_BOSS = (102, 0, 0)
CANDY_COLOR = (255, 165, 0)
COLOR_PREDICT = (255, 0, 255) 
COLOR_TRAP = (200, 40, 40)
LOSETEXT = (255, 0, 0)
WINTEXT = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Haunted Maze Game")
clock = pygame.time.Clock()


def generate_maze(size):
    maze = []
    for row in range(size):
        maze.append([1] * size)

    stack = [(1, 1)]
    maze[1][1] = 0

    while len(stack) > 0:
        current_x = stack[-1][0]
        current_y = stack[-1][1]

        neighbors = []
        for step in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            next_x = current_x + step[0]
            next_y = current_y + step[1]
            if next_x > 0 and next_x < size - 1 and next_y > 0 and next_y < size - 1:
                if maze[next_y][next_x] == 1:
                    neighbors.append((next_x, next_y))

        if len(neighbors) > 0:
            chosen = random.choice(neighbors)
            next_x = chosen[0]
            next_y = chosen[1]
            wall_x = (current_x + next_x) // 2
            wall_y = (current_y + next_y) // 2
            maze[wall_y][wall_x] = 0
            maze[next_y][next_x] = 0
            stack.append((next_x, next_y))
        else:
            stack.pop() 

    BREAK_CHANCE = 0.4
    BREAK_LIMIT = 50
    broken = 0

    while broken < BREAK_LIMIT:
        row = random.randint(1, size - 2)
        col = random.randint(1, size - 2)
        roll = random.random()
        path_count = 0
        for step in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if maze[row + step[0]][col + step[1]] == 0:
                path_count += 1

        if roll <= BREAK_CHANCE and maze[row][col] == 1 and path_count == 2:
            maze[row][col] = 0
            broken += 1

    return maze

def is_open(maze, x, y):
    if x < 0 or x >= GRID_SIZE:
        return False
    if y < 0 or y >= GRID_SIZE:
        return False
    if maze[y][x] == 1:
        return False
    return True


def cell_center(cell):
    center_x = cell[0] * CELL_SIZE + CELL_SIZE // 2
    center_y = cell[1] * CELL_SIZE + CELL_SIZE // 2
    return (center_x, center_y)


def can_see(maze, from_pos, target_pos, max_distance):
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for distance in range(1, max_distance + 1):
            look_x = from_pos[0] + direction[0] * distance
            look_y = from_pos[1] + direction[1] * distance

            if not is_open(maze, look_x, look_y):
                break 
            if [look_x, look_y] == target_pos:
                return True
    return False


def heuristic(a, b):

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_astar_path(maze, start, target):
    if start == target:
        return [start]

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    came_from[start] = None
    cost_so_far = {}
    cost_so_far[start] = 0

    while len(frontier) > 0:
        priority, current = heapq.heappop(frontier)

        if current == target:
            break

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_x = current[0] + direction[0]
            next_y = current[1] + direction[1]
            next_node = (next_x, next_y)

            if is_open(maze, next_x, next_y):
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(target, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

    if target not in came_from:
        return []

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


class MarkovPredictor:
    def __init__(self):
        self.counts = {}
        self.last_move = None

    def update(self, move):
        if self.last_move is not None:
            if self.last_move not in self.counts:
                self.counts[self.last_move] = {}
            if move not in self.counts[self.last_move]:
                self.counts[self.last_move][move] = 0
            self.counts[self.last_move][move] += 1
        self.last_move = move

    def most_likely(self, previous):
        if previous not in self.counts:
            return None, 0

        total = 0
        best_move = None
        best_count = 0
        for move in self.counts[previous]:
            count = self.counts[previous][move]
            total += count
            if count > best_count:
                best_count = count
                best_move = move
        return best_move, best_count / total

    def predict(self):
        move = self.most_likely(self.last_move)
        return move

    def predict_sequence(self, steps):
        result = []
        previous = self.last_move
        confidence = 1.0
        for i in range(steps):
            move, probability = self.most_likely(previous)
            if move is None:
                break
            confidence = confidence * probability
            result.append((move, confidence))
            previous = move
        return result



class Player:
    def __init__(self):
        self.pos = [GRID_SIZE - 2, GRID_SIZE - 2]
        self.move_cooldown = 150
        self.last_move_time = pygame.time.get_ticks()
        self.score = 0
        self.markov = MarkovPredictor()
        self.markov_correct = 0
        self.markov_total = 0
        self.last_prediction = None

    def update(self, keys, maze, now):
        move_x = 0
        move_y = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            move_y = -1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            move_y = 1
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move_x = -1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move_x = 1

        if move_x == 0 and move_y == 0:
            return False
        if now - self.last_move_time <= self.move_cooldown:
            return False

        new_x = self.pos[0] + move_x
        new_y = self.pos[1] + move_y
        if not is_open(maze, new_x, new_y):
            return False

        self.pos = [new_x, new_y]
        self.last_move_time = now

        move = (move_x, move_y)
        if self.last_prediction is not None:
            self.markov_total += 1
            if self.last_prediction == move:
                self.markov_correct += 1
        self.markov.update(move)
        self.last_prediction = self.markov.predict()
        return True

    def predicted_positions(self, maze, steps=8):
        pos = [self.pos[0], self.pos[1]]
        result = []
        for item in self.markov.predict_sequence(steps):
            move = item[0]
            confidence = item[1]
            new_x = pos[0] + move[0]
            new_y = pos[1] + move[1]
            if not is_open(maze, new_x, new_y):
                break
            pos = [new_x, new_y]
            result.append(((new_x, new_y), confidence))
        return result

    def draw(self, surface):
        rect = pygame.Rect(self.pos[0] * CELL_SIZE + 4, self.pos[1] * CELL_SIZE + 4,
                           CELL_SIZE - 8, CELL_SIZE - 8)
        pygame.draw.rect(surface, COLOR_PLAYER, rect, border_radius=4)

class Boss:
    LOOKAHEAD = 8

    def __init__(self):
        self.pos = [1, 1]
        self.dir = (1, 0)
        self.sightline = 10
        self.move_cooldown = 150
        self.last_move_time = pygame.time.get_ticks()
        self.state = "CHASE"
        self.target = None
        self.path = []
        self.predicted = []
        self.best_score = 0

    def choose_target(self, maze, player):
        self.predicted = player.predicted_positions(maze, self.LOOKAHEAD)


        options = []
        options.append((tuple(player.pos), 1.0, 0))
        steps_ahead = 1
        for item in self.predicted:
            options.append((item[0], item[1], steps_ahead))
            steps_ahead += 1

        best_score = None
        best_target = None
        best_path = []

        for option in options:
            cell = option[0]
            confidence = option[1]
            steps_ahead = option[2]

            path = get_astar_path(maze, tuple(self.pos), cell)
            if len(path) == 0:
                continue
            boss_steps = len(path) - 1

            score = confidence * 10 - boss_steps

            if steps_ahead > 0:
                score = score + 1.5 * steps_ahead

                boss_time = boss_steps * self.move_cooldown
                player_time = steps_ahead * player.move_cooldown
                if boss_time <= player_time:
                    score = score + 8
                else:
                    score = score - 8

            if best_score is None or score > best_score:
                best_score = score
                best_target = cell
                best_path = path

        if best_score is None:
            best_score = 0
        self.best_score = best_score
        return best_target, best_path

    def update(self, maze, player, now):

        if can_see(maze, self.pos, player.pos, self.sightline):
            cooldown = self.move_cooldown
        else:
            cooldown = self.move_cooldown * 2

        if now - self.last_move_time <= cooldown:
            return

        self.target, self.path = self.choose_target(maze, player)

        if self.target == tuple(player.pos):
            self.state = "CHASE"
        else:
            self.state = "INTERCEPT"

        if len(self.path) > 1:
            next_cell = self.path[1]
            self.dir = (next_cell[0] - self.pos[0], next_cell[1] - self.pos[1])
            self.pos = [next_cell[0], next_cell[1]]
        self.last_move_time = now

    def draw(self, surface):
        center_x = self.pos[0] * CELL_SIZE + CELL_SIZE // 2
        center_y = self.pos[1] * CELL_SIZE + CELL_SIZE // 2
        size = CELL_SIZE // 2 - 2
        half = size // 2
        pygame.draw.circle(surface, COLOR_BOSS, (center_x, center_y), size)


        forward_x = self.dir[0]
        forward_y = self.dir[1]
        side_x = -forward_y
        side_y = forward_x
        mouth = [
            (center_x, center_y),
            (center_x + forward_x * size + side_x * half, center_y + forward_y * size + side_y * half),
            (center_x + forward_x * size - side_x * half, center_y + forward_y * size - side_y * half),
        ]
        pygame.draw.polygon(surface, COLOR_BG, mouth)

    def draw_debug(self, surface, font):

        for cell in self.path:
            pygame.draw.circle(surface, COLOR_PATH, cell_center(cell), CELL_SIZE // 6)
        for item in self.predicted:
            pygame.draw.circle(surface, COLOR_PREDICT, cell_center(item[0]), CELL_SIZE // 4, 2)
        if self.target is not None:
            pygame.draw.circle(surface, COLOR_BOSS, cell_center(self.target), CELL_SIZE // 2, 2)

        lines = []
        lines.append(f"Boss state: {self.state}")
        lines.append(f"Target: {self.target}  score: {self.best_score:.1f}")
        lines.append(f"Predicted steps: {len(self.predicted)}")
        y = 22
        for text in lines:
            surface.blit(font.render(text, True, (255, 255, 255)), (5, y))
            y += 16


class NPC:
    PATROL_DELAY = 320
    SEARCH_DELAY = 260
    CHASE_DELAY = 260
    RETREAT_DELAY = 180

    SEARCH_TIME = 2500 
    RETREAT_TIME = 4000

    def __init__(self):
        self.pos = [GRID_SIZE - 2, 1]
        self.dir = (-1, 0)
        self.sightline = 6
        self.last_move_time = pygame.time.get_ticks()
        self.state = "PATROL"
        self.target = None
        self.path = []
        self.last_seen = None
        self.search_until = None
        self.retreat_until = 0

    def get_delay(self):
        if self.state == "PATROL":
            return self.PATROL_DELAY
        elif self.state == "SEARCH":
            return self.SEARCH_DELAY
        elif self.state == "CHASE":
            return self.CHASE_DELAY
        else:
            return self.RETREAT_DELAY

    def random_waypoint(self, maze):
        while True:
            x = random.randint(1, GRID_SIZE - 2)
            y = random.randint(1, GRID_SIZE - 2)
            if maze[y][x] == 0 and [x, y] != self.pos:
                return (x, y)

    def far_waypoint(self, maze, from_pos):
        best_cell = None
        best_distance = -1
        for i in range(40):
            cell = self.random_waypoint(maze)
            distance = heuristic(cell, from_pos)
            if distance > best_distance:
                best_cell = cell
                best_distance = distance
        return best_cell

    def start_retreat(self, maze, player, now):
        self.state = "RETREAT"
        self.retreat_until = now + self.RETREAT_TIME
        self.target = self.far_waypoint(maze, player.pos)
        self.path = []

    def update(self, maze, player, now):
        if self.state == "RETREAT":
            if now >= self.retreat_until:
                self.state = "PATROL"
                self.target = None
        elif can_see(maze, self.pos, player.pos, self.sightline):
            self.last_seen = tuple(player.pos)
            self.state = "CHASE"
        elif self.state == "CHASE":
            self.state = "SEARCH"
            self.search_until = None


        if self.state == "RETREAT":
            if self.target is None or tuple(self.pos) == self.target:
                self.target = self.far_waypoint(maze, player.pos)

        elif self.state == "CHASE":
            self.target = tuple(player.pos)

        elif self.state == "SEARCH":
            self.target = self.last_seen
            if tuple(self.pos) == self.target:
                if self.search_until is None:
                    self.search_until = now + self.SEARCH_TIME
                elif now >= self.search_until:
                    self.state = "PATROL"
                    self.target = None

        if self.state == "PATROL":
            if self.target is None or tuple(self.pos) == self.target:
                self.target = self.random_waypoint(maze)


        if self.target is None:
            return
        if now - self.last_move_time <= self.get_delay():
            return

        self.path = get_astar_path(maze, tuple(self.pos), self.target)
        if len(self.path) > 1:
            next_cell = self.path[1]
            self.dir = (next_cell[0] - self.pos[0], next_cell[1] - self.pos[1])
            self.pos = [next_cell[0], next_cell[1]]
        self.last_move_time = now

    def draw(self, surface):
        center_x = self.pos[0] * CELL_SIZE + CELL_SIZE // 2
        center_y = self.pos[1] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 3
        pygame.draw.circle(surface, COLOR_GHOST, (center_x, center_y), radius)

        for side in [-1, 1]:
            eye_x = center_x + self.dir[0] * 3 + (-self.dir[1]) * side * 4
            eye_y = center_y + self.dir[1] * 3 + self.dir[0] * side * 4
            pygame.draw.circle(surface, (255, 255, 255), (eye_x, eye_y), 3)

    def draw_debug(self, surface, font):
        for cell in self.path:
            pygame.draw.circle(surface, (255, 150, 150), cell_center(cell), CELL_SIZE // 8)
        if self.target is not None:
            pygame.draw.circle(surface, COLOR_GHOST, cell_center(self.target), CELL_SIZE // 2, 2)

        path_length = len(self.path) - 1
        if path_length < 0:
            path_length = 0

        lines = []
        lines.append(f"NPC state: {self.state}")
        lines.append(f"NPC target: {self.target}")
        lines.append(f"NPC path length: {path_length}")
        y = 74
        for text in lines:
            surface.blit(font.render(text, True, (255, 150, 150)), (5, y))
            y += 16



def spawn_items(maze, candy_pct, trap_pct, entities):
    candies = []
    traps = []

    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] != 0:
                continue

            cell = [c, r]
            if cell in entities:
                continue


            near_start = False
            for e in entities:
                if heuristic(cell, e) <= 4:
                    near_start = True

            roll = random.random() * 100
            if roll < candy_pct:
                candies.append(cell)
            elif roll < candy_pct + trap_pct and not near_start:
                traps.append(cell)


    if len(candies) == 0:
        open_cells = []
        for r in range(len(maze)):
            for c in range(len(maze[0])):
                cell = [c, r]
                if maze[r][c] == 0 and cell not in entities and cell not in traps:
                    open_cells.append(cell)
        candies.append(random.choice(open_cells))

    return candies, traps


def respawn_candy(maze, candylist, traps, ghost_pos, player_pos):
    open_cells = []
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            cell = [c, r]
            if maze[r][c] == 0 and cell not in candylist and cell not in traps \
               and cell != player_pos and cell != ghost_pos:
                open_cells.append(cell)

    if len(open_cells) > 0:
        candylist.append(random.choice(open_cells))


def new_level(candy_pct, trap_pct):
    maze = generate_maze(GRID_SIZE)
    player = Player()
    boss = Boss()
    npc = NPC()

    for thing in [player, boss, npc]:
        maze[thing.pos[1]][thing.pos[0]] = 0

    candylist, traps = spawn_items(maze, candy_pct, trap_pct,
                                   [player.pos, boss.pos, npc.pos])
    return maze, player, boss, npc, candylist, traps



def draw_menu(surface, font, title_font, settings, index):
    surface.fill(COLOR_BG)
    title = title_font.render("Haunted Maze Game", True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(WIDTH // 2, 120)))

    rows = []
    rows.append(f"Candy chance:  < {settings['candy']}% >")
    rows.append(f"Trap chance:  < {settings['trap']}% >")
    for i in range(len(rows)):
        if i == index:
            color = (255, 220, 0)
        else:
            color = (255, 255, 255)
        text = font.render(rows[i], True, color)
        surface.blit(text, text.get_rect(center=(WIDTH // 2, 240 + i * 50)))

    hints = []
    hints.append("UP/DOWN: select    LEFT/RIGHT: change")
    hints.append("F2 for controls    F1 for AI debug")
    hints.append("ENTER: start    ESC: quit")
    for i in range(len(hints)):
        text = font.render(hints[i], True, (160, 160, 180))
        surface.blit(text, text.get_rect(center=(WIDTH // 2, 380 + i * 30)))


async def main():
    font = pygame.font.Font(None, 30)
    end_font = pygame.font.Font(None, 60)
    debug_font = pygame.font.Font(None, 20)


    scene = "MENU"
    settings = {"candy": 4, "trap": 2}
    max_values = {"candy": 15, "trap": 6}
    setting_names = ["candy", "trap"]
    menu_index = 0

    game_over = False
    win = False
    debug = False
    show_controls = False
    end_score = 30
    message = ""
    message_until = 0

    maze, player, boss, npc, candylist, traps = new_level(settings["candy"], settings["trap"])

    running = True
    while running:
        current_time = pygame.time.get_ticks()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif scene == "MENU":
                    name = setting_names[menu_index]
                    if event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                        menu_index = (menu_index + 1) % len(setting_names)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        settings[name] = settings[name] - 1
                        if settings[name] < 0:
                            settings[name] = 0
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        settings[name] = settings[name] + 1
                        if settings[name] > max_values[name]:
                            settings[name] = max_values[name]
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        maze, player, boss, npc, candylist, traps = new_level(
                            settings["candy"], settings["trap"])
                        game_over = False
                        win = False
                        scene = "PLAY"

                else:
                    if event.key == pygame.K_F1:
                        debug = not debug
                    elif event.key == pygame.K_F2:
                        show_controls = not show_controls
                    elif event.key == pygame.K_r:
                        maze, player, boss, npc, candylist, traps = new_level(
                            settings["candy"], settings["trap"])
                        game_over = False
                        win = False
                    elif event.key == pygame.K_m:
                        scene = "MENU"

        if scene == "MENU":
            draw_menu(screen, font, end_font, settings, menu_index)
            pygame.display.flip()
            clock.tick(30)
            await asyncio.sleep(0)
            continue


        if not game_over and not win:
            keys = pygame.key.get_pressed()

            if player.update(keys, maze, current_time):

                if player.pos in candylist:
                    candylist.remove(player.pos)
                    player.score = player.score + 1
                    respawn_candy(maze, candylist, traps, boss.pos, player.pos)

                if player.pos in traps:
                    player.score = player.score - 2
                    if player.score < 0:
                        player.score = 0
                    message = "Trap! -2"
                    message_until = current_time + 1000

            boss.update(maze, player, current_time)
            npc.update(maze, player, current_time)

            if player.pos == boss.pos:
                game_over = True
            elif player.score >= end_score:
                win = True

            if player.pos == npc.pos and npc.state != "RETREAT":
                player.score = player.score - 1
                if player.score < 0:
                    player.score = 0
                message = "Caught by NPC! -1"
                message_until = current_time + 1000
                npc.start_retreat(maze, player, current_time)


        screen.fill(COLOR_BG)


        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if maze[row][col] == 1:
                    wall_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, COLOR_WALL, wall_rect)


        for candy in candylist:
            pygame.draw.circle(screen, CANDY_COLOR, cell_center(candy), CELL_SIZE // 6)

        for trap in traps:
            x = trap[0] * CELL_SIZE
            y = trap[1] * CELL_SIZE
            pygame.draw.line(screen, COLOR_TRAP, (x + 4, y + 4), (x + CELL_SIZE - 4, y + CELL_SIZE - 4), 3)
            pygame.draw.line(screen, COLOR_TRAP, (x + CELL_SIZE - 4, y + 4), (x + 4, y + CELL_SIZE - 4), 3)

        if debug:
            if player.markov_total > 0:
                accuracy = player.markov_correct / player.markov_total * 100
            else:
                accuracy = 0
            text = f"Markov accuracy: {accuracy:.1f}% ({player.markov_correct}/{player.markov_total})"
            screen.blit(debug_font.render(text, True, (255, 255, 255)), (5, 5))

            boss.draw_debug(screen, debug_font)
            npc.draw_debug(screen, debug_font)

            text = (f"Generated: {len(candylist)} candy ({settings['candy']}%), "
                    f"{len(traps)} traps ({settings['trap']}%)")
            screen.blit(debug_font.render(text, True, (255, 255, 255)), (5, 126))

        npc.draw(screen)
        boss.draw(screen)
        player.draw(screen)


        score_text = f"Score: {player.score}/{end_score}"
        screen.blit(debug_font.render(score_text, True, (255, 255, 255)), (5, 583))
        if current_time < message_until:
            message_surface = font.render(message, True, (255, 255, 255))
            screen.blit(message_surface, message_surface.get_rect(center=(WIDTH // 2, 12)))


        if game_over or win:
            if game_over:
                title = "YOU DIED!"
                color = LOSETEXT
            else:
                title = "YOU WIN!"
                color = WINTEXT

            title_surface = end_font.render(title, True, color)
            screen.blit(title_surface, title_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

            score_surface = font.render(f"Score: {player.score}", True, (255, 255, 255))
            screen.blit(score_surface, score_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))

            help_surface = font.render("R: restart    M: menu    ESC: quit", True, (200, 200, 200))
            screen.blit(help_surface, help_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 90)))

        if show_controls:
            panel = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 200))
            screen.blit(panel, (0, 0))

            lines = ["CONTROLS",
                     "WASD / Arrows : move",
                     "F1 : AI debug",
                     "F2 : show/hide this help",
                     "R : new maze / restart",
                     "M : back to menu",
                     "ESC : quit"]
            for i in range(len(lines)):
                if i == 0:
                    line_font = end_font
                else:
                    line_font = font
                line_surface = line_font.render(lines[i], True, (255, 255, 255))
                screen.blit(line_surface, line_surface.get_rect(center=(WIDTH // 2, 120 + i * 50)))

        pygame.display.flip()
        clock.tick(30)
        await asyncio.sleep(0)

    pygame.quit()


asyncio.run(main())