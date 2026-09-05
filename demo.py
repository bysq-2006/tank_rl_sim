from __future__ import annotations

import pygame

from core import TankGame
from renderer import PygameRenderer


def main() -> None:
    """持续试玩：一局判定胜负后积分，再换地图开下一局。"""
    game = TankGame()
    renderer = PygameRenderer(caption="Tank Game  |  R 换图  Esc 退出")
    kills = [0, 0]
    print(f"map: {game.maze.rows} x {game.maze.cols}")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    renderer.reset_fx()
                    print(f"map: {game.maze.rows} x {game.maze.cols}")
        keys = pygame.key.get_pressed()
        throttle = 2 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 1
        steer = 0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 2 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 1
        fire = int(keys[pygame.K_j] or keys[pygame.K_SPACE])
        game.update([(throttle, steer, fire), (1, 1, 0)])
        renderer.draw(game, kills)
        if game.is_over:
            if game.winner is not None:
                kills[game.winner] += 1
            game.reset()
            renderer.reset_fx()
            print(f"map: {game.maze.rows} x {game.maze.cols}")
        renderer.tick(game.physics_hz)
    renderer.close()


if __name__ == "__main__":
    main()
