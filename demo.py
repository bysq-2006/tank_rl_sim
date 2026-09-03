from __future__ import annotations

import pygame

from core import TankGame
from renderer import PygameRenderer


def main() -> None:
    """运行键盘控制、核心更新和画面显示组成的试玩循环。"""
    game = TankGame()
    renderer = PygameRenderer()
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
                    print(f"map: {game.maze.rows} x {game.maze.cols}")
        keys = pygame.key.get_pressed()
        # W/S 或上下方向键固定控制前进和后退。
        throttle = 2 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 1
        # A/D 或左右方向键固定控制角速度方向。
        steer = 0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 2 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 1
        # 按住 J 或空格持续请求发射，实际射速仍由核心冷却时间限制。
        fire = int(keys[pygame.K_j] or keys[pygame.K_SPACE])
        was_over = game.is_over
        # 演示只控制 0 号坦克；1 号坦克使用外部传入的静止控制，核心不会替它决策。
        game.update([(throttle, steer, fire), (1, 1, 0)])
        if game.is_over and not was_over:
            print(f"game over; winner={game.winner}; press R to reset")
        renderer.draw(game)
        renderer.tick(game.physics_hz)
    renderer.close()


if __name__ == "__main__":
    main()
