"""M0 测试：T1（生成不变量） + T5（reward 单测）。"""

from __future__ import annotations

import pytest

from .maze import (
    ACTIONS,
    AGENT,
    GOAL,
    INF,
    LEVEL_RULES,
    LEVELS,
    REWARD_COLLISION,
    REWARD_FORMAT,
    REWARD_STEP,
    REWARD_TERMINAL,
    WALL,
    Maze,
    MazeEnv,
    bfs_field,
    generate,
    manhattan,
)
from .render import STYLES, render_grid, render_observation, render_state

N_SEEDS = 200


# ---------------------------------------------------------------- T1


@pytest.mark.parametrize("level", LEVELS)
def test_generate_invariants(level: str):
    """1000 张随机图：可解、起点终点合法、距离在预算内。"""
    rules = LEVEL_RULES[level]
    for seed in range(N_SEEDS):
        maze = generate(level=level, seed=seed)
        d = maze.distance(maze.start)

        assert d != INF, f"{level} seed={seed} 不可解"
        assert maze.start != maze.goal
        assert maze.start not in maze.walls
        assert maze.goal not in maze.walls
        assert d >= rules["min_dist"], f"{level} seed={seed} 太近: {d}"
        assert d <= maze.max_steps - 2, f"{level} seed={seed} 超预算: {d}"
        assert d == maze.path_length
        assert d - manhattan(maze.start, maze.goal) == maze.detour


@pytest.mark.parametrize("level", LEVELS)
def test_detour_constraint(level: str):
    """detour 必须是偶数，且不低于级别要求。"""
    rules = LEVEL_RULES[level]
    for seed in range(N_SEEDS):
        maze = generate(level=level, seed=seed)
        assert maze.detour % 2 == 0, f"{level} seed={seed} detour={maze.detour} 不是偶数"
        assert maze.detour >= rules["min_detour"], (
            f"{level} seed={seed} detour={maze.detour} < {rules['min_detour']}"
        )


def test_l0_has_no_walls():
    for seed in range(N_SEEDS):
        maze = generate(level="L0", seed=seed)
        assert not maze.walls
        assert maze.detour == 0


@pytest.mark.parametrize("level", ("L1", "L2", "L3", "L4"))
def test_walls_present(level: str):
    for seed in range(50):
        maze = generate(level=level, seed=seed)
        assert maze.walls, f"{level} seed={seed} 没有墙"


def test_generate_deterministic():
    a = generate(level="L3", seed=1234)
    b = generate(level="L3", seed=1234)
    assert a.walls == b.walls
    assert a.start == b.start
    assert a.goal == b.goal
    assert a.dist == b.dist


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        generate(level="L9")


@pytest.mark.parametrize("level", ("L3", "L4"))
def test_u_wall_forces_moving_away(level: str):
    """L3/L4 的最短路必须出现"离目标更远"的一步。

    这是 BFS 距离奖励相对 Manhattan 的唯一价值：如果奖励用曼哈顿，
    这一步会吃负奖励，与任务目标直接对抗。
    """
    checked = 0
    for seed in range(N_SEEDS):
        maze = generate(level=level, seed=seed)
        if maze.detour < 2:
            continue
        mans = [manhattan(p, maze.goal) for p in maze.shortest_path()]
        assert any(b > a for a, b in zip(mans, mans[1:])), (
            f"{level} seed={seed} 最短路没有远离步"
        )
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------- T5


def make_known_maze() -> Maze:
    """
    .....
    ..#..
    ..#..
    ..#..
    .....
    start=(0,2) goal=(4,2)，只能绕上/下走，最短路 8，detour 4（含 2 次远离步）。
    """
    walls = frozenset({(2, 1), (2, 2), (2, 3)})
    goal = (4, 2)
    dist = bfs_field(5, 5, walls, goal)
    return Maze(
        width=5,
        height=5,
        walls=walls,
        start=(0, 2),
        goal=goal,
        max_steps=10,
        dist=dist,
    )


def test_bfs_known_distances():
    maze = make_known_maze()
    assert maze.distance((0, 2)) == 8
    assert maze.distance((1, 2)) == 7
    assert maze.distance((0, 1)) == 7
    assert maze.distance((4, 2)) == 0
    assert maze.distance((2, 2)) == INF
    assert maze.detour == 4
    assert maze.path_length == 8


def test_step_toward_goal():
    env = MazeEnv(make_known_maze())
    r = env.step("right")
    assert env.pos == (1, 2)
    assert r.reward == pytest.approx(1.0 + REWARD_STEP)
    assert r.info["delta"] == 1
    assert not r.done


def test_step_away_from_goal():
    env = MazeEnv(make_known_maze())
    env.pos = (1, 2)
    r = env.step("left")
    assert env.pos == (0, 2)
    assert r.reward == pytest.approx(-1.0 + REWARD_STEP)
    assert r.info["delta"] == -1


def test_collision_wall():
    env = MazeEnv(make_known_maze())
    env.pos = (1, 2)
    r = env.step("right")  # (2,2) 是墙
    assert env.pos == (1, 2)
    assert r.reward == pytest.approx(REWARD_COLLISION + REWARD_STEP)
    assert r.info["reason"] == "collision"
    assert not r.done


def test_collision_out_of_bounds():
    env = MazeEnv(make_known_maze())
    r = env.step("left")  # x=-1
    assert env.pos == (0, 2)
    assert r.reward == pytest.approx(REWARD_COLLISION + REWARD_STEP)
    assert not r.done


def test_reach_goal():
    env = MazeEnv(make_known_maze())
    env.pos = (3, 2)
    r = env.step("right")
    assert env.pos == (4, 2)
    assert r.done and env.terminated
    assert r.info["reason"] == "goal"
    assert r.reward == pytest.approx(1.0 + REWARD_STEP + REWARD_TERMINAL)


def test_bad_format_terminates():
    env = MazeEnv(make_known_maze())
    r = env.step("northeast")
    assert r.done and env.terminated
    assert r.info["reason"] == "bad_format"
    assert r.reward == pytest.approx(REWARD_FORMAT + REWARD_STEP)
    assert env.pos == (0, 2)


def test_truncation():
    maze = make_known_maze()
    env = MazeEnv(Maze(**{**maze.__dict__, "max_steps": 3}))
    r1 = env.step("up")
    r2 = env.step("up")
    r3 = env.step("down")
    assert not r1.done and not r2.done
    assert r3.done and env.truncated
    assert r3.info["reason"] == "truncated"
    assert env.steps == 3


def test_step_after_done_raises():
    env = MazeEnv(make_known_maze())
    env.step("northeast")
    with pytest.raises(RuntimeError):
        env.step("up")


def test_reset():
    env = MazeEnv(make_known_maze())
    env.step("up")
    env.reset()
    assert env.pos == env.maze.start
    assert env.steps == 0
    assert env.last_actions == []
    assert not env.done


def test_reward_is_potential_shaping():
    """所有合法移动：reward == bfs_d(s) - bfs_d(s') - 0.1，delta ∈ {-1,0,1}。"""
    maze = generate(level="L4", seed=7)
    env = MazeEnv(maze)
    for i in range(200):
        if env.done:
            env.reset()
        pos = env.pos
        d0 = maze.distance(pos)
        action = list(ACTIONS)[i % 4]
        dx, dy = ACTIONS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        blocked = not maze.is_free(*nxt)
        r = env.step(action)
        if blocked:
            assert r.reward == pytest.approx(REWARD_COLLISION + REWARD_STEP)
            assert env.pos == pos
        else:
            delta = d0 - maze.distance(nxt)
            assert delta in (-1, 0, 1)
            expected = delta + REWARD_STEP
            if nxt == maze.goal:
                expected += REWARD_TERMINAL
            assert r.reward == pytest.approx(expected)
            assert env.pos == nxt


# ---------------------------------------------------------------- render


def test_render_shape():
    maze = generate(level="L4", seed=3)
    rows = render_grid(maze, maze.start)
    assert len(rows) == maze.height
    assert all(len(row) == maze.width for row in rows)
    flat = "".join(rows)
    assert flat.count(AGENT) == 1
    assert flat.count(GOAL) == 1
    assert flat.count(WALL) == len(maze.walls)


def test_render_state_fields():
    maze = generate(level="L2", seed=5)
    text = render_state(maze, maze.start, ["up", "left"], style="grid")
    lines = text.splitlines()
    assert lines[0] == (f"maze ({maze.width}x{maze.height}, #=wall .=free A=you G=goal,"
                        f" x=right y=down):")
    dx = maze.goal[0] - maze.start[0]
    dy = maze.goal[1] - maze.start[1]
    assert lines[-2] == (f"pos=({maze.start[0]},{maze.start[1]})"
                         f" goal=({maze.goal[0]},{maze.goal[1]})"
                         f" delta=({dx:+d},{dy:+d})")
    assert lines[-1] == "last=up,left"


def test_render_state_no_history():
    maze = generate(level="L0", seed=0)
    text = render_state(maze, maze.start, style="grid")
    assert text.splitlines()[-1] == "last=none"


def test_render_observation_matches_state():
    maze = generate(level="L3", seed=9)
    assert render_observation(maze, maze.start) == render_state(maze, maze.start)


def test_agent_marker_tracks_position():
    maze = generate(level="L1", seed=2)
    env = MazeEnv(maze)
    env.step("up")
    flat = "".join(render_grid(maze, env.pos))
    idx = flat.index(AGENT)
    x, y = idx % maze.width, idx // maze.width
    assert (x, y) == env.pos


@pytest.mark.parametrize("style", STYLES)
def test_render_styles(style: str):
    maze = generate(level="L3", seed=114)
    text = render_state(maze, maze.start, ["up", "left"], style=style)
    if style == "minimal":
        assert text.startswith("pos=(")
        assert "walls:" in text
        return
    if style == "rel":
        assert text.startswith("map 12x12")
        assert "origin=you" in text
        assert "goal: delta=(" in text
        assert "bounds:" in text
        assert "walls_rel:" in text
        assert "pos=(" not in text
        assert "last=" not in text
        return
    assert "pos=(" in text
    assert "delta=(" in text
    assert "last=up,left" in text
    assert "maze" in text.splitlines()[0]
    dx = maze.goal[0] - maze.start[0]
    dy = maze.goal[1] - maze.start[1]
    assert f"delta=({dx:+d},{dy:+d})" in text


def test_render_rel_bounds_and_relative_walls():
    maze = generate(level="L0", seed=0)
    pos = (3, 9)
    text = render_state(maze, pos, ["up"], style="rel")
    dx = maze.goal[0] - pos[0]
    dy = maze.goal[1] - pos[1]
    assert f"goal: delta=({dx:+d},{dy:+d})" in text
    assert "bounds: left=3 right=8 up=9 down=2" in text
    assert "walls_rel: none" in text
    assert "pos=(" not in text

    walled = generate(level="L3", seed=114)
    text = render_state(walled, walled.start, style="rel")
    assert len(walled.walls) > 0
    wx, wy = next(iter(walled.walls))
    rel = (wx - walled.start[0], wy - walled.start[1])
    assert f"({rel[0]:+d},{rel[1]:+d})" in text


def test_render_style_rejects_unknown():
    maze = generate(level="L0", seed=0)
    with pytest.raises(ValueError):
        render_state(maze, maze.start, style="nope")


def test_render_grid_visited_marker():
    maze = generate(level="L0", seed=0)
    text = render_state(maze, maze.start, style="grid",
                        visited=[(maze.start[0] + 1, maze.start[1])])
    assert "," in text
    assert "=visited" in text.splitlines()[0]


def test_memory_zero_records_nothing():
    maze = generate(level="L0", seed=0)
    env = MazeEnv(maze, memory=0)
    env.step("up")
    env.step("down")
    assert env.last_actions == []
