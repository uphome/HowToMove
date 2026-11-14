from __future__ import annotations

import json

from dataset import (
    MapRecord, build_eval_states, content_hash, load_records, optimal_actions,
    save_records, split_for_hash, state_positions,
)
from env.maze import ACTIONS, generate


def test_content_hash_ignores_wall_order():
    a = content_hash(12, 12, [(1, 2), (3, 4)], (0, 0), (5, 5))
    b = content_hash(12, 12, [(3, 4), (1, 2)], (0, 0), (5, 5))
    assert a == b
    c = content_hash(12, 12, [(1, 2)], (0, 0), (5, 5))
    assert a != c


def test_split_is_deterministic_and_exhaustive():
    import hashlib

    seen = set()
    for i in range(2000):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        assert split_for_hash(h) == split_for_hash(h)
        seen.add(split_for_hash(h))
    assert seen == {"train", "val", "test_id", "test_hard"}


def test_record_roundtrip(tmp_path):
    maze = generate(level="L3", seed=7)
    rec = MapRecord.from_maze(maze, ident="x", seed=7)
    path = tmp_path / "m.jsonl"
    assert save_records([rec], path) == 1
    back = load_records(path)[0]
    assert back == rec
    assert back.to_maze().path_length == maze.path_length
    assert json.loads(json.dumps(rec.to_dict()))["hash"] == rec.hash


def test_optimal_actions_match_bfs():
    maze = generate(level="L4", seed=3)
    pos = maze.start
    acts = optimal_actions(maze, pos)
    assert acts
    d = maze.distance(pos)
    for a in acts:
        dx, dy = ACTIONS[a]
        assert maze.distance((pos[0] + dx, pos[1] + dy)) == d - 1
    for a in set(ACTIONS) - set(acts):
        dx, dy = ACTIONS[a]
        nxt = (pos[0] + dx, pos[1] + dy)
        assert (not maze.is_free(*nxt)) or maze.distance(nxt) != d - 1


def test_state_positions_respects_min_dist():
    import random

    maze = generate(level="L2", seed=1)
    pos = state_positions(maze, 5, random.Random(0), min_dist=4)
    assert pos
    assert all(maze.distance(p) >= 4 for p in pos)


def test_build_eval_states():
    recs = [MapRecord.from_maze(generate(level="L3", seed=s), seed=s)
            for s in range(5)]
    states = build_eval_states(recs, 5)
    assert len(states) == 5
    for s in states:
        assert s["optimal_actions"]
        assert s["state"].startswith("pos=(")
        assert "walls:" in s["state"]
