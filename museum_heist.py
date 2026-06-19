from __future__ import annotations

import heapq
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import variables as cfg


Cell = Tuple[int, int]
Wall = frozenset[Cell]


def wall(a: Cell, b: Cell) -> Wall:
    return frozenset((a, b))


@dataclass
class Topology:
    name: str
    rows: int
    cols: int
    walls: Set[Wall] = field(default_factory=set)
    description: str = ""

    @property
    def n_rooms(self) -> int:
        return self.rows * self.cols

    def rooms(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def index(self, cell: Cell) -> int:
        r, c = cell
        return r * self.cols + c

    def cell(self, idx: int) -> Cell:
        return divmod(idx, self.cols)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def connected(self, a: Cell, b: Cell) -> bool:
        return wall(a, b) not in self.walls

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            other
            for other in candidates
            if self.in_bounds(other) and self.connected(cell, other)
        ]

    def is_connected_graph(self) -> bool:
        rooms = self.rooms()
        if not rooms:
            return True
        seen = {rooms[0]}
        stack = [rooms[0]]
        while stack:
            current = stack.pop()
            for nb in self.neighbors(current):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(rooms)

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "description": self.description,
            "walls": [
                [list(a), list(b)]
                for a, b in sorted(
                    (tuple(sorted(w)) for w in self.walls),
                    key=lambda pair: (pair[0], pair[1]),
                )
            ],
        }

    @staticmethod
    def from_json(data: Dict[str, object]) -> "Topology":
        walls = set()
        for pair in data.get("walls", []):
            a_raw, b_raw = pair
            walls.add(wall(tuple(a_raw), tuple(b_raw)))  # type: ignore[arg-type]
        return Topology(
            name=str(data.get("name", "custom")),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            walls=walls,
            description=str(data.get("description", "")),
        )


def add_vertical_barrier(
    walls: Set[Wall], rows: int, col_left: int, door_rows: Iterable[int]
) -> None:
    doors = set(door_rows)
    for r in range(rows):
        if r not in doors:
            walls.add(wall((r, col_left), (r, col_left + 1)))


def add_horizontal_barrier(
    walls: Set[Wall], cols: int, row_top: int, door_cols: Iterable[int]
) -> None:
    doors = set(door_cols)
    for c in range(cols):
        if c not in doors:
            walls.add(wall((row_top, c), (row_top + 1, c)))


def build_topologies() -> Dict[str, Topology]:
    open4 = Topology(
        name="open4x4",
        rows=4,
        cols=4,
        description="Open 4x4 museum with no internal walls.",
    )

    gallery_walls: Set[Wall] = set()
    add_vertical_barrier(gallery_walls, rows=5, col_left=1, door_rows=[0, 3])
    add_vertical_barrier(gallery_walls, rows=5, col_left=3, door_rows=[1, 4])
    add_horizontal_barrier(gallery_walls, cols=5, row_top=1, door_cols=[1, 4])
    add_horizontal_barrier(gallery_walls, cols=5, row_top=3, door_cols=[0, 2])
    gallery = Topology(
        name="gallery5x5",
        rows=5,
        cols=5,
        walls=gallery_walls,
        description="Five-by-five gallery with separated exhibition areas and doors.",
    )

    maze_walls: Set[Wall] = set()
    add_vertical_barrier(maze_walls, rows=6, col_left=0, door_rows=[0, 2, 4])
    add_vertical_barrier(maze_walls, rows=6, col_left=2, door_rows=[1, 3, 5])
    add_vertical_barrier(maze_walls, rows=6, col_left=4, door_rows=[0, 2, 4])
    add_horizontal_barrier(maze_walls, cols=6, row_top=2, door_cols=[0, 2, 3, 5])
    add_horizontal_barrier(maze_walls, cols=6, row_top=4, door_cols=[0, 1, 3, 4, 5])
    maze = Topology(
        name="maze6x6",
        rows=6,
        cols=6,
        walls=maze_walls,
        description="Six-by-six museum with corridors and choke points.",
    )

    topologies = {top.name: top for top in [open4, gallery, maze]}
    for top in topologies.values():
        if not top.is_connected_graph():
            raise RuntimeError(f"Topology {top.name} is not connected")
    return topologies


TOPOLOGIES = build_topologies()


@dataclass
class HeistConfig:
    beta: float = 4.0
    max_steps: int = 120
    reward_catch: float = 1.0
    reward_escape: float = -1.0
    reward_timeout: float = -0.35
    step_penalty: float = -0.002
    unseen_safe_age: int = 24


@dataclass
class StepInfo:
    reward: float
    done: bool
    outcome: Optional[str]
    selected: Cell
    thief_before: Cell
    thief_after: Cell
    stolen: bool
    event: str = ""


class MuseumHeistEnv:
    def __init__(
        self,
        topology: Topology,
        config: HeistConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.topology = topology
        self.config = config
        self.rng = rng or random.Random()
        self.last_selected: List[int] = []
        self.start: Cell = (0, 0)
        self.target: Cell = (0, 0)
        self.thief_pos: Cell = (0, 0)
        self.stolen = False
        self.step_count = 0
        self.done = False
        self.outcome: Optional[str] = None
        self.active_camera: Optional[Cell] = None

    def reset(self, start: Optional[Cell] = None, target: Optional[Cell] = None) -> None:
        rooms = self.topology.rooms()
        if start is None:
            start = self.rng.choice(rooms)
        if target is None:
            candidates = [cell for cell in rooms if cell != start]
            target = self.rng.choice(candidates)
        if start == target:
            raise ValueError("start and target must be different rooms")

        self.start = start
        self.target = target
        self.thief_pos = start
        self.stolen = False
        self.step_count = 0
        self.done = False
        self.outcome = None
        self.active_camera = None
        self.last_selected = [-10_000 for _ in range(self.topology.n_rooms)]

    def rounds_since_selected(self, cell: Cell) -> int:
        idx = self.topology.index(cell)
        last = self.last_selected[idx]
        if last < 0:
            return self.config.unseen_safe_age + self.step_count
        return max(1, self.step_count - last + 1)

    def move_cost(self, destination: Cell) -> float:
        n = self.rounds_since_selected(destination)
        return 1.0 + self.config.beta * (2.0 ** (-(n - 1)))

    def current_goal(self) -> Cell:
        return self.start if self.stolen else self.target

    def thief_next_cell(self) -> Cell:
        goal = self.current_goal()
        if self.thief_pos == goal:
            return self.thief_pos

        dist: Dict[Cell, float] = {self.thief_pos: 0.0}
        parent: Dict[Cell, Cell] = {}
        heap: List[Tuple[float, float, Cell]] = [(0.0, self.rng.random(), self.thief_pos)]

        while heap:
            cost, _, cell = heapq.heappop(heap)
            if cost > dist.get(cell, math.inf) + 1e-12:
                continue
            if cell == goal:
                break

            neighbors = self.topology.neighbors(cell)
            self.rng.shuffle(neighbors)
            for nb in neighbors:
                new_cost = cost + self.move_cost(nb)
                old = dist.get(nb, math.inf)
                if new_cost + 1e-12 < old:
                    dist[nb] = new_cost
                    parent[nb] = cell
                    heapq.heappush(heap, (new_cost, self.rng.random(), nb))
                elif abs(new_cost - old) <= 1e-12 and self.rng.random() < 0.5:
                    parent[nb] = cell

        if goal not in parent and goal != self.thief_pos:
            return self.thief_pos

        cell = goal
        while parent.get(cell) != self.thief_pos:
            if cell not in parent:
                return self.thief_pos
            cell = parent[cell]
        return cell

    def step(self, action_idx: int) -> StepInfo:
        if self.done:
            raise RuntimeError("Cannot step an episode that has already ended")
        if not 0 <= action_idx < self.topology.n_rooms:
            raise ValueError(f"Invalid action index {action_idx}")

        selected = self.topology.cell(action_idx)
        thief_before = self.thief_pos
        self.step_count += 1
        self.active_camera = selected
        self.last_selected[action_idx] = self.step_count

        if selected == self.thief_pos:
            return self._finish(
                reward=self.config.reward_catch + self.time_bonus(),
                outcome="caught",
                selected=selected,
                thief_before=thief_before,
                thief_after=self.thief_pos,
                event="camera caught thief",
            )

        if self.stolen and selected == self.target:
            return self._finish(
                reward=0.0,
                outcome="detected",
                selected=selected,
                thief_before=thief_before,
                thief_after=self.thief_pos,
                event="stolen painting detected",
            )

        self.thief_pos = self.thief_next_cell()
        event = ""

        if selected == self.thief_pos:
            return self._finish(
                reward=self.config.reward_catch + self.time_bonus(),
                outcome="caught",
                selected=selected,
                thief_before=thief_before,
                thief_after=self.thief_pos,
                event="thief entered active camera room",
            )

        if not self.stolen and self.thief_pos == self.target:
            self.stolen = True
            event = "painting stolen"
        elif self.stolen and self.thief_pos == self.start:
            return self._finish(
                reward=self.config.reward_escape,
                outcome="escaped",
                selected=selected,
                thief_before=thief_before,
                thief_after=self.thief_pos,
                event="thief escaped",
            )

        if self.step_count >= self.config.max_steps:
            return self._finish(
                reward=self.config.reward_timeout,
                outcome="timeout",
                selected=selected,
                thief_before=thief_before,
                thief_after=self.thief_pos,
                event="timeout",
            )

        return StepInfo(
            reward=self.config.step_penalty,
            done=False,
            outcome=None,
            selected=selected,
            thief_before=thief_before,
            thief_after=self.thief_pos,
            stolen=self.stolen,
            event=event,
        )

    def time_bonus(self) -> float:
        remaining = max(0, self.config.max_steps - self.step_count)
        return 0.25 * (remaining / max(1, self.config.max_steps))

    def _finish(
        self,
        reward: float,
        outcome: str,
        selected: Cell,
        thief_before: Cell,
        thief_after: Cell,
        event: str,
    ) -> StepInfo:
        self.done = True
        self.outcome = outcome
        return StepInfo(
            reward=reward,
            done=True,
            outcome=outcome,
            selected=selected,
            thief_before=thief_before,
            thief_after=thief_after,
            stolen=self.stolen,
            event=event,
        )


class SoftmaxPolicyAgent:
    def __init__(
        self,
        n_actions: int,
        tau: float = 0.8,
        lr: float = 0.03,
        gamma: float = 0.99,
        seed: Optional[int] = None,
        theta: Optional[Sequence[float]] = None,
    ) -> None:
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        self.n_actions = n_actions
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.theta = list(theta) if theta is not None else [0.0 for _ in range(n_actions)]
        if len(self.theta) != n_actions:
            raise ValueError("theta length does not match number of actions")
        self.baseline = 0.0
        self.baseline_ready = False

    def probs(self) -> List[float]:
        scaled = [v / self.tau for v in self.theta]
        max_scaled = max(scaled)
        exps = [math.exp(v - max_scaled) for v in scaled]
        total = sum(exps)
        return [v / total for v in exps]

    def sample_action(self) -> Tuple[int, List[float]]:
        probs = self.probs()
        x = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x <= acc:
                return i, probs
        return len(probs) - 1, probs

    def update(self, trajectory: Sequence[Tuple[int, List[float], float]]) -> float:
        returns: List[float] = []
        g = 0.0
        for _, _, reward in reversed(trajectory):
            g = reward + self.gamma * g
            returns.append(g)
        returns.reverse()
        if not returns:
            return 0.0

        episode_return = returns[0]
        if not self.baseline_ready:
            self.baseline = episode_return
            self.baseline_ready = True
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * episode_return

        for (action, probs, _), ret in zip(trajectory, returns):
            advantage = max(-2.5, min(2.5, ret - self.baseline))
            for i in range(self.n_actions):
                grad = (1.0 if i == action else 0.0) - probs[i]
                self.theta[i] += self.lr * advantage * grad / self.tau

        self.theta = [max(-25.0, min(25.0, v)) for v in self.theta]
        return episode_return

    def to_json(self) -> Dict[str, object]:
        return {
            "theta": self.theta,
            "tau": self.tau,
            "lr": self.lr,
            "gamma": self.gamma,
            "baseline": self.baseline,
        }


class HeistRenderer:
    def __init__(
        self,
        topology: Topology,
        cell_size: int = 82,
        speed: float = 0.04,
        title: str = "Museum Heist",
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.topology = topology
        self.cell_size = cell_size
        self.speed = speed
        self.closed = False
        self.margin = 28
        self.side_panel = 315
        width = topology.cols * cell_size + self.margin * 2 + self.side_panel
        height = max(topology.rows * cell_size + self.margin * 2, 470)

        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#f5f6f8")
        self.canvas.pack()

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def draw(
        self,
        env: MuseumHeistEnv,
        probs: Sequence[float],
        episode: int,
        metrics: Dict[str, object],
        pause: bool = True,
    ) -> None:
        if self.closed:
            return
        try:
            self.canvas.delete("all")
            self._draw_grid(env, probs)
            self._draw_panel(env, probs, episode, metrics)
            self.root.update_idletasks()
            self.root.update()
            if pause and self.speed > 0:
                time.sleep(self.speed)
        except Exception:
            self.closed = True

    def _heat_color(self, value: float, max_value: float) -> str:
        ratio = 0.0 if max_value <= 0 else min(1.0, value / max_value)
        low = (238, 242, 247)
        high = (230, 82, 62)
        mid = (255, 202, 95)
        if ratio < 0.5:
            local = ratio / 0.5
            rgb = tuple(int(low[i] + (mid[i] - low[i]) * local) for i in range(3))
        else:
            local = (ratio - 0.5) / 0.5
            rgb = tuple(int(mid[i] + (high[i] - mid[i]) * local) for i in range(3))
        return "#%02x%02x%02x" % rgb

    def _draw_grid(self, env: MuseumHeistEnv, probs: Sequence[float]) -> None:
        cs = self.cell_size
        m = self.margin
        max_p = max(probs) if probs else 1.0
        top = env.topology

        for idx, p in enumerate(probs):
            r, c = top.cell(idx)
            x0 = m + c * cs
            y0 = m + r * cs
            x1 = x0 + cs
            y1 = y0 + cs
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=self._heat_color(p, max_p),
                outline="#c8cdd4",
                width=1,
            )
            self.canvas.create_text(
                x0 + 9,
                y0 + 8,
                anchor="nw",
                text=f"{p * 100:4.1f}%",
                fill="#23272f",
                font=("Segoe UI", 9),
            )

        for w in top.walls:
            a, b = tuple(w)
            ar, ac = a
            br, bc = b
            if ar == br:
                x = m + max(ac, bc) * cs
                y0 = m + ar * cs
                y1 = y0 + cs
                self.canvas.create_line(x, y0, x, y1, fill="#151922", width=5)
            else:
                y = m + max(ar, br) * cs
                x0 = m + ac * cs
                x1 = x0 + cs
                self.canvas.create_line(x0, y, x1, y, fill="#151922", width=5)

        def center(cell: Cell) -> Tuple[float, float]:
            r, c = cell
            return m + c * cs + cs / 2, m + r * cs + cs / 2

        self._label_cell(env.start, "S", "#2f855a")
        self._label_cell(env.target, "P" if not env.stolen else "X", "#8a3ffc")
        if env.active_camera is not None:
            x, y = center(env.active_camera)
            self.canvas.create_oval(
                x - 19, y - 19, x + 19, y + 19, outline="#0f172a", width=3
            )
            self.canvas.create_text(x, y + 24, text="CAM", fill="#0f172a", font=("Segoe UI", 8, "bold"))
        x, y = center(env.thief_pos)
        self.canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill="#111827", outline="")
        self.canvas.create_text(x, y, text="T", fill="#ffffff", font=("Segoe UI", 11, "bold"))

    def _label_cell(self, cell: Cell, label: str, color: str) -> None:
        r, c = cell
        cs = self.cell_size
        m = self.margin
        x = m + c * cs + cs - 18
        y = m + r * cs + 17
        self.canvas.create_oval(x - 13, y - 13, x + 13, y + 13, fill=color, outline="")
        self.canvas.create_text(x, y, text=label, fill="#ffffff", font=("Segoe UI", 10, "bold"))

    def _draw_panel(
        self,
        env: MuseumHeistEnv,
        probs: Sequence[float],
        episode: int,
        metrics: Dict[str, object],
    ) -> None:
        x = self.margin + self.topology.cols * self.cell_size + 28
        y = self.margin
        self.canvas.create_text(
            x,
            y,
            anchor="nw",
            text="Museum Heist",
            fill="#111827",
            font=("Segoe UI", 20, "bold"),
        )
        y += 44
        lines = [
            f"Topology: {env.topology.name}",
            f"Episode: {episode}",
            f"Step: {env.step_count}/{env.config.max_steps}",
            f"Beta thief prudence: {env.config.beta:.2f}",
            f"Stolen painting: {'yes' if env.stolen else 'no'}",
            f"Start: {env.start}   Target: {env.target}",
            f"Thief: {env.thief_pos}   Camera: {env.active_camera}",
        ]
        for line in lines:
            self.canvas.create_text(
                x, y, anchor="nw", text=line, fill="#242936", font=("Segoe UI", 10)
            )
            y += 23
        y += 10

        stat_lines = [
            f"Window catch: {float(metrics.get('catch_rate', 0.0)) * 100:5.1f}%",
            f"Window detect: {float(metrics.get('detect_rate', 0.0)) * 100:5.1f}%",
            f"Window escape: {float(metrics.get('escape_rate', 0.0)) * 100:5.1f}%",
            f"Avg steps: {float(metrics.get('avg_steps', 0.0)):5.1f}",
            f"Last return: {float(metrics.get('last_return', 0.0)):6.3f}",
        ]
        for line in stat_lines:
            self.canvas.create_text(
                x, y, anchor="nw", text=line, fill="#111827", font=("Segoe UI", 11, "bold")
            )
            y += 24

        y += 18
        best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
        self.canvas.create_text(
            x,
            y,
            anchor="nw",
            text=f"Most watched room: {self.topology.cell(best_idx)} ({probs[best_idx] * 100:.1f}%)",
            fill="#111827",
            font=("Segoe UI", 10),
        )
        y += 30

        legend = [
            ("S", "start room", "#2f855a"),
            ("P", "painting room", "#8a3ffc"),
            ("X", "painting already stolen", "#8a3ffc"),
            ("T", "thief", "#111827"),
        ]
        for tag, desc, color in legend:
            self.canvas.create_oval(x, y, x + 18, y + 18, fill=color, outline="")
            self.canvas.create_text(
                x + 9,
                y + 9,
                text=tag,
                fill="#ffffff",
                font=("Segoe UI", 8, "bold"),
            )
            self.canvas.create_text(
                x + 27, y + 1, anchor="nw", text=desc, fill="#242936", font=("Segoe UI", 9)
            )
            y += 25


def make_renderer(
    enabled: bool,
    topology: Topology,
    speed: float,
    title: str,
) -> Optional[HeistRenderer]:
    if not enabled:
        return None
    try:
        return HeistRenderer(topology=topology, speed=speed, title=title)
    except Exception as exc:
        print(f"[warn] Render disabled because tkinter could not start: {exc}")
        return None


def empty_metrics() -> Dict[str, object]:
    return {
        "catch_rate": 0.0,
        "detect_rate": 0.0,
        "escape_rate": 0.0,
        "avg_steps": 0.0,
        "last_return": 0.0,
    }


def metrics_from_window(outcomes: Deque[str], steps: Deque[int], last_return: float) -> Dict[str, object]:
    total = max(1, len(outcomes))
    return {
        "catch_rate": sum(1 for x in outcomes if x == "caught") / total,
        "detect_rate": sum(1 for x in outcomes if x == "detected") / total,
        "escape_rate": sum(1 for x in outcomes if x == "escaped") / total,
        "timeout_rate": sum(1 for x in outcomes if x == "timeout") / total,
        "avg_steps": sum(steps) / max(1, len(steps)),
        "last_return": last_return,
    }


def run_episode(
    env: MuseumHeistEnv,
    agent: SoftmaxPolicyAgent,
    train: bool,
    renderer: Optional[HeistRenderer] = None,
    episode: int = 0,
    metrics: Optional[Dict[str, object]] = None,
    render_steps: bool = False,
) -> Tuple[str, int, float]:
    env.reset()
    trajectory: List[Tuple[int, List[float], float]] = []
    total_reward = 0.0
    metrics = metrics or empty_metrics()

    if renderer and render_steps:
        renderer.draw(env, agent.probs(), episode, metrics, pause=True)

    while not env.done:
        action, probs = agent.sample_action()
        info = env.step(action)
        trajectory.append((action, probs, info.reward))
        total_reward += info.reward
        if renderer and render_steps:
            renderer.draw(env, agent.probs(), episode, metrics, pause=True)

    if train:
        total_reward = agent.update(trajectory)

    outcome = env.outcome or "unknown"
    return outcome, env.step_count, total_reward


def evaluate_agent(
    topology: Topology,
    agent: SoftmaxPolicyAgent,
    config: HeistConfig,
    episodes: int,
    seed: Optional[int] = None,
    renderer: Optional[HeistRenderer] = None,
    render_steps: bool = False,
) -> Dict[str, object]:
    rng = random.Random(seed)
    env = MuseumHeistEnv(topology, config, rng=rng)
    outcomes: Deque[str] = deque(maxlen=episodes)
    steps: Deque[int] = deque(maxlen=episodes)
    returns: List[float] = []
    metrics = empty_metrics()

    old_lr = agent.lr
    agent.lr = 0.0
    for ep in range(1, episodes + 1):
        outcome, step_count, total_return = run_episode(
            env,
            agent,
            train=False,
            renderer=renderer,
            episode=ep,
            metrics=metrics,
            render_steps=render_steps,
        )
        outcomes.append(outcome)
        steps.append(step_count)
        returns.append(total_return)
        metrics = metrics_from_window(outcomes, steps, total_return)
        if renderer and not render_steps:
            renderer.draw(env, agent.probs(), ep, metrics, pause=False)
    agent.lr = old_lr

    total = max(1, len(outcomes))
    return {
        "episodes": episodes,
        "caught": sum(1 for x in outcomes if x == "caught"),
        "detected": sum(1 for x in outcomes if x == "detected"),
        "escaped": sum(1 for x in outcomes if x == "escaped"),
        "timeout": sum(1 for x in outcomes if x == "timeout"),
        "catch_rate": sum(1 for x in outcomes if x == "caught") / total,
        "detect_rate": sum(1 for x in outcomes if x == "detected") / total,
        "escape_rate": sum(1 for x in outcomes if x == "escaped") / total,
        "timeout_rate": sum(1 for x in outcomes if x == "timeout") / total,
        "avg_steps": sum(steps) / max(1, len(steps)),
        "avg_return": sum(returns) / max(1, len(returns)),
    }


def train_agent(args: SimpleNamespace) -> Path:
    topology = TOPOLOGIES[args.topology]
    config = HeistConfig(
        beta=args.beta,
        max_steps=args.max_steps,
        reward_catch=args.reward_catch,
        reward_escape=args.reward_escape,
        step_penalty=args.step_penalty,
    )
    rng = random.Random(args.seed)
    env = MuseumHeistEnv(topology, config, rng=rng)
    agent = SoftmaxPolicyAgent(
        topology.n_rooms,
        tau=args.tau,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
    )
    renderer = make_renderer(args.render, topology, args.speed, "Museum Heist - training")

    outcomes: Deque[str] = deque(maxlen=args.window)
    steps: Deque[int] = deque(maxlen=args.window)
    metrics = empty_metrics()
    checkpoints: List[Dict[str, object]] = []
    started = time.time()

    print(
        f"Training {args.episodes} episodes | topology={topology.name} | "
        f"tau={args.tau} | beta={args.beta} | render={args.render}"
    )
    for ep in range(1, args.episodes + 1):
        render_this = bool(renderer) and (
            ep == 1 or ep == args.episodes or ep % max(1, args.render_every) == 0
        )
        outcome, step_count, total_return = run_episode(
            env,
            agent,
            train=True,
            renderer=renderer,
            episode=ep,
            metrics=metrics,
            render_steps=render_this,
        )
        outcomes.append(outcome)
        steps.append(step_count)
        metrics = metrics_from_window(outcomes, steps, total_return)

        if ep % max(1, args.log_every) == 0 or ep == 1 or ep == args.episodes:
            elapsed = time.time() - started
            print(
                f"ep {ep:5d}/{args.episodes} | "
                f"catch {metrics['catch_rate'] * 100:5.1f}% | "
                f"detect {metrics['detect_rate'] * 100:5.1f}% | "
                f"escape {metrics['escape_rate'] * 100:5.1f}% | "
                f"steps {metrics['avg_steps']:5.1f} | "
                f"return {total_return:7.3f} | {elapsed:5.1f}s"
            )
            checkpoints.append(
                {
                    "episode": ep,
                    "catch_rate": metrics["catch_rate"],
                    "detect_rate": metrics["detect_rate"],
                    "escape_rate": metrics["escape_rate"],
                    "avg_steps": metrics["avg_steps"],
                    "last_return": total_return,
                }
            )

        if renderer and not render_this and ep % max(1, args.render_every) == 0:
            renderer.draw(env, agent.probs(), ep, metrics, pause=False)

    eval_stats = evaluate_agent(
        topology=topology,
        agent=agent,
        config=config,
        episodes=args.eval_episodes,
        seed=None if args.seed is None else args.seed + 999,
    )
    print_stats("Evaluation after training", eval_stats)

    if args.no_save:
        return Path("")

    path = save_model(
        agent=agent,
        topology=topology,
        config=config,
        args=args,
        train_metrics=metrics,
        eval_stats=eval_stats,
        checkpoints=checkpoints,
        path=args.save,
    )
    print(f"Saved model: {path}")
    return path


def save_model(
    agent: SoftmaxPolicyAgent,
    topology: Topology,
    config: HeistConfig,
    args: SimpleNamespace,
    train_metrics: Dict[str, object],
    eval_stats: Dict[str, object],
    checkpoints: List[Dict[str, object]],
    path: Optional[str] = None,
) -> Path:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    if path:
        out = Path(path)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = models_dir / f"{stamp}_{topology.name}_tau{args.tau}_beta{args.beta}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topology": topology.to_json(),
        "agent": agent.to_json(),
        "heist_config": {
            "beta": config.beta,
            "max_steps": config.max_steps,
            "reward_catch": config.reward_catch,
            "reward_escape": config.reward_escape,
            "reward_timeout": config.reward_timeout,
            "step_penalty": config.step_penalty,
            "unseen_safe_age": config.unseen_safe_age,
        },
        "training_args": {
            "episodes": args.episodes,
            "window": args.window,
            "seed": args.seed,
            "topology": topology.name,
        },
        "training_window_metrics": train_metrics,
        "evaluation": eval_stats,
        "checkpoints": checkpoints[-100:],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_model(path: Path) -> Tuple[Topology, SoftmaxPolicyAgent, HeistConfig, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    topology = Topology.from_json(data["topology"])
    agent_data = data["agent"]
    config_data = data.get("heist_config", {})
    config = HeistConfig(
        beta=float(config_data.get("beta", 4.0)),
        max_steps=int(config_data.get("max_steps", 120)),
        reward_catch=float(config_data.get("reward_catch", 1.0)),
        reward_escape=float(config_data.get("reward_escape", -1.0)),
        reward_timeout=float(config_data.get("reward_timeout", -0.35)),
        step_penalty=float(config_data.get("step_penalty", -0.002)),
        unseen_safe_age=int(config_data.get("unseen_safe_age", 24)),
    )
    agent = SoftmaxPolicyAgent(
        topology.n_rooms,
        tau=float(agent_data.get("tau", 0.8)),
        lr=float(agent_data.get("lr", 0.0)),
        gamma=float(agent_data.get("gamma", 0.99)),
        theta=agent_data["theta"],
    )
    return topology, agent, config, data


def list_model_files() -> List[Path]:
    models = sorted(Path("models").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return models


def select_model_interactively() -> Optional[Path]:
    models = list_model_files()
    if not models:
        print("No models found in ./models. Train one first.")
        return None
    print("Available models:")
    for i, path in enumerate(models, start=1):
        print(f"  {i}. {path}")
    while True:
        choice = input("Choose a model number: ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid choice. Type a number or q to quit.")


def resolve_model_path(model_arg: Optional[str]) -> Optional[Path]:
    if model_arg in {None, ""}:
        return select_model_interactively()
    if model_arg == "latest":
        models = list_model_files()
        if not models:
            print("No models found in ./models. Train one first.")
            return None
        return models[0]
    path = Path(model_arg)
    if not path.exists():
        print(f"Model not found: {path}")
        return None
    return path


def print_stats(title: str, stats: Dict[str, object]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"episodes    : {stats['episodes']}")
    print(f"caught      : {stats['caught']} ({stats['catch_rate'] * 100:.1f}%)")
    print(f"detected    : {stats['detected']} ({stats['detect_rate'] * 100:.1f}%)")
    print(f"escaped     : {stats['escaped']} ({stats['escape_rate'] * 100:.1f}%)")
    print(f"timeout     : {stats['timeout']} ({stats['timeout_rate'] * 100:.1f}%)")
    print(f"avg steps   : {stats['avg_steps']:.2f}")
    print(f"avg return  : {stats['avg_return']:.3f}")


def command_evaluate(args: SimpleNamespace) -> None:
    path = resolve_model_path(args.model)
    if path is None:
        return
    topology, agent, config, _ = load_model(path)
    if args.beta is not None:
        config.beta = args.beta
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    renderer = make_renderer(args.render, topology, args.speed, "Museum Heist - evaluation")
    stats = evaluate_agent(
        topology=topology,
        agent=agent,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        renderer=renderer,
        render_steps=args.render,
    )
    print_stats(f"Evaluation for {path}", stats)
    if renderer and not renderer.closed:
        print("Close the render window to finish.")
        while not renderer.closed:
            try:
                renderer.root.update()
                time.sleep(0.05)
            except Exception:
                break


def command_play(args: SimpleNamespace) -> None:
    path = resolve_model_path(args.model)
    if path is None:
        return
    topology, agent, config, data = load_model(path)
    if args.beta is not None:
        config.beta = args.beta
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    print(f"Loaded model: {path}")
    if "evaluation" in data:
        eval_data = data["evaluation"]
        print(
            "Saved eval: "
            f"catch={eval_data.get('catch_rate', 0) * 100:.1f}% "
            f"detect={eval_data.get('detect_rate', 0) * 100:.1f}% "
            f"escape={eval_data.get('escape_rate', 0) * 100:.1f}%"
        )

    renderer = make_renderer(args.render, topology, args.speed, "Museum Heist - play model")
    stats = evaluate_agent(
        topology=topology,
        agent=agent,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        renderer=renderer,
        render_steps=args.render,
    )
    print_stats("Play summary", stats)
    if renderer and not renderer.closed:
        print("Close the render window to finish.")
        while not renderer.closed:
            try:
                renderer.root.update()
                time.sleep(0.05)
            except Exception:
                break


def command_list_topologies(_: SimpleNamespace) -> None:
    print("Available topologies:")
    for top in TOPOLOGIES.values():
        print(
            f"- {top.name}: {top.rows}x{top.cols}, "
            f"{len(top.walls)} walls. {top.description}"
        )


def command_list_models(_: SimpleNamespace) -> None:
    models = list_model_files()
    if not models:
        print("No models found in ./models.")
        return
    for path in models:
        print(path)


def validate_variables() -> None:
    if cfg.TOPOLOGY not in TOPOLOGIES:
        valid = ", ".join(sorted(TOPOLOGIES))
        raise ValueError(f"Unknown TOPOLOGY={cfg.TOPOLOGY!r}. Valid options: {valid}")
    if cfg.MODE not in {"train", "play", "evaluate", "topologies", "models"}:
        raise ValueError("MODE must be: train, play, evaluate, topologies or models")
    if cfg.TAU <= 0:
        raise ValueError("TAU must be greater than zero")
    if cfg.TRAIN_EPISODES <= 0:
        raise ValueError("TRAIN_EPISODES must be greater than zero")
    if cfg.MAX_STEPS <= 0:
        raise ValueError("MAX_STEPS must be greater than zero")


def build_train_args() -> SimpleNamespace:
    return SimpleNamespace(
        episodes=cfg.TRAIN_EPISODES,
        topology=cfg.TOPOLOGY,
        beta=cfg.BETA,
        tau=cfg.TAU,
        lr=cfg.LEARNING_RATE,
        gamma=cfg.GAMMA,
        max_steps=cfg.MAX_STEPS,
        window=cfg.ROLLING_WINDOW,
        eval_episodes=cfg.EVAL_EPISODES_AFTER_TRAINING,
        seed=cfg.SEED,
        save=cfg.SAVE_PATH,
        no_save=not cfg.SAVE_MODEL,
        log_every=cfg.LOG_EVERY,
        render_every=cfg.RENDER_EVERY,
        speed=cfg.RENDER_SPEED,
        render=cfg.RENDER,
        reward_catch=cfg.REWARD_CATCH,
        reward_escape=cfg.REWARD_ESCAPE,
        step_penalty=cfg.STEP_PENALTY,
    )


def build_eval_args(episodes: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=cfg.MODEL,
        episodes=episodes,
        beta=cfg.BETA,
        max_steps=cfg.MAX_STEPS,
        seed=cfg.SEED,
        speed=cfg.RENDER_SPEED,
        render=cfg.RENDER,
    )


def main() -> int:
    validate_variables()
    mode = cfg.MODE.strip().lower()
    print(f"MODE={mode} | RENDER={cfg.RENDER}")

    if mode == "train":
        train_agent(build_train_args())
    elif mode == "evaluate":
        command_evaluate(build_eval_args(cfg.EVALUATE_EPISODES))
    elif mode == "play":
        command_play(build_eval_args(cfg.PLAY_EPISODES))
    elif mode == "topologies":
        command_list_topologies(SimpleNamespace())
    elif mode == "models":
        command_list_models(SimpleNamespace())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
