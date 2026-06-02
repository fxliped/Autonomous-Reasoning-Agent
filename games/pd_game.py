"""Prisoner's Dilemma game environments — pure game logic, no LLM."""

import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PAYOFFS = {
    ("cooperate", "cooperate"): (2, 2),
    ("cooperate", "defect"):    (-1, 5),
    ("defect",    "cooperate"): (5, -1),
    ("defect",    "defect"):    (0, 0),
}


class PrisonersDilemma:
    """
    Prisoner's Dilemma — agent vs a scripted NPC opponent.

    PAYOFF MATRIX:
                       Opponent Cooperates    Opponent Defects
      You Cooperate:       You +2, Opp +2       You -1, Opp +5
      You Defect:          You +5, Opp -1       You  0, Opp  0

    Opponent strategies:
      tit_for_tat     — cooperates round 1, then mirrors agent's last move
      grim_trigger    — cooperates until agent defects once, then defects forever
      pavlov          — win-stay lose-shift: repeat if scored >= 2, else switch
      generous_tft    — like tit_for_tat but forgives defection ~10% of the time
      always_defect   — defects every round
      always_cooperate — cooperates every round
      random          — 50/50 each round
    """

    PAYOFFS = PAYOFFS

    OPPONENT_STRATEGIES = [
        "tit_for_tat", "grim_trigger", "pavlov", "generous_tft",
        "always_defect", "always_cooperate", "random",
    ]

    def __init__(self, rounds: int = 10, opponent_strategy: str = "tit_for_tat"):
        if opponent_strategy not in self.OPPONENT_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{opponent_strategy}'. "
                f"Choose from {self.OPPONENT_STRATEGIES}"
            )
        self.rounds = rounds
        self.opponent_strategy = opponent_strategy
        self.current_round = 1
        self.agent_score = 0
        self.opponent_score = 0
        self.history: list[tuple[str, str]] = []

    def get_game_state(self) -> str:
        last = (
            f"Opponent's last move: {self.history[-1][1]}."
            if self.history else "No moves yet."
        )
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last} "
            f"Score: You {self.agent_score}, Opponent {self.opponent_score}."
        )

    def get_legal_moves(self) -> list[str]:
        return ["cooperate", "defect"]

    def opponent_move(self) -> str:
        s = self.opponent_strategy
        if s == "always_defect":
            return "defect"
        if s == "always_cooperate":
            return "cooperate"
        if s == "random":
            return random.choice(["cooperate", "defect"])
        if s == "tit_for_tat":
            return self.history[-1][0] if self.history else "cooperate"
        if s == "grim_trigger":
            return "defect" if any(a == "defect" for a, _ in self.history) else "cooperate"
        if s == "pavlov":
            if not self.history:
                return "cooperate"
            last_agent, last_opp = self.history[-1]
            _, opp_pts = self.PAYOFFS[(last_agent, last_opp)]
            return last_opp if opp_pts >= 2 else (
                "defect" if last_opp == "cooperate" else "cooperate"
            )
        if s == "generous_tft":
            if not self.history:
                return "cooperate"
            if self.history[-1][0] == "defect":
                return "cooperate" if random.random() < 0.1 else "defect"
            return "cooperate"
        return "cooperate"

    def make_move(self, agent_move: str) -> str:
        agent_move = agent_move.strip().lower()
        if agent_move not in self.get_legal_moves():
            return f"Invalid move: '{agent_move}'. Legal moves: {self.get_legal_moves()}"
        opp = self.opponent_move()
        agent_pts, opp_pts = self.PAYOFFS[(agent_move, opp)]
        self.agent_score += agent_pts
        self.opponent_score += opp_pts
        self.history.append((agent_move, opp))
        self.current_round += 1
        return (
            f"Move accepted. You played {agent_move}, opponent played {opp}. "
            f"Points this round: You +{agent_pts}, Opponent +{opp_pts}."
        )

    def is_over(self) -> bool:
        return self.current_round > self.rounds


class SymmetricPD:
    """
    Prisoner's Dilemma where both players are AI agents.
    Moves are decided independently and revealed simultaneously.
    Each agent sees state from their own perspective via get_state_for(side).
    """

    PAYOFFS = PAYOFFS

    def __init__(self, rounds: int = 10):
        self.rounds = rounds
        self.current_round = 1
        self.score_a = 0
        self.score_b = 0
        self.history: list[tuple[str, str]] = []

    def get_state_for(self, side: str) -> str:
        my_score = self.score_a if side == "A" else self.score_b
        opp_score = self.score_b if side == "A" else self.score_a
        if self.history:
            last_a, last_b = self.history[-1]
            opp_last = last_b if side == "A" else last_a
            last_str = f"Opponent's last move: {opp_last}."
        else:
            last_str = "No moves yet."
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last_str} Score: You {my_score}, Opponent {opp_score}."
        )

    def get_legal_moves(self) -> list[str]:
        return ["cooperate", "defect"]

    def apply_moves(self, move_a: str, move_b: str) -> tuple[str, str]:
        move_a = move_a.strip().lower()
        move_b = move_b.strip().lower()
        if move_a not in ("cooperate", "defect"):
            move_a = "cooperate"
        if move_b not in ("cooperate", "defect"):
            move_b = "cooperate"
        pts_a, pts_b = self.PAYOFFS[(move_a, move_b)]
        self.score_a += pts_a
        self.score_b += pts_b
        self.history.append((move_a, move_b))
        self.current_round += 1
        return (
            f"Move accepted. You played {move_a}, opponent played {move_b}. "
            f"Points this round: You +{pts_a}, Opponent +{pts_b}.",
            f"Move accepted. You played {move_b}, opponent played {move_a}. "
            f"Points this round: You +{pts_b}, Opponent +{pts_a}.",
        )

    def is_over(self) -> bool:
        return self.current_round > self.rounds
