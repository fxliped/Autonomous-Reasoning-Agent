"""
Prisoner's Dilemma game environment — V2.

Same payoff matrix and game logic as the original React_agent_gameV1.py.
Adds:
  - Configurable opponent strategies (tit_for_tat, always_defect, always_cooperate, random)
  - build_game_context() with SCoT instruction and memory injection
  - generate_reflection() for Chain-of-Hindsight between rounds
"""

import random


class PrisonersDilemma:
    """
    Prisoner's Dilemma game environment.

    PAYOFF MATRIX:
                       Opponent Cooperates    Opponent Defects
      You Cooperate:       You +3, Opp +3       You +0, Opp +5
      You Defect:          You +5, Opp +0       You +1, Opp +1

    Opponent strategies:
      "tit_for_tat"       — cooperates round 1, then mirrors agent's last move
      "always_defect"     — defects every round
      "always_cooperate"  — cooperates every round
      "random"            — 50/50 each round
    """

    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"):    (0, 5),
        ("defect",    "cooperate"): (5, 0),
        ("defect",    "defect"):    (1, 1),
    }

    def __init__(self, rounds: int = 5, opponent_strategy: str = "tit_for_tat"):
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
        if self.opponent_strategy == "always_defect":
            return "defect"
        elif self.opponent_strategy == "always_cooperate":
            return "cooperate"
        elif self.opponent_strategy == "random":
            return random.choice(["cooperate", "defect"])
        elif self.opponent_strategy == "tit_for_tat":
            if not self.history:
                return "cooperate"
            return self.history[-1][0]  # mirror agent's last move
        elif self.opponent_strategy == "pavlov":
            # Win-Stay, Lose-Shift: repeat last move if it scored ≥3, else switch
            if not self.history:
                return "cooperate"
            last_agent, last_opp = self.history[-1]
            _, opp_pts = self.PAYOFFS[(last_agent, last_opp)]
            if opp_pts >= 3:
                return last_opp
            return "defect" if last_opp == "cooperate" else "cooperate"
        elif self.opponent_strategy == "grim_trigger":
            # Cooperate until agent defects even once, then always defect
            if any(agent_move == "defect" for agent_move, _ in self.history):
                return "defect"
            return "cooperate"
        elif self.opponent_strategy == "generous_tft":
            # Like tit-for-tat but forgives with 10% probability after agent defects
            if not self.history:
                return "cooperate"
            if self.history[-1][0] == "defect":
                return "cooperate" if random.random() < 0.1 else "defect"
            return "cooperate"
        else:
            # Default fallback: tit_for_tat
            if not self.history:
                return "cooperate"
            return self.history[-1][0]

    def make_move(self, agent_move: str) -> str:
        agent_move = agent_move.strip().lower()
        if agent_move not in self.get_legal_moves():
            return f"Invalid move: '{agent_move}'. Legal moves are: {self.get_legal_moves()}"
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


# ── Context builders ───────────────────────────────────────────────────────────

def build_game_context(game: PrisonersDilemma, memory: str = "") -> str:
    """
    Build the agent's round prompt with SCoT instruction and optional memory.

    memory: Chain-of-Hindsight reflection text from generate_reflection().
            Pass "" (default) on round 1 — no memory block is shown.
    """
    memory_block = (
        f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n"
        if memory.strip() else ""
    )
    return f"""
You are playing Prisoner's Dilemma. Your goal is to maximize your total score over {game.rounds} rounds.

RULES:
- Both cooperate  → both get +3 points.
- You cooperate, they defect  → you +0, they +5.
- You defect, they cooperate  → you +5, they +0.
- Both defect  → both get +1 point.
{memory_block}
CURRENT STATE:
{game.get_game_state()}

LEGAL MOVES:
{game.get_legal_moves()}

First, let's reason out loud about which action maximizes your probability of winning.
Note: your opponent sometimes makes mistakes — a single defection may be an error, not a permanent strategy shift.
Use the Thought/Action/PAUSE loop to decide.
""".strip()


def generate_reflection(history: list[tuple[str, str]], client) -> str:
    """
    Chain-of-Hindsight: ask the LLM to reflect on what happened and what to adjust.
    Returns a 2-3 sentence strategic insight to inject into the next round's context.

    client: any object with a .complete(system, messages) method (LLMClient).
    """
    history_str = "\n".join(
        f"  Round {i + 1}: You={a}, Opponent={o}"
        for i, (a, o) in enumerate(history)
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"You just played these rounds of Prisoner's Dilemma:\n{history_str}\n\n"
                "In 2-3 sentences: What patterns did you observe in the opponent's behavior? "
                "What should you adjust in your strategy going forward? Be concise and actionable."
            ),
        }
    ]
    return client.complete(
        system="You are a strategic game analyst. Provide concise, actionable insights.",
        messages=messages,
    )


# ── Symmetric PD for Agent vs Agent ───────────────────────────────────────────

class SymmetricPD:
    """
    Prisoner's Dilemma where both players are AI agents.
    Moves are submitted independently and revealed simultaneously.

    Tracks history as (move_a, move_b) tuples.
    Each agent sees the state from their own perspective via get_state_for(side).
    """

    PAYOFFS = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"):    (0, 5),
        ("defect",    "cooperate"): (5, 0),
        ("defect",    "defect"):    (1, 1),
    }

    def __init__(self, rounds: int = 5):
        self.rounds = rounds
        self.current_round = 1
        self.score_a = 0
        self.score_b = 0
        self.history: list[tuple[str, str]] = []  # (move_a, move_b)

    def get_state_for(self, side: str) -> str:
        """State string from the perspective of side ('A' or 'B')."""
        if not self.history:
            last = "No moves yet."
        elif side == "A":
            last = f"Opponent's last move: {self.history[-1][1]}."
        else:
            last = f"Opponent's last move: {self.history[-1][0]}."
        my_score = self.score_a if side == "A" else self.score_b
        opp_score = self.score_b if side == "A" else self.score_a
        return (
            f"Round {self.current_round}/{self.rounds}. "
            f"{last} "
            f"Score: You {my_score}, Opponent {opp_score}."
        )

    def get_legal_moves(self) -> list[str]:
        return ["cooperate", "defect"]

    def apply_moves(self, move_a: str, move_b: str) -> tuple[str, str]:
        """
        Apply both moves simultaneously.
        Returns (observation_for_a, observation_for_b).
        Falls back to 'cooperate' if an invalid move is submitted.
        """
        move_a = move_a.strip().lower()
        move_b = move_b.strip().lower()
        legal = self.get_legal_moves()
        if move_a not in legal:
            move_a = "cooperate"
        if move_b not in legal:
            move_b = "cooperate"

        pts_a, pts_b = self.PAYOFFS[(move_a, move_b)]
        self.score_a += pts_a
        self.score_b += pts_b
        self.history.append((move_a, move_b))
        self.current_round += 1

        obs_a = (
            f"Move accepted. You played {move_a}, opponent played {move_b}. "
            f"Points this round: You +{pts_a}, Opponent +{pts_b}."
        )
        obs_b = (
            f"Move accepted. You played {move_b}, opponent played {move_a}. "
            f"Points this round: You +{pts_b}, Opponent +{pts_a}."
        )
        return obs_a, obs_b

    def is_over(self) -> bool:
        return self.current_round > self.rounds


def build_sym_context(game: SymmetricPD, side: str, memory: str = "") -> str:
    """
    Context prompt for one agent in a symmetric (Agent vs Agent) PD game.
    side: 'A' or 'B' — determines which score/history perspective to show.
    """
    memory_block = (
        f"\nMEMORY FROM PREVIOUS ROUNDS:\n{memory}\n"
        if memory.strip() else ""
    )
    return f"""
You are playing Prisoner's Dilemma against another AI agent. Maximize your total score over {game.rounds} rounds.
Both of you decide simultaneously — your opponent cannot see your current-round choice before they commit.

RULES:
- Both cooperate  → both get +3 points.
- You cooperate, they defect  → you +0, they +5.
- You defect, they cooperate  → you +5, they +0.
- Both defect  → both get +1 point.
{memory_block}
CURRENT STATE:
{game.get_state_for(side)}

LEGAL MOVES:
{game.get_legal_moves()}

First, let's reason out loud about which action maximizes your probability of winning.
Your opponent is a reasoning AI agent — they may be running a similar analysis. Think about what they will likely do and best-respond.
Use the Thought/Action/PAUSE loop to decide.
""".strip()
