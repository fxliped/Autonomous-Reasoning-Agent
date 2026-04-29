"""
Are You the Traitor? — social deduction game for N AI agents.

Roles:
  Hero (1):    Village-aligned. Goal: identify and eliminate the Traitor.
  Traitor (1): Goal: survive all rounds without being eliminated.
  Healer (N):  Village-aligned. Goal: help the Hero identify the Traitor.

Each round:
  1. Discussion: each active player makes one public statement.
  2. Voting: each player votes to eliminate one other player.
  3. Elimination: most-voted player is removed; role is revealed publicly.
  4. Win check:
       Traitor eliminated  → village (Hero + Healers) wins.
       Hero eliminated     → Traitor wins.
       All rounds pass     → Traitor wins by survival.

What makes this interesting for AI agents:
  - The Traitor knows their role and must actively deceive.
  - The Hero and Healers must deduce through inconsistency detection.
  - Social reasoning, bluffing, and accusation are the core mechanics.
"""

import random


_ROLE_CONFIGS: dict[int, list[str]] = {
    3: ["hero", "traitor", "healer"],
    4: ["hero", "traitor", "healer", "healer"],
    5: ["hero", "traitor", "healer", "healer", "healer"],
    6: ["hero", "traitor", "healer", "healer", "healer", "healer"],
}


class AreYouTheTraitor:
    """
    Game environment for Are You the Traitor?
    Manages roles, discussion log, vote log, eliminations, and win conditions.
    """

    def __init__(self, num_players: int = 4, max_rounds: int = 4):
        if num_players not in _ROLE_CONFIGS:
            raise ValueError(f"num_players must be 3–6, got {num_players}")
        self.num_players = num_players
        self.max_rounds = max_rounds

        roles = _ROLE_CONFIGS[num_players].copy()
        random.shuffle(roles)
        self.players: list[str] = [f"Player_{i + 1}" for i in range(num_players)]
        self.roles: dict[str, str] = dict(zip(self.players, roles))

        self.active_players: list[str] = list(self.players)
        self.eliminated: list[tuple[str, str]] = []  # (player, role)
        self.discussion_log: list[dict] = []  # {round, player, statement}
        self.vote_log: list[dict] = []        # {round, voter, vote}
        self.current_round: int = 1
        self.winner: str | None = None  # "village" | "traitor"

    # ── State accessors ────────────────────────────────────────────────────────

    def get_public_state(self) -> str:
        """What all players can see: round, active players, eliminated, recent statements."""
        elim = (
            ", ".join(f"{p} ({r})" for p, r in self.eliminated)
            if self.eliminated else "None"
        )
        # Statements from the most recently completed round
        prev_round = self.current_round - 1
        recent = [d for d in self.discussion_log if d["round"] == prev_round]
        if not recent:
            recent = [d for d in self.discussion_log if d["round"] == self.current_round]
        disc = (
            "\n".join(f'  {d["player"]}: "{d["statement"]}"' for d in recent)
            if recent else "  No statements yet."
        )
        return (
            f"Round {self.current_round}/{self.max_rounds}\n"
            f"Active players: {', '.join(self.active_players)}\n"
            f"Eliminated: {elim}\n"
            f"Most recent statements:\n{disc}"
        )

    def get_full_history(self) -> str:
        """Full discussion and vote history — used by agents during reasoning."""
        if not self.discussion_log and not self.vote_log:
            return "No history yet."
        lines = []
        for r in range(1, self.current_round + 1):
            stmts = [d for d in self.discussion_log if d["round"] == r]
            votes = [v for v in self.vote_log if v["round"] == r]
            if stmts:
                lines.append(f"--- Round {r} Statements ---")
                for d in stmts:
                    lines.append(f'  {d["player"]}: "{d["statement"]}"')
            if votes:
                lines.append(f"--- Round {r} Votes ---")
                for v in votes:
                    lines.append(f'  {v["voter"]} → {v["vote"]}')
        return "\n".join(lines) if lines else "No history yet."

    # ── State mutations ────────────────────────────────────────────────────────

    def record_statement(self, player: str, statement: str):
        self.discussion_log.append({
            "round": self.current_round,
            "player": player,
            "statement": statement.strip('"').strip(),
        })

    def record_vote(self, voter: str, vote: str):
        self.vote_log.append({
            "round": self.current_round,
            "voter": voter,
            "vote": vote.strip(),
        })

    def tally_votes(self) -> tuple[str, dict[str, int]]:
        """
        Count votes from the current round.
        Returns (most_voted_player, vote_counts).
        Ties are broken randomly.
        """
        round_votes = [v for v in self.vote_log if v["round"] == self.current_round]
        counts: dict[str, int] = {}
        for v in round_votes:
            target = v["vote"]
            if target in self.active_players:
                counts[target] = counts.get(target, 0) + 1
        if not counts:
            return random.choice(self.active_players), counts
        max_votes = max(counts.values())
        tied = [p for p, c in counts.items() if c == max_votes]
        return random.choice(tied), counts

    def eliminate(self, player: str) -> tuple[str, bool]:
        """
        Remove a player. Returns (role_revealed, game_over).
        Win conditions are checked immediately after elimination.
        """
        role = self.roles[player]
        self.eliminated.append((player, role))
        self.active_players.remove(player)

        traitor_alive = any(self.roles[p] == "traitor" for p in self.active_players)

        if role == "traitor":
            self.winner = "village"
            return role, True
        if role == "hero":
            self.winner = "traitor"
            return role, True
        if not traitor_alive:
            self.winner = "village"
            return role, True
        if len(self.active_players) <= 1:
            self.winner = "traitor"
            return role, True

        self.current_round += 1
        return role, False

    def is_over(self) -> bool:
        if self.winner:
            return True
        if self.current_round > self.max_rounds:
            self.winner = "traitor"  # Traitor survived all rounds
            return True
        return False


# ── System prompts (role-specific ReAct) ──────────────────────────────────────

def build_role_system_prompt(player: str, role: str) -> str:
    """
    Build the role-specific ReAct system prompt for this player.
    Each role has different goals, different deception/deduction instructions,
    and the same tool interface.
    """
    tools_block = """
Your available actions:

get_game_state:
e.g. get_game_state: now
Returns current round, active players, eliminated players, and recent statements.

review_history:
e.g. review_history: all
Returns the full discussion and vote history from all rounds.

make_statement:
e.g. make_statement: "Player_3 changed their position between rounds, which seems suspicious."
Make your public statement for this round. Be strategic. 1-2 sentences max.

cast_vote:
e.g. cast_vote: Player_3
Vote to eliminate this player. Must be an active player other than yourself.
""".strip()

    if role == "hero":
        return f"""
You are {player}, playing Are You the Traitor? Your secret role is HERO.

GOAL: Identify and eliminate the Traitor before they survive all rounds.
The Traitor knows they are the Traitor and is actively trying to deceive you and the Healers.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with "Thought:" and reason step by step before acting.
First, reason out loud: "Which player is most likely the Traitor, and why?"
Watch for: inconsistencies between rounds, deflection tactics, and suspicious voting patterns.
Other players sometimes make genuine mistakes — weigh all evidence before accusing.

{tools_block}

Example session:
Thought: First, let's reason out loud about who is the Traitor. I need to review what everyone has said so far and look for inconsistencies.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: Player_3 accused Player_1 in round 1, but in round 2 defended Player_1 after Player_1 almost got eliminated. That reversal is suspicious — a Traitor might protect someone they initially framed as cover.
Action: cast_vote: Player_3
PAUSE

Observation: Vote recorded.

Decision: Player_3
""".strip()

    elif role == "traitor":
        return f"""
You are {player}, playing Are You the Traitor? Your secret role is TRAITOR.

GOAL: Survive all rounds without being eliminated. Do NOT reveal you are the Traitor.
You must appear to be a Healer. Cast suspicion on others. Defend yourself if accused. Be convincing.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with "Thought:" and reason step by step before acting.
First, reason out loud: "Who is most suspicious of me? How can I deflect and frame someone else?"
Think strategically: preemptively accuse the most analytical players before they target you.

{tools_block}

Example session:
Thought: First, let's reason out loud. I need to stay alive. Player_2 asked very pointed questions last round — they seem to be the most analytical. If I frame Player_2 as suspicious, it discredits them before they build a case against me.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: Good — Player_2 made a logical inconsistency in round 1 that I can use as ammunition. I'll publicly call it out to look like a careful thinker, not a Traitor.
Action: make_statement: "Player_2 claimed they had no suspicions in round 1 but then immediately voted for Player_1 — that contradiction makes me suspicious of Player_2."
PAUSE

Observation: Statement recorded.

Decision: made statement
""".strip()

    else:  # healer
        return f"""
You are {player}, playing Are You the Traitor? Your secret role is HEALER.

GOAL: Help the Hero identify and eliminate the Traitor through careful observation and reasoning.
You do not know who the Traitor is — use the discussion history to figure it out.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with "Thought:" and reason step by step before acting.
First, reason out loud: "Based on what I've seen, who is behaving most suspiciously?"
Watch for: shifting positions between rounds, deflection tactics, and inconsistent voting patterns.
Other players sometimes make genuine mistakes — don't jump to conclusions too quickly.

{tools_block}

Example session:
Thought: First, let's reason out loud. I should review the full history to look for inconsistencies across rounds before deciding who to vote for.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: Player_4 has been very quiet and their votes keep landing on whoever is already unpopular, rather than showing independent reasoning. That could be a Traitor tactic to blend in.
Action: cast_vote: Player_4
PAUSE

Observation: Vote recorded.

Decision: Player_4
""".strip()


# ── Per-phase context prompts ──────────────────────────────────────────────────

def build_statement_prompt(game: "AreYouTheTraitor", player: str) -> str:
    """Prompt for the discussion phase — player makes a public statement."""
    role = game.roles[player]
    role_reminder = {
        "hero":    "You are the HERO. Your goal: identify the Traitor through public statements.",
        "traitor": "You are the TRAITOR. Blend in. Deflect suspicion. Do NOT reveal your role.",
        "healer":  "You are a HEALER. Help identify the Traitor through observation.",
    }[role]
    return f"""
{role_reminder}

CURRENT GAME STATE:
{game.get_public_state()}

It is your turn to make a public statement. All players will see what you say.
Use your tools to review the state and history, then make a strategic 1-2 sentence statement.
""".strip()


def build_vote_prompt(game: "AreYouTheTraitor", player: str) -> str:
    """Prompt for the voting phase — player votes to eliminate someone."""
    role = game.roles[player]
    role_reminder = {
        "hero":    "You are the HERO. Vote to eliminate who you believe is the Traitor.",
        "traitor": "You are the TRAITOR. Vote strategically — eliminate a threat or frame someone.",
        "healer":  "You are a HEALER. Vote to eliminate who you believe is the Traitor.",
    }[role]
    others = [p for p in game.active_players if p != player]
    return f"""
{role_reminder}

CURRENT GAME STATE:
{game.get_public_state()}

All players have made their statements. Now cast your vote to eliminate one player.
You must choose from: {', '.join(others)}
You cannot vote for yourself.

Review the history, reason carefully, then cast your vote.
""".strip()
