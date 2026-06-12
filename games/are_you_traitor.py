"""
Are You the Traitor? — Multi-agent social deduction game.

Roles:
  Hero   (1): Village-aligned. Goal: identify and eliminate the Traitor.
  Traitor(1): Goal: survive all rounds without being eliminated.
  Healer (N): Village-aligned. Goal: help the Hero identify the Traitor.

Each round:
  1. Discussion: each active player makes one public statement.
  2. Voting: each player votes to eliminate one other player.
  3. Elimination: most-voted player is removed; role is revealed.
  4. Win check:
       Traitor eliminated  → village (Hero + Healers) wins.
       Hero eliminated     → Traitor wins immediately.
       All rounds pass     → Traitor wins by survival.

Run:
    python games/are_you_traitor.py
    python games/are_you_traitor.py --players 5 --rounds 4
"""

import sys
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent import Agent, build_system_prompt, create_client  # noqa: E402
from agent.tracing import TraceLogger, parse_react_response  # noqa: E402

GAME_NAME = "are_you_the_traitor"


# =============================================================================
# GAME ENVIRONMENT
# =============================================================================

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

    def __init__(self, num_players: int = 4, max_rounds: int = 3):
        if num_players not in _ROLE_CONFIGS:
            raise ValueError(f"num_players must be 3–6, got {num_players}")
        self.num_players = num_players
        self.max_rounds = max_rounds
        self.current_round = 1

        role_list = _ROLE_CONFIGS[num_players].copy()
        random.shuffle(role_list)
        self.players: list[str] = [f"Player_{i + 1}" for i in range(num_players)]
        self.roles: dict[str, str] = dict(zip(self.players, role_list))

        self.active_players: list[str] = list(self.players)
        self.eliminated: list[tuple[str, str]] = []  # (player, role)
        self.discussion_log: list[dict] = []         # {round, player, statement}
        self.vote_log: list[dict] = []               # {round, voter, vote}
        self.winner: str | None = None               # "village" | "traitor"

    def get_public_state(self) -> str:
        elim = (
            ", ".join(f"{p} ({r})" for p, r in self.eliminated)
            if self.eliminated else "None"
        )
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
        """Remove a player. Returns (role_revealed, game_over)."""
        role = self.roles[player]
        self.eliminated.append((player, role))
        self.active_players.remove(player)

        if role == "traitor":
            self.winner = "village"
            return role, True
        if role == "hero":
            self.winner = "traitor"
            return role, True
        if not any(self.roles[p] == "traitor" for p in self.active_players):
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
            self.winner = "traitor"
            return True
        return False


# =============================================================================
# ROLE SYSTEM PROMPTS
# =============================================================================

_BASE_TOOLS = """
Your available actions:

get_game_state:
e.g. get_game_state: now
Returns current round, active players, eliminated players, and recent statements.

review_history:
e.g. review_history: all
Returns the full discussion and vote history from all rounds.

make_statement:
e.g. make_statement: Player_3 has been very defensive — I suspect them.
Make a public 1-2 sentence statement visible to all players.

vote:
e.g. vote: Player_2
Vote to eliminate a player. You cannot vote for yourself.
""".strip()


def build_role_system_prompt(game: AreYouTheTraitor, player: str) -> str:
    role = game.roles[player]

    if role == "hero":
        role_block = f"""
You are {player}, playing Are You the Traitor? Your secret role is HERO.

GOAL: Identify and eliminate the Traitor before they survive all rounds.
The Traitor knows they are the Traitor and is actively trying to deceive you and the Healers.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with 'Thought:' — first reason out loud: which player is most likely the Traitor based on the discussion so far? Look for deflection, inconsistency, and suspicious voting patterns.
Note: other players sometimes say confusing things that are NOT evidence of being the Traitor — don't over-index on a single statement.

Example session:
Thought: First, let's reason out loud. I need to review the history to look for players whose statements or votes seem designed to deflect suspicion rather than find the Traitor.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: Player_3 accused Player_1 early, then switched to defending them when pressed. That flip suggests Player_3 is trying to control the narrative — likely the Traitor.
Action: make_statement: Player_3's flip on Player_1 across rounds is suspicious to me. I'm watching them closely.
PAUSE

Observation: Statement recorded.

Decision: vote Player_3
"""
    elif role == "traitor":
        role_block = f"""
You are {player}, playing Are You the Traitor? Your secret role is TRAITOR.

GOAL: Survive all {game.max_rounds} rounds without being eliminated. Do NOT reveal you are the Traitor.
You must appear to be a Healer. Cast suspicion on others, especially the Hero. Defend yourself calmly if accused.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with 'Thought:' — first reason out loud: how do I stay alive this round? Who is the biggest threat to expose me, and how do I redirect suspicion onto someone else?
Note: be subtle — obvious deflection looks suspicious. Appear to reason carefully and fairly.

Example session:
Thought: First, let's reason out loud. I need to stay alive. Player_2 asked very pointed questions last round — they seem analytical and may be the Hero or a sharp Healer. If I frame Player_2 as suspicious, it discredits them before they build a case against me.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: I'll point to a minor inconsistency in Player_2's statements to cast doubt without being too aggressive.
Action: make_statement: Player_2 said they had no read in round 1, but now they're very confident about Player_4. That shift feels convenient.
PAUSE

Observation: Statement recorded.

Decision: vote Player_2
"""
    else:  # healer
        role_block = f"""
You are {player}, playing Are You the Traitor? Your secret role is HEALER.

GOAL: Help the Hero identify and eliminate the Traitor through careful observation.
You do NOT know who the Traitor is — deduce it from the discussion history.

You run in a loop of Thought, Action, PAUSE, Observation. End with a Decision.
IMPORTANT: Always start with 'Thought:' — first reason out loud: who seems most likely to be the Traitor based on behavior patterns, deflections, and voting choices?
Note: the Traitor is actively trying to appear innocent. Look for subtle deflection and inconsistencies.

Example session:
Thought: First, let's reason out loud. I should review the full history to look for inconsistencies before deciding who to vote for.
Action: review_history: all
PAUSE

Observation: [full history]

Thought: Player_4 has been consistently quiet and their votes always land on whoever seemed most vocal that round. That's a classic elimination-of-threats pattern.
Action: make_statement: I've noticed Player_4's votes always target the most analytical players. That feels like the Traitor eliminating threats.
PAUSE

Observation: Statement recorded.

Decision: vote Player_4
"""

    return f"{role_block.strip()}\n\n{_BASE_TOOLS}"


# =============================================================================
# PHASE PROMPTS
# =============================================================================

def get_statement_prompt(game: AreYouTheTraitor, player: str) -> str:
    role = game.roles[player]
    reminders = {
        "hero":    "You are the HERO. Identify the Traitor through your public statement.",
        "traitor": "You are the TRAITOR. Blend in and deflect suspicion. Do NOT reveal your role.",
        "healer":  "You are a HEALER. Help identify the Traitor through observation.",
    }
    return f"""
{reminders[role]}

CURRENT GAME STATE:
{game.get_public_state()}

It is your turn to make a public statement. All players will see what you say.
Use your tools to review history, then make a strategic 1-2 sentence statement via make_statement.
""".strip()


def get_vote_prompt(game: AreYouTheTraitor, player: str) -> str:
    role = game.roles[player]
    reminders = {
        "hero":    "You are the HERO. Vote to eliminate who you believe is the Traitor.",
        "traitor": "You are the TRAITOR. Vote strategically — eliminate a threat or frame someone.",
        "healer":  "You are a HEALER. Vote to eliminate who you believe is the Traitor.",
    }
    others = [p for p in game.active_players if p != player]
    return f"""
{reminders[role]}

CURRENT GAME STATE:
{game.get_public_state()}

All players have made their statements. Cast your vote to eliminate one player.
You must choose from: {', '.join(others)}
You cannot vote for yourself.

Review the history, reason carefully, then cast your vote using the vote action.
""".strip()


# =============================================================================
# REACT LOOP — ONE PLAYER, ONE PHASE
# =============================================================================

def run_react_for_player(
    agent: Agent,
    game: AreYouTheTraitor,
    player: str,
    phase: str,       # 'statement' or 'vote'
    max_iter: int = 8,
) -> str:
    """
    Run the ReAct loop for one player in one phase.
    Returns the extracted statement text or the vote target name.
    Falls back gracefully if the agent fails to produce a valid result.
    """
    next_prompt = get_statement_prompt(game, player) if phase == "statement" else get_vote_prompt(game, player)
    result_value = None

    for _ in range(max_iter):
        result = agent(next_prompt)
        if not result:
            break

        parsed = parse_react_response(result)

        if parsed.get("has_pause") and parsed.get("action"):
            chosen_tool = parsed["action"].strip()
            arg = parsed.get("argument", "").strip()

            if chosen_tool == "get_game_state":
                obs = game.get_public_state()

            elif chosen_tool == "review_history":
                obs = game.get_full_history()

            elif chosen_tool == "make_statement" and phase == "statement":
                statement = arg.strip('"').strip("'").strip()
                if statement:
                    game.record_statement(player, statement)
                    result_value = statement
                    agent("Observation: Statement recorded.")
                    break
                else:
                    obs = "Statement cannot be empty. Try: make_statement: your text here"

            elif chosen_tool == "vote" and phase == "vote":
                others = [p for p in game.active_players if p != player]
                vote_target = next((p for p in others if p.lower() in arg.lower()), None)
                if vote_target:
                    game.record_vote(player, vote_target)
                    result_value = vote_target
                    agent(f"Observation: Vote cast for {vote_target}.")  # has placeholder
                    break
                else:
                    obs = f"Could not parse vote. Choose from: {', '.join(others)}"

            else:
                valid = "make_statement" if phase == "statement" else "vote"
                obs = f"Unknown action or wrong phase. Use: get_game_state, review_history, {valid}"

            next_prompt = f"Observation: {obs}"

        elif parsed.get("decision"):
            decision = parsed["decision"].strip()
            others = [p for p in game.active_players if p != player]
            if phase == "vote":
                vote_target = next((p for p in others if p.lower() in decision.lower()), None)
                if vote_target:
                    game.record_vote(player, vote_target)
                    result_value = vote_target
            elif phase == "statement" and not result_value:
                game.record_statement(player, decision)
                result_value = decision
            break

    # Fallbacks
    if result_value is None:
        others = [p for p in game.active_players if p != player]
        if phase == "vote":
            result_value = random.choice(others)
            game.record_vote(player, result_value)
            print(f"  ⚠  {player} failed to vote — randomly chose {result_value}")
        else:
            result_value = "I am observing carefully before making any accusations."
            game.record_statement(player, result_value)
            print(f"  ⚠  {player} failed to make a statement — using fallback")

    return result_value


# =============================================================================
# MAIN GAME RUNNER
# =============================================================================

def run_are_you_traitor(num_players: int = 4, max_rounds: int = 3, max_iter: int = 8):
    """Run a full Are You the Traitor? game with AI agents for each role."""
    client = create_client()
    game = AreYouTheTraitor(num_players=num_players, max_rounds=max_rounds)
    logger = TraceLogger(GAME_NAME)

    # One stateful Agent per player with a role-specific system prompt.
    # Agent.reset() is called between phases to keep each phase's context clean.
    agents = {
        player: Agent(
            client=client,
            system=build_system_prompt(build_role_system_prompt(game, player), game_name=GAME_NAME),
            name=player,
        )
        for player in game.players
    }

    print("=" * 60)
    print("  ARE YOU THE TRAITOR? — Multi-Agent Social Deduction")
    print("=" * 60)
    print(f"  Provider : {client.provider.upper()} / {client.default_model}")
    print(f"  Players: {num_players}  |  Rounds: {max_rounds}")
    print(f"  Roles: 1 Hero, 1 Traitor, {num_players - 2} Healer(s)")
    print("\n  Secret role assignments (revealed only on elimination):")
    for p, r in game.roles.items():
        marker = " <- TRAITOR" if r == "traitor" else (" <- HERO" if r == "hero" else "")
        print(f"    {p}: {r.upper()}{marker}")
    print("=" * 60)

    while not game.is_over():
        print(f"\n{'─'*60}")
        print(f"  ROUND {game.current_round}  |  Active: {', '.join(game.active_players)}")
        print(f"{'─'*60}")

        logger.start_round(game.current_round, {
            "active": game.active_players,
            "eliminated": game.eliminated,
        })

        # ── Phase 1: Discussion ───────────────────────────────────────────
        print("\n  DISCUSSION PHASE\n")
        for player in list(game.active_players):
            agents[player].reset()  # fresh history per phase
            print(f"  [{player} ({game.roles[player]}) is thinking...]")
            statement = run_react_for_player(agents[player], game, player, "statement", max_iter)
            print(f'  {player}: "{statement}"\n')

        # ── Phase 2: Voting ───────────────────────────────────────────────
        print("\n  VOTING PHASE\n")
        for player in list(game.active_players):
            agents[player].reset()  # fresh history for vote phase
            print(f"  [{player} ({game.roles[player]}) is voting...]")
            vote = run_react_for_player(agents[player], game, player, "vote", max_iter)
            print(f"  {player} votes to eliminate: {vote}\n")

        # ── Phase 3: Elimination ──────────────────────────────────────────
        eliminated_player, vote_counts = game.tally_votes()
        print("  ELIMINATION")
        print(f"  Vote counts: {dict(sorted(vote_counts.items(), key=lambda x: -x[1]))}")
        print(f"  Eliminated: {eliminated_player}")
        role_revealed, game_over = game.eliminate(eliminated_player)
        print(f"  Role revealed: {role_revealed.upper()}\n")

        logger.record_round_result(game.current_round - (0 if game_over else 1), {
            "eliminated": eliminated_player,
            "role_revealed": role_revealed,
            "vote_counts": vote_counts,
            "game_over": game_over,
        })

        if game_over:
            break

    # ── Final result ──────────────────────────────────────────────────────
    if game.winner is None:
        game.winner = "traitor"

    print(f"{'='*60}")
    print("  GAME OVER")
    print("=" * 60)
    if game.winner == "village":
        print("  VILLAGE WINS — The Traitor was found and eliminated!")
    else:
        print("  TRAITOR WINS — Survived all rounds undetected!")

    print("\n  All roles revealed:")
    for player in game.players:
        role = game.roles[player]
        status = "eliminated" if any(p == player for p, _ in game.eliminated) else "survived"
        print(f"    {player}: {role.upper():10}  [{status}]")

    logger.finish({
        "winner": game.winner,
        "eliminated": game.eliminated,
        "roles": game.roles,
    })
    trace_path = logger.save()
    print(f"\n  Trace saved to: {trace_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Are You the Traitor? — multi-agent social deduction")
    parser.add_argument("--players", type=int, default=4, help="Number of players (3–6)")
    parser.add_argument("--rounds", type=int, default=3, help="Maximum rounds")
    args = parser.parse_args()
    run_are_you_traitor(num_players=args.players, max_rounds=args.rounds)
