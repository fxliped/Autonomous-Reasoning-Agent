"""
AltruAgent tournament runner — repeated_pd.

Authenticates, joins the queue, and plays matches using TournamentAgent.
Opponent profiles are saved to agent/memory/ and persist across runs.

Setup (.env):
  ALTRUAGENT_API_KEY=sk_agent_xxx

Usage:
  python tournament/runner.py                        # join queue, play forever
  python tournament/runner.py --session <id>         # play one specific session
  python tournament/runner.py --tournament <id>      # resume a tournament
  python tournament/runner.py --signup "MyAgentName" # register once
"""

import sys
import json
import time
import os
from pathlib import Path

# Force line-buffered stdout so prints appear immediately even when piped to tee.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
from dotenv import load_dotenv

load_dotenv()

from tournament.core.agent import TournamentAgent  # noqa: E402

CONTROL = "https://llw83cu38l.execute-api.us-west-2.amazonaws.com"
GAME_SERVER_FALLBACK = "http://GameAp-Servi-tOMiXPVqPFIe-185462629.us-west-2.elb.amazonaws.com"

ACTION_COOPERATE = 0
ACTION_DEFECT = 1

POLL_INTERVAL = 2       # seconds between state polls
MAX_EMPTY_WAIT = 300    # seconds before auto-resign if legal_actions stays empty


# =============================================================================
# HTTP CLIENT
# =============================================================================

class AltruAgentClient:
    """Thin wrapper around the AltruAgent control-plane and GameAPI."""

    def __init__(self, api_key: str, game_server_url: str = GAME_SERVER_FALLBACK):
        self.api_key = api_key
        self.game_server = game_server_url
        self.access_token: str | None = None

    # --- auth ---

    def login(self) -> None:
        data = self._post("/auth/agent/login", {"api_key": self.api_key}, auth=False)
        self.access_token = data["access_token"]
        me = self._get("/auth/agent/me")
        self.agent_name: str = me.get("name", "")
        print(f"[Auth] Logged in as '{self.agent_name}'.")

    def _relogin_if_401(self, resp: requests.Response) -> bool:
        if resp.status_code == 401:
            print("[Auth] 401 — re-logging in once.")
            self.login()
            return True
        return False

    def check_claimed(self) -> bool:
        data = self._get("/auth/agent/me")
        status = data.get("status", "")
        if status != "claimed":
            print(f"[Auth] Agent status is '{status}' — not claimed yet. Stop.")
            return False
        return True

    # --- queues ---

    def list_queues(self) -> list[dict]:
        return self._get("/queues").get("queues", [])

    def join_queue(self, queue_id: str) -> str:
        """Join queue. Returns tournament_id."""
        data = self._post(f"/queues/{queue_id}/join", {})
        error = data.get("error")
        if error == "already_joined":
            print("[Queue] Already in this queue's waiting instance.")
        elif error == "queue_tournament_in_progress":
            tid = (data.get("next_actions") or [{}])[0].get("tournament_id") or ""
            print(f"[Queue] A tournament is already in progress: {tid}")
            return tid
        elif error:
            raise RuntimeError(f"Queue join error: {error} — {data.get('detail','')}")
        return data["tournament_id"]

    def poll_queue(self, queue_id: str) -> dict:
        return self._get(f"/queues/{queue_id}")

    # --- tournaments ---

    def join_tournament(self, tournament_id: str) -> dict:
        """Explicitly join a tournament by ID. Required when entering via admin share."""
        return self._post(f"/tournaments/{tournament_id}/join", {})

    def poll_tournament(self, tournament_id: str) -> dict:
        return self._get(f"/tournaments/{tournament_id}")

    # --- game API ---

    def get_game_state(self, session_id: str, game_server: str | None = None) -> dict:
        return self._get(f"/games/{session_id}", base=game_server or self.game_server)

    def step(self, session_id: str, action: int, game_server: str | None = None) -> dict:
        return self._post(
            f"/games/{session_id}/step",
            {"action": action},
            base=game_server or self.game_server,
        )

    def send_message(self, session_id: str, content: str,
                     game_server: str | None = None,
                     recipients: list[int] | None = None) -> dict:
        # recipients defaults to [1] (opponent in 2-player game per API docs).
        # Pass recipients=[0] if the state reveals we are player index 1.
        return self._post(
            f"/games/{session_id}/message",
            {"recipients": recipients if recipients is not None else [1],
             "content": content, "type": "chat"},
            base=game_server or self.game_server,
        )

    def terminate_messaging(self, session_id: str, game_server: str | None = None) -> dict:
        return self._post(
            f"/games/{session_id}/message",
            {"recipients": [], "type": "terminate"},
            base=game_server or self.game_server,
        )

    def resign(self, session_id: str, game_server: str | None = None) -> dict:
        return self._post(
            f"/games/{session_id}/resign", {},
            base=game_server or self.game_server,
        )

    # --- internals ---

    def _headers(self, auth: bool = True) -> dict:
        h = {"Content-Type": "application/json"}
        if auth and self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def _get(self, path: str, base: str = CONTROL) -> dict:
        resp = requests.get(f"{base}{path}", headers=self._headers(), timeout=30)
        if self._relogin_if_401(resp):
            resp = requests.get(f"{base}{path}", headers=self._headers(), timeout=30)
        return resp.json()

    def _post(self, path: str, body: dict, auth: bool = True,
              base: str = CONTROL) -> dict:
        resp = requests.post(f"{base}{path}", json=body,
                             headers=self._headers(auth), timeout=30)
        if auth and self._relogin_if_401(resp):
            resp = requests.post(f"{base}{path}", json=body,
                                 headers=self._headers(), timeout=30)
        return resp.json()


# =============================================================================
# STATE HELPERS
# =============================================================================

def _my_name(state: dict) -> str:
    """Our player name from current_player (set when it's our turn to act)."""
    return (state.get("current_player") or {}).get("name", "")


def _extract_match_history(state: dict, my_player_name: str) -> list[dict]:
    """
    Build match_history from state.round_history (all completed rounds).
    Each entry: {round, my_action, opp_action, my_pts, opp_pts, opp_msg}
    Messages are not in round_history — opp_msg is filled in separately by the runner.
    """
    history = []
    for rnd in state.get("round_history") or []:
        actions = rnd.get("actions") or {}
        labels = rnd.get("action_labels") or {}
        payoffs = rnd.get("payoffs") or {}

        opp_name = next((k for k in actions if k != my_player_name), "")
        my_action = labels.get(my_player_name, "?").lower()
        opp_action = labels.get(opp_name, "?").lower()
        my_pts = payoffs.get(my_player_name, 0.0)
        opp_pts = payoffs.get(opp_name, 0.0)

        history.append({
            "round": rnd.get("round", len(history) + 1),
            "my_action": my_action,
            "opp_action": opp_action,
            "my_pts": my_pts,
            "opp_pts": opp_pts,
            "opp_msg": "",  # filled in by play_session when messages are read
        })
    return history


def _find_opponent_message(state: dict, my_player_name: str) -> str:
    """Extract the opponent's message from state.new_messages (revealed in MOVING phase)."""
    for msg in state.get("new_messages") or []:
        if msg.get("type") == "chat" and msg.get("sender_name", "") != my_player_name:
            return msg.get("content", "")
        # fallback: sender by index — we are not index 0 if my_player_name maps to player 1
        if msg.get("type") == "chat" and "sender_name" not in msg:
            return msg.get("content", "")
    return ""


def _cumulative_scores(state: dict, my_player_name: str) -> tuple[float, float]:
    scores = state.get("cumulative_scores") or {}
    my_score = float(scores.get(my_player_name, 0.0))
    opp_score = float(next((v for k, v in scores.items() if k != my_player_name), 0.0))
    return my_score, opp_score


def _opponent_name(state: dict, my_player_name: str) -> str:
    scores = state.get("cumulative_scores") or {}
    return next((k for k in scores if k != my_player_name), "unknown")


# =============================================================================
# GAME LOOP
# =============================================================================

def play_session(
    client: AltruAgentClient,
    session_id: str,
    agent: TournamentAgent,
    game_server: str,
) -> None:
    """
    Play one repeated_pd session to completion.
    Handles messaging phase (blind) and moving phase separately.
    """
    print(f"\n[Session] Starting {session_id}")
    my_name = ""
    opp_name = ""
    message_sent_this_round: str = ""
    msg_log: dict[int, str] = {}      # round_num → opponent message (filled in MOVING phase)
    my_msg_log: dict[int, str] = {}   # round_num → agent's own message (filled in MESSAGING phase)
    empty_wait_start: float | None = None
    opp_recipient_index: int = 1      # opponent player index for send_message; resolved on first state
    _moving_phase_start: float | None = None  # when we first detected the moving phase for this round

    while True:
        state = client.get_game_state(session_id, game_server)

        if state.get("is_terminal"):
            returns = state.get("returns") or {}
            if isinstance(returns, dict):
                my_avg = float(returns.get(my_name, 0.0))
                opp_avg = float(next((v for k, v in returns.items() if k != my_name), 0.0))
            else:
                my_avg = opp_avg = 0.0
            print(f"\n[Session] Terminal. My avg score: {my_avg:.3f} | Opp: {opp_avg:.3f}")
            if agent.match_rounds:  # only persist if we actually played rounds
                agent.end_match(my_avg, opp_avg)
            else:
                print("[Session] No rounds played — skipping end_match.")
            return

        phase = state.get("phase", "moving")
        next_action = ((state.get("next_actions") or [{}])[0]).get("action", "")
        current_round = state.get("current_round", 1)
        total_rounds = state.get("total_rounds", 8)

        # Resolve our player name and opponent recipient index once
        if not my_name and state.get("current_player"):
            my_name = _my_name(state)
        if not opp_name and my_name:
            opp_name = _opponent_name(state, my_name)
            # Determine opponent's player index for targeted messaging
            players = state.get("players") or []
            if my_name in players:
                my_idx = players.index(my_name)
                opp_recipient_index = 1 - my_idx  # 0→1 or 1→0

        # --- WAIT ---
        if next_action == "wait_for_opponent":
            time.sleep(POLL_INTERVAL)
            continue

        if next_action == "game_over":
            returns = state.get("returns") or {}
            my_avg = float(returns.get(my_name, 0.0)) if isinstance(returns, dict) else 0.0
            opp_avg = float(next((v for k, v in returns.items() if k != my_name), 0.0)) if isinstance(returns, dict) else 0.0
            if agent.match_rounds:
                agent.end_match(my_avg, opp_avg)
            return

        # --- MESSAGING PHASE ---
        if phase == "messaging" or next_action in ("send_message", "terminate_messaging"):
            my_score, opp_score = _cumulative_scores(state, my_name)
            match_history = _extract_match_history(state, my_name)
            # Patch in stored messages for completed rounds
            for entry in match_history:
                if entry["round"] in msg_log:
                    entry["opp_msg"] = msg_log[entry["round"]]
                if entry["round"] in my_msg_log:
                    entry["my_msg"] = my_msg_log[entry["round"]]

            # Compose our message (blind — don't see opponent's yet)
            if next_action != "terminate_messaging":
                _msg_start = time.monotonic()
                try:
                    message_sent_this_round = agent.compose_message(
                        round_num=current_round,
                        total_rounds=total_rounds,
                        match_history=match_history,
                        my_score=my_score,
                        opp_score=opp_score,
                    )
                except Exception as e:
                    print(f"[Messaging] compose_message failed ({e}) — using fallback")
                    message_sent_this_round = (
                        "I cooperate with partners who cooperate. Mutual +2/round beats the gamble."
                    )
                print(f"[R{current_round}] Composed in {time.monotonic()-_msg_start:.1f}s")
                my_msg_log[current_round] = message_sent_this_round
                resp = client.send_message(session_id, message_sent_this_round, game_server,
                                           recipients=[opp_recipient_index])
                err = resp.get("error")
                if err == "messages_quota_exceeded":
                    print(f"[Messaging] Quota exceeded — terminating directly.")
                elif err:
                    print(f"[Messaging] Send error: {err}")

            # Terminate to advance to MOVING phase
            client.terminate_messaging(session_id, game_server)
            time.sleep(2)
            continue

        # --- MOVING PHASE ---
        if phase == "moving" or next_action == "make_move":
            # Track when we first entered this phase so the agent's time budget is accurate.
            if _moving_phase_start is None:
                _moving_phase_start = time.monotonic()

            legal = state.get("legal_actions") or []
            if not legal:
                # Waiting for opponent to finish messaging or move
                if empty_wait_start is None:
                    empty_wait_start = time.time()
                elif time.time() - empty_wait_start > MAX_EMPTY_WAIT:
                    print("[Session] Waited too long with empty legal_actions — resigning.")
                    client.resign(session_id, game_server)
                    return
                time.sleep(POLL_INTERVAL)
                continue
            empty_wait_start = None

            # Opponent's message from the messaging phase just revealed in new_messages
            opp_message = _find_opponent_message(state, my_name)
            msg_log[current_round] = opp_message

            my_score, opp_score = _cumulative_scores(state, my_name)
            match_history = _extract_match_history(state, my_name)
            for entry in match_history:
                if entry["round"] in msg_log:
                    entry["opp_msg"] = msg_log[entry["round"]]
                if entry["round"] in my_msg_log:
                    entry["my_msg"] = my_msg_log[entry["round"]]

            # Subtract time already spent in this phase so the agent doesn't exceed the
            # platform's 30s per-round limit (25s budget minus polling overhead).
            elapsed_in_phase = time.monotonic() - _moving_phase_start
            time_budget = max(8.0, 25.0 - elapsed_in_phase)

            action_int = agent.choose_action(
                round_num=current_round,
                total_rounds=total_rounds,
                opponent_message=opp_message,
                my_message=message_sent_this_round,
                match_history=match_history,
                my_score=my_score,
                opp_score=opp_score,
                time_budget_seconds=time_budget,
            )

            # Validate action is in legal_actions
            if action_int not in legal:
                action_int = legal[0]

            action_label = "Cooperate" if action_int == 0 else "Defect"
            print(f"[R{current_round}] Submitting: {action_label}")

            client.step(session_id, action_int, game_server)
            _moving_phase_start = None  # reset for next round

            # Retry reading round result — server may lag up to ~8s after step.
            last_rnd = {}
            for _ in range(4):
                time.sleep(POLL_INTERVAL)
                new_state = client.get_game_state(session_id, game_server)
                last_rnd = new_state.get("last_round") or {}
                if last_rnd.get("round") == current_round:
                    break

            if last_rnd.get("round") == current_round:
                actions = last_rnd.get("action_labels") or {}
                payoffs = last_rnd.get("payoffs") or {}
                opp_name_local = next((k for k in actions if k != my_name), opp_name)
                opp_action_label = actions.get(opp_name_local, "?").lower()
                my_pts = float(payoffs.get(my_name, 0.0))
                opp_pts = float(payoffs.get(opp_name_local, 0.0))
                agent.record_round_result(current_round, opp_action_label, my_pts, opp_pts)
                print(f"[R{current_round}] Result: I {action_label} | They {opp_action_label} | pts me {my_pts:+.0f} them {opp_pts:+.0f}")
            else:
                print(f"[R{current_round}] Warning: round result not available after retries — classification update skipped.")
            continue

        # Unknown state — poll
        time.sleep(POLL_INTERVAL)


# =============================================================================
# TOURNAMENT LOOP
# =============================================================================

def run_tournament(client: AltruAgentClient, tournament_id: str, verbose: bool = False) -> None:
    """Poll a tournament and play each child session as it becomes available."""
    print(f"\n[Tournament] Watching {tournament_id}")

    while True:
        t = client.poll_tournament(tournament_id)
        status = t.get("status") or t.get("tournament", {}).get("status", "")

        if status == "completed":
            print(f"[Tournament] Completed.")
            lb = t.get("leaderboard")
            if lb:
                print("[Tournament] Final leaderboard:")
                for entry in lb:
                    print(f"  #{entry.get('rank','?')} {entry.get('name','?')} — avg {entry.get('average_payoff','?')}")
            return

        # Discover child sessions to play
        viewer = t.get("viewer") or {}
        next_actions = viewer.get("next_actions") or []
        action_code = (next_actions[0].get("action") if next_actions else "") or ""

        if action_code == "tournament_complete":
            print("[Tournament] Received tournament_complete signal.")
            return

        if action_code == "play_child_session":
            session_id = next_actions[0].get("session_id") or ""
            game_server = next_actions[0].get("game_server_url") or GAME_SERVER_FALLBACK
            if not session_id:
                # also try active_child_session_ids
                ids = t.get("active_child_session_ids") or []
                session_id = ids[0] if ids else ""

            if session_id:
                # Determine opponent name from tournament roster
                roster = t.get("participants") or []
                my_agent_id = viewer.get("agent_id") or ""
                opp_entry = next((p for p in roster if p.get("agent_id") != my_agent_id), {})
                opp_id = opp_entry.get("name") or opp_entry.get("agent_id") or "unknown"

                # Disable debate in live games — 3 LLM calls can't fit in a 30s round budget.
                agent = TournamentAgent(opponent_id=opp_id, verbose=verbose, use_debate=False)

                # Inject leaderboard position + end-game targeting data
                lb = t.get("leaderboard") or []
                if lb:
                    my_name = getattr(client, "agent_name", "")
                    my_entry = next((e for e in lb if e.get("name") == my_name), None)
                    opp_lb = next((e for e in lb if e.get("name") == opp_id), None)

                    if my_entry:
                        my_rank = int(my_entry.get("rank", len(lb)))
                        my_score = float(my_entry.get("average_payoff") or 0.0)

                        # Score of the player ranked just above us (rank - 1)
                        above = next(
                            (e for e in lb if int(e.get("rank", 0)) == my_rank - 1),
                            None,
                        )
                        score_gap = (
                            my_score - float(above.get("average_payoff", my_score))
                            if above else None
                        )
                        # Matches remaining = total participants - 1 - matches played so far
                        matches_played = len(t.get("completed_sessions") or [])
                        matches_remaining = max(0, len(roster) - 1 - matches_played)

                        agent.update_leaderboard(
                            rank=my_rank,
                            total_players=len(lb),
                            score_gap_to_above=score_gap,
                            opponent_rank=int(opp_lb.get("rank", len(lb))) if opp_lb else None,
                            matches_remaining=matches_remaining,
                        )

                play_session(client, session_id, agent, game_server)  # verbose is in agent

        elif action_code == "wait_for_child_match":
            pass  # normal — no active match yet

        time.sleep(POLL_INTERVAL)


# =============================================================================
# QUEUE LOOP
# =============================================================================

def run_queue_loop(client: AltruAgentClient, queue_id: str, verbose: bool = False) -> None:
    """Join a queue, wait for the tournament instance, play it, then rejoin."""
    print(f"\n[Queue] Joining queue {queue_id}")

    while True:
        tournament_id = client.join_queue(queue_id)

        # Wait for tournament to start
        print(f"[Queue] In tournament {tournament_id}. Waiting for start...")
        while True:
            q = client.poll_queue(queue_id)
            waiting = q.get("current_waiting_tournament")
            if waiting is None:
                print("[Queue] Tournament started.")
                break
            secs = (waiting or {}).get("seconds_until_start", "?")
            print(f"[Queue] Waiting... ~{secs}s until start.")
            time.sleep(POLL_INTERVAL)

        run_tournament(client, tournament_id, verbose=verbose)

        print("[Queue] Tournament done. Rejoining queue in 10s...")
        time.sleep(10)


# =============================================================================
# ONE-TIME SIGNUP
# =============================================================================

def signup(name: str, description: str = "Strategic PD agent") -> None:
    """Register a new agent. Run once. Save the api_key printed."""
    resp = requests.post(
        f"{CONTROL}/auth/agent/signup",
        json={"name": name, "description": description},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    data = resp.json()
    if data.get("error") == "name_taken":
        print(f"Name '{name}' is already taken. Choose another.")
        return
    api_key = data.get("api_key", "")
    claim_token = data.get("claim_token", "")
    print(f"\nAgent '{name}' created!")
    print(f"  api_key:     {api_key}")
    print(f"  claim_token: {claim_token}")
    print(f"\nAdd to your .env:  ALTRUAGENT_API_KEY={api_key}")
    print("Return claim_token to your professor/admin to claim the agent.")
    print("Do NOT play until the agent is claimed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AltruAgent runner — repeated PD tournament")
    parser.add_argument("--signup", metavar="NAME",
                        help="Register a new agent with this name (run once)")
    parser.add_argument("--session", metavar="SESSION_ID",
                        help="Play a specific session directly")
    parser.add_argument("--tournament", metavar="TOURNAMENT_ID",
                        help="Join an existing tournament by ID")
    parser.add_argument("--queue", metavar="QUEUE_ID",
                        help="Join a specific queue by ID (default: auto-detect)")
    parser.add_argument("--opponent", metavar="NAME", default="unknown",
                        help="Opponent name for memory (used with --session)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full LLM reasoning, debate output, and Bayes breakdown")
    args = parser.parse_args()

    if args.signup:
        signup(args.signup)
        sys.exit(0)

    api_key = os.getenv("ALTRUAGENT_API_KEY") or os.getenv("altruagent_api_key")
    if not api_key:
        print("ERROR: Set ALTRUAGENT_API_KEY in your .env file.")
        sys.exit(1)

    client = AltruAgentClient(api_key=api_key)
    client.login()

    if not client.check_claimed():
        sys.exit(1)

    if args.session:
        agent = TournamentAgent(opponent_id=args.opponent, verbose=args.verbose)
        play_session(client, args.session, agent, GAME_SERVER_FALLBACK)

    elif args.tournament:
        # Explicitly join before polling (required when entering via admin-shared tournament ID)
        client.join_tournament(args.tournament)
        run_tournament(client, args.tournament, verbose=args.verbose)

    else:
        # Auto-detect queue
        queue_id = args.queue
        if not queue_id:
            queues = client.list_queues()
            pd_queues = [q for q in queues if q.get("game_type") == "repeated_pd"
                         and q.get("is_active")]
            if not pd_queues:
                print("No active repeated_pd queues found. Ask the admin to create one.")
                sys.exit(1)
            queue_id = pd_queues[0]["queue_id"]
            print(f"[Queue] Auto-selected queue: {queue_id} ({pd_queues[0].get('name','')})")

        run_queue_loop(client, queue_id, verbose=args.verbose)
