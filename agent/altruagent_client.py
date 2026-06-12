"""HTTP client for the AltruAgent competition platform.

Thin wrapper over the control plane (auth, competitions, tournaments, queues)
and the data plane / GameAPI (game state, moves, messaging). Handles the
bearer-token lifecycle and the platform's one-retry 401 re-login policy.

Security: api_key and access_token are ONLY ever sent to the control plane
and to the game_server_url returned by the control plane. Never anywhere else.

Docs: https://llw83cu38l.execute-api.us-west-2.amazonaws.com/skill.md
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

# Load ~/.env (walks up from this file's dir, same as agent.agent does).
load_dotenv()

CONTROL_PLANE = "https://llw83cu38l.execute-api.us-west-2.amazonaws.com"
GAME_SERVER_FALLBACK = (
    "http://GameAp-Servi-tOMiXPVqPFIe-185462629.us-west-2.elb.amazonaws.com"
)


def normalize_game_server_url(url: Optional[str]) -> str:
    """Prepend http:// if scheme-less; fall back to the documented default."""
    if not url:
        return GAME_SERVER_FALLBACK
    if url.startswith("http://") or url.startswith("https://"):
        return url.rstrip("/")
    return f"http://{url}".rstrip("/")


class AltruAgentError(Exception):
    """Raised on a non-recoverable platform error.

    Carries the machine-readable `error` code and the `recovery_action` dict
    from the platform's canonical error envelope so callers can branch on
    `err.code` / `err.recovery_action["action"]` instead of parsing prose.
    """

    def __init__(self, status: int, code: str, detail: str, recovery_action: Optional[dict] = None):
        self.status = status
        self.code = code
        self.detail = detail
        self.recovery_action = recovery_action or {}
        super().__init__(f"[{status} {code}] {detail}")


class AltruAgentClient:
    """Stateful client: holds the api_key, mints/refreshes the access_token."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        agent_name: Optional[str] = None,
        control_plane: str = CONTROL_PLANE,
        poll_interval: float = 5.0,
        timeout: float = 30.0,
        max_retries_429: int = 4,
        retry_backoff: float = 0.75,
    ):
        self.api_key = api_key or os.getenv("ALTRUAGENT_API_KEY")
        self.agent_name = agent_name or os.getenv("ALTRUAGENT_AGENT_NAME")
        if not self.api_key:
            raise ValueError(
                "No api_key. Set ALTRUAGENT_API_KEY in ~/.env or pass api_key=..."
            )
        self.control = control_plane.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.access_token: Optional[str] = None
        self._session = requests.Session()
        # 429 resilience: the game server throttles bursty message/step traffic.
        # A 429 is transient, so retry with bounded exponential backoff instead
        # of crashing the match. Budget is small (~0.75+1.5+3+6s worst case) so a
        # retry during the move phase still lands inside the ~30s move window.
        self.max_retries_429 = max_retries_429
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------------
    # Core request machinery
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        auth: bool = True,
        allow_relogin: bool = True,
    ) -> dict:
        """Issue one request, resilient to the platform's two soft errors.

        - 401 (when authed): re-login once and retry once (platform policy).
        - 429 rate_limit_exceeded: the game server throttles bursty message/step
          traffic. A 429 is transient, so instead of crashing the whole match we
          retry with bounded exponential backoff + jitter, then give up.

        Returns the parsed JSON dict. Raises AltruAgentError on other non-2xx, or
        on a 429 that survives every retry.
        """
        if auth and not self.access_token:
            self.login()

        backoff = self.retry_backoff
        for attempt in range(self.max_retries_429 + 1):
            resp = self._session.request(
                method, url, headers=self._headers(), json=json_body, timeout=self.timeout
            )

            if resp.status_code == 401 and auth and allow_relogin:
                # Platform policy: exactly one re-login + one retry, then give up.
                self.login()
                resp = self._session.request(
                    method, url, headers=self._headers(), json=json_body, timeout=self.timeout
                )

            if resp.status_code == 429 and attempt < self.max_retries_429:
                sleep_for = backoff + random.uniform(0.0, 0.5)
                endpoint = url.rsplit("/", 1)[-1] or url
                print(f"  [429 rate_limit on {method} .../{endpoint} -> "
                      f"retry {attempt + 1}/{self.max_retries_429} in {sleep_for:.1f}s]")
                time.sleep(sleep_for)
                backoff *= 2
                continue

            return self._parse(resp)

        return self._parse(resp)

    @staticmethod
    def _parse(resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if 200 <= resp.status_code < 300:
            return data
        raise AltruAgentError(
            status=resp.status_code,
            code=data.get("error", f"http_{resp.status_code}"),
            detail=data.get("detail", resp.text[:300]),
            recovery_action=data.get("recovery_action"),
        )

    # ------------------------------------------------------------------
    # Auth (control plane)
    # ------------------------------------------------------------------
    def login(self) -> str:
        """Exchange api_key for a fresh access_token (JWT). Safe to call again."""
        resp = self._session.post(
            f"{self.control}/auth/agent/login",
            json={"api_key": self.api_key},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        data = self._parse(resp)
        token = data.get("access_token")
        if not token:
            raise AltruAgentError(resp.status_code, "no_access_token", "login returned no token")
        self.access_token = token
        return token

    def me(self) -> dict:
        """GET /auth/agent/me — claim-gate check. status must be 'claimed' to play."""
        return self._request("GET", f"{self.control}/auth/agent/me")

    def is_claimed(self) -> bool:
        return self.me().get("status") == "claimed"

    # ------------------------------------------------------------------
    # Competitions (single matches) — control plane
    # ------------------------------------------------------------------
    def list_competitions(self) -> list[dict]:
        return self._request("GET", f"{self.control}/competitions").get("competitions", [])

    def get_competition(self, session_id: str) -> dict:
        return self._request("GET", f"{self.control}/competitions/{session_id}")

    def join_competition(self, session_id: str) -> dict:
        return self._request("POST", f"{self.control}/competitions/{session_id}/join", json_body={})

    # ------------------------------------------------------------------
    # Tournaments — control plane
    # ------------------------------------------------------------------
    def list_tournaments(self) -> dict:
        return self._request("GET", f"{self.control}/tournaments", auth=False)

    def get_tournament(self, tournament_id: str) -> dict:
        """Authenticated → response includes the `viewer` object (required for play)."""
        return self._request("GET", f"{self.control}/tournaments/{tournament_id}")

    def join_tournament(self, tournament_id: str) -> dict:
        return self._request("POST", f"{self.control}/tournaments/{tournament_id}/join", json_body={})

    # ------------------------------------------------------------------
    # Queues — control plane (canonical multi-agent entry point)
    # ------------------------------------------------------------------
    def list_queues(self) -> list[dict]:
        return self._request("GET", f"{self.control}/queues", auth=False).get("queues", [])

    def get_queue(self, queue_id: str) -> dict:
        return self._request("GET", f"{self.control}/queues/{queue_id}")

    def join_queue(self, queue_id: str) -> dict:
        """Returns {tournament_id, ...}. Save the tournament_id."""
        return self._request("POST", f"{self.control}/queues/{queue_id}/join", json_body={})

    # ------------------------------------------------------------------
    # Gameplay (data plane / GameAPI) — under game_server_url
    # ------------------------------------------------------------------
    def game_state(self, game_server_url: str, session_id: str) -> dict:
        base = normalize_game_server_url(game_server_url)
        return self._request("GET", f"{base}/games/{session_id}")

    def step(self, game_server_url: str, session_id: str, action: int) -> dict:
        """Submit a move. action is an int from the state's legal_actions."""
        base = normalize_game_server_url(game_server_url)
        return self._request(
            "POST", f"{base}/games/{session_id}/step", json_body={"action": int(action)}
        )

    def messages(self, game_server_url: str, session_id: str, since: int = -1) -> dict:
        """GET /games/<sid>/messages?since=<index> — exclusive message cursor.

        Returns every visible message with index > `since`. Under blind
        messaging the current round's opponent message stays hidden until the
        phase flips to MOVING, but past rounds remain visible — so this is the
        reliable read (state.new_messages only holds the current round and is
        empty while blind). Start `since` at -1 to fetch from the beginning.
        """
        base = normalize_game_server_url(game_server_url)
        return self._request(
            "GET", f"{base}/games/{session_id}/messages?since={int(since)}"
        )

    def send_message(
        self,
        game_server_url: str,
        session_id: str,
        content: str,
        recipients: Optional[list[int]] = None,
        msg_type: str = "chat",
    ) -> dict:
        base = normalize_game_server_url(game_server_url)
        body: dict[str, Any] = {"recipients": recipients or [], "type": msg_type}
        if msg_type == "chat":
            body["content"] = content
        return self._request("POST", f"{base}/games/{session_id}/message", json_body=body)

    def terminate_messaging(self, game_server_url: str, session_id: str) -> dict:
        base = normalize_game_server_url(game_server_url)
        return self._request(
            "POST", f"{base}/games/{session_id}/message", json_body={"recipients": [], "type": "terminate"}
        )

    def resign(self, game_server_url: str, session_id: str) -> dict:
        base = normalize_game_server_url(game_server_url)
        return self._request("POST", f"{base}/games/{session_id}/resign", json_body={})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def sleep(self) -> None:
        time.sleep(self.poll_interval)


def _verify_auth() -> None:
    """Standalone check: login, hit the claim gate, print status. No gameplay."""
    client = AltruAgentClient()
    print(f"Agent name (from env): {client.agent_name}")
    print("Logging in...")
    client.login()
    print("  access_token acquired (len="
          f"{len(client.access_token)} chars)")
    me = client.me()
    status = me.get("status")
    print(f"GET /auth/agent/me -> status = {status!r}")
    if status == "claimed":
        print("✅ Agent is CLAIMED and ready to play.")
        na = me.get("next_actions", [{}])
        if na:
            print(f"   next_action: {na[0].get('action')}")
    else:
        print("⛔ Agent is NOT claimed. Claim it in the web UI before playing.")


if __name__ == "__main__":
    _verify_auth()
