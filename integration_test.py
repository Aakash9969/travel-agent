"""
Integration test for the full agent loop WITHOUT any API keys.

A FakeProvider scripts the LLM side per turn, while everything else is
real: real weather (Open-Meteo), real memory writes, real book_trip ->
.ics creation, and the real streaming/pause/resume endpoint flow.

The whole scenario runs twice - once in the Anthropic message format,
once in the OpenAI-compatible format (what NVIDIA NIM uses) - and the
fake validates the conversation state invariants for each format, so
both provider adapters are proven to keep the session valid.

Scenario per format:
  1. chat        -> agent calls check_weather, then book_trip -> approval_required
  2. decline     -> no calendar file written, agent re-plans and answers
  3. chat        -> agent saves a preference, then book_trip -> approval_required
  4. new message while pending -> implicit decline, state stays valid
  5. approve     -> .ics file is really written, trip logged in memory

Run with:
    python integration_test.py
"""

import json
import os
import sys
import tempfile

import llm
from llm import ModelTurn, ToolCall
from memory import Memory
from fastapi.testclient import TestClient

import server


class FakeProvider(llm.BaseProvider):
    """Scripts model turns and validates the session state invariants."""

    def __init__(self, label: str, fmt: str, script):
        super().__init__(model=f"fake-{label}")
        self.name = label
        self.fmt = fmt
        self.script = list(script)
        self.turns_seen = 0

    # ---- state validation ----
    def _validate_session(self, messages):
        if self.fmt == "anthropic":
            if not messages:
                return
            last = messages[-1]
            # After an assistant message with tool_use blocks, the next
            # message must be a user message with matching tool_results.
            for i, msg in enumerate(messages):
                content = msg.get("content")
                if msg.get("role") == "assistant" and isinstance(content, list):
                    used_ids = {b.id for b in content if getattr(b, "type", "") == "tool_use"}
                    if used_ids and i + 1 < len(messages):
                        nxt = messages[i + 1]
                        assert nxt.get("role") == "user", f"tool_use must be followed by user msg: {nxt}"
                        got = {b.get("tool_use_id") for b in nxt["content"]
                               if isinstance(b, dict) and b.get("type") == "tool_result"}
                        assert used_ids <= got, f"missing tool_result for {used_ids - got}"
        else:  # openai
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    ids = [c["id"] for c in msg["tool_calls"]]
                    assert i + 1 < len(messages), "assistant tool_calls must be followed by tool results"
                    nxt = messages[i + 1]
                    assert nxt.get("role") == "tool", f"expected tool msg, got {nxt}"
                    assert nxt.get("tool_call_id") == ids[0], "tool results must follow call order"

    # ---- BaseProvider interface ----
    def complete(self, system, tools, messages):
        self._validate_session(messages)
        self.turns_seen += 1
        assert self.script, f"FakeProvider ({self.name}) ran out of scripted turns"
        assert any(t["name"] == "check_weather" for t in tools) if self.turns_seen == 1 else True
        turn = self.script.pop(0)
        # raw messages in the provider's native format
        if self.fmt == "anthropic":
            blocks = []
            if turn.text:
                from anthropic.types import TextBlock
                blocks.append(TextBlock(type="text", text=turn.text))
            for c in turn.tool_calls:
                from anthropic.types import ToolUseBlock
                blocks.append(ToolUseBlock(type="tool_use", id=c.id, name=c.name, input=c.input))
            raw = {"role": "assistant", "content": blocks}
        else:
            raw = {"role": "assistant", "content": turn.text or "",
                   **({"tool_calls": [
                       {"id": c.id, "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.input)}}
                       for c in turn.tool_calls]} if turn.tool_calls else {})}
        return ModelTurn(text=turn.text, tool_calls=list(turn.tool_calls), raw=raw)

    def tool_results(self, results, followup_text=None):
        if self.fmt == "anthropic":
            content = [{"type": "tool_result", "tool_use_id": r["call_id"], "content": r["content"]}
                      for r in results]
            if followup_text:
                content.append({"type": "text", "text": followup_text})
            return [{"role": "user", "content": content}]
        messages = [{"role": "tool", "tool_call_id": r["call_id"], "content": r["content"]}
                    for r in results]
        if followup_text:
            messages.append({"role": "user", "content": followup_text})
        return messages

    def user_text(self, text):
        return ({"role": "user", "content": text})


def turn(text="", calls=None):
    return ModelTurn(text=text, tool_calls=calls or [])


def call(cid, name, **input):
    return ToolCall(id=cid, name=name, input=input)


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def run_scenario(fmt: str, label: str):
    print(f"\n===== format: {label} =====")
    tmp = tempfile.mkdtemp(prefix=f"ta_integration_{fmt}_")
    server.memory = Memory(path=os.path.join(tmp, "memory.json"))
    import tools
    ics_dir = os.path.join(tmp, "itineraries")
    real_create = tools.create_calendar_event

    def create_to_tmp(*args, **kwargs):
        kwargs["out_dir"] = ics_dir
        return real_create(*args, **kwargs)

    tools.create_calendar_event = create_to_tmp

    http = TestClient(server.app)
    http.post("/api/reset")

    def install(script):
        llm.set_provider(FakeProvider(label, fmt, script))

    # ---- 1. chat: weather tool call, then book_trip -> pause ----
    install([
        turn(calls=[call("c1", "check_weather", city="Goa", date="2026-10-15")]),
        turn(calls=[call("c2", "book_trip",
                         title="Goa getaway", start_datetime="2026-10-20T09:00:00",
                         end_datetime="2026-10-22T18:00:00", destination="Goa",
                         flight_summary="IND-122 BLR->GOI", hotel_summary="Beach resort",
                         total_cost="USD 412")]),
    ])
    r = http.post("/api/chat", json={"message": "Plan a Goa trip"})
    ev = events(r)
    types = [e["type"] for e in ev]
    assert types[-1] == "done", types
    assert types[0] == "tool_call" and ev[0]["name"] == "check_weather", ev[0]
    assert "observation" in types and "ok" in ev[1]["result"], ev[1]
    assert "approval_required" in types, types
    pending = ev[-2]["pending_approval"]
    assert pending["title"] == "Goa getaway" and pending["destination"] == "Goa", pending
    assert ev[-1]["pending_approval"] is not None, "done must echo the pending approval"
    print("1. chat -> tool calls + approval_required      OK")

    # ---- 2. decline: nothing booked, agent re-plans ----
    install([turn("No problem - tell me what to change.")])
    r = http.post("/api/approve", json={"approved": False})
    ev = events(r)
    types = [e["type"] for e in ev]
    book_obs = [e for e in ev if e["type"] == "observation" and e["name"] == "book_trip"]
    assert book_obs and book_obs[0]["result"]["ok"] is False, ev
    assert "text" in types and ev[-1]["type"] == "done", types
    assert ev[-1]["pending_approval"] is None
    assert not os.path.exists(ics_dir), "decline must not write an .ics file"
    assert server.memory.data["past_trips"] == [], "decline must not log a trip"
    print("2. decline -> nothing booked, agent re-plans   OK")

    # ---- 3. new plan with a remembered preference ----
    install([
        turn(calls=[call("c3", "remember_preference", key="likes_beaches", value="yes")]),
        turn(calls=[call("c4", "book_trip",
                         title="Goa getaway v2", start_datetime="2026-10-20T09:00:00",
                         end_datetime="2026-10-22T18:00:00", destination="Goa",
                         flight_summary="IND-122", hotel_summary="Beach resort",
                         total_cost="USD 390")]),
    ])
    r = http.post("/api/chat", json={"message": "Try a cheaper plan"})
    ev = events(r)
    types = [e["type"] for e in ev]
    assert "approval_required" in types, types
    mem_event = [e for e in ev if e["type"] == "observation" and e["name"] == "remember_preference"]
    assert mem_event and mem_event[0]["memory"]["likes_beaches"] == "yes", "memory update must stream to the UI"
    ctx = server.memory.as_context_string()
    assert "likes_beaches: yes" in ctx, ctx
    print("3. chat -> preference saved + new approval     OK")

    # ---- 4. new message while pending -> implicit decline, state stays valid ----
    install([turn("Sure - what would you like to change?")])
    r = http.post("/api/chat", json={"message": "Actually make it November"})
    ev = events(r)
    types = [e["type"] for e in ev]
    assert "error" not in types, ev
    assert any(e["type"] == "observation" and e["name"] == "book_trip"
               and e["result"]["ok"] is False for e in ev), "pending booking must be implicitly declined"
    assert any(e["type"] == "text" for e in ev), types
    assert ev[-1]["pending_approval"] is None, "stale approval must be cleared"
    r = http.post("/api/approve", json={"approved": True})
    ev = events(r)
    assert any(e["type"] == "error" and "Nothing is pending" in e["content"] for e in ev), ev
    assert not os.path.exists(ics_dir), "no booking must happen from the superseded approval"
    print("4. new message while pending -> implicit decline OK")

    # ---- 5. final plan, approve -> real .ics written + trip logged ----
    install([
        turn(calls=[call("c5", "book_trip",
                         title="Goa getaway v3", start_datetime="2026-11-20T09:00:00",
                         end_datetime="2026-11-22T18:00:00", destination="Goa",
                         flight_summary="IND-122", hotel_summary="Beach resort",
                         total_cost="USD 390")]),
        turn("Booked! Calendar file is ready."),
    ])
    r = http.post("/api/chat", json={"message": "November please"})
    ev = events(r)
    assert any(e["type"] == "approval_required" for e in ev), [e["type"] for e in ev]

    r = http.post("/api/approve", json={"approved": True})
    ev = events(r)
    book_obs = [e for e in ev if e["type"] == "observation" and e["name"] == "book_trip"]
    assert book_obs and book_obs[0]["result"]["ok"] is True, ev
    ics_path = book_obs[0]["result"]["calendar_file_written_to"]
    assert os.path.isfile(ics_path), "approve must write a real .ics file"
    with open(ics_path, "rb") as f:
        content = f.read().decode("utf-8")
    assert "SUMMARY:Goa getaway v3" in content.replace("\r\n ", "") and "END:VCALENDAR" in content
    assert server.memory.data["past_trips"], "approve must log the trip in memory"
    assert server.memory.data["past_trips"][0]["total_cost"] == "USD 390"
    assert any(e["type"] == "text" and "Booked" in e["content"] for e in ev), ev
    print("5. approve -> .ics written + trip logged       OK")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_scenario("anthropic", "anthropic")
    run_scenario("openai", "nvidia")
    print("\nAll integration tests passed (both provider formats).")
