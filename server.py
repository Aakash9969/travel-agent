"""
FastAPI backend for the Mini Autonomous Travel Agent.

Real pieces only: real LLM tool-use (Anthropic Claude, NVIDIA NIM, or any
OpenAI-compatible endpoint), real Amadeus flight/hotel search, real
Open-Meteo weather, real on-disk memory, and a real .ics calendar file as
the final action.

Instead of blocking on a terminal input() for approval, the agent PAUSES
and emits an "approval_required" event; the frontend shows an approve/
decline control, and clicking it calls /api/approve, which resumes the
same conversation exactly where it left off.

Responses from /api/chat and /api/approve are newline-delimited JSON
(NDJSON) streams: each agent thought, tool call, and observation is
emitted as it happens so the browser can show the agent working in real
time. Each stream always ends with a single {"type": "done", ...} event.

Run with:
    uvicorn server:app --reload
"""

import os
import json
import threading
from typing import Dict, Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import tools
import llm
from memory import Memory

app = FastAPI(title="Mini Autonomous Travel Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = Memory()

# Single local user -> one in-memory session is enough for this tool, but we
# still guard it with a lock so a second request while the agent is mid-turn
# can't corrupt the conversation state.
SESSION_LOCK = threading.Lock()
SESSION: Dict = {
    "messages": [],  # provider-native conversation messages
    "pending": None,             # {"tool_use_id": ..., "input": ...}
    "pending_tool_results": [],  # results from sibling tool calls in the same turn
}

# Neutral tool definitions; llm.py converts them to the provider's format.
TOOL_DEFS = [
    {
        "name": "search_flights",
        "description": "Search real flight offers between two places on a given date via the Amadeus API.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin city name or IATA code"},
                "destination": {"type": "string", "description": "Destination city name or IATA code"},
                "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["origin", "destination", "departure_date"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Search real hotel offers in a city for given check-in/check-out dates via the Amadeus API.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name or IATA city code"},
                "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
    {
        "name": "check_weather",
        "description": "Get a real live weather forecast for a city and date via Open-Meteo (covers roughly the next 16 days only).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["city", "date"],
        },
    },
    {
        "name": "remember_preference",
        "description": "Persist a user preference to long-term memory for future trips (e.g. preferred_airline, max_budget, hotel_stars_min, home_airport, seat_class - or any custom key).",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "book_trip",
        "description": (
            "Finalize the trip: writes a real .ics calendar file to disk and logs the "
            "trip in memory. This ALWAYS pauses for explicit human approval in the "
            "interface before it actually runs - you never need to ask the user "
            "separately in text first, the approval UI handles that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_datetime": {"type": "string", "description": "ISO 8601, e.g. 2026-12-10T09:00:00"},
                "end_datetime": {"type": "string", "description": "ISO 8601"},
                "destination": {"type": "string"},
                "flight_summary": {"type": "string"},
                "hotel_summary": {"type": "string"},
                "total_cost": {"type": "string"},
            },
            "required": ["title", "start_datetime", "end_datetime", "destination"],
        },
    },
]

SYSTEM_PROMPT = """You are an autonomous travel-planning agent embedded in a web app.

Lifecycle you must follow:
1. Understand the user's goal from their request.
2. Consult remembered preferences (given below) before planning.
3. Plan which tools you need (flights, hotels, weather).
4. Call tools, observe real results, and reason about the best option
   given the user's preferences and budget.
5. If something fails or exceeds budget, revise your plan (re-plan) and
   try again with different parameters rather than giving up.
6. Once you have a concrete itinerary, call book_trip with the details.
   The interface itself will pause and show the user an approve/decline
   control before book_trip actually executes - you do not need to ask
   for approval in text yourself, just call the tool once you have a
   final, priced plan you'd recommend.
7. If you learn a durable preference (e.g. they mention a budget or an
   airline they like), call remember_preference.

Known long-term preferences (from memory):
{memory_context}

Be concise and concrete. Always use real prices/times returned by tools,
never invent numbers.
"""


def execute_tool(name: str, tool_input: dict) -> dict:
    try:
        if name == "search_flights":
            return {"ok": True, "flights": tools.search_flights(
                tool_input["origin"], tool_input["destination"], tool_input["departure_date"]
            )}
        if name == "search_hotels":
            return {"ok": True, "hotels": tools.search_hotels(
                tool_input["city"], tool_input["check_in"], tool_input["check_out"]
            )}
        if name == "check_weather":
            return {"ok": True, "weather": tools.check_weather(tool_input["city"], tool_input["date"])}
        if name == "remember_preference":
            memory.set(tool_input["key"], tool_input["value"])
            return {"ok": True, "saved": {tool_input["key"]: tool_input["value"]}}
        if name == "book_trip":
            path = tools.create_calendar_event(
                title=tool_input["title"],
                start_iso=tool_input["start_datetime"],
                end_iso=tool_input["end_datetime"],
                description=(
                    f"Flight: {tool_input.get('flight_summary', '')}\n"
                    f"Hotel: {tool_input.get('hotel_summary', '')}\n"
                    f"Total: {tool_input.get('total_cost', '')}"
                ),
                location=tool_input["destination"],
            )
            memory.log_trip({
                "destination": tool_input["destination"],
                "flight_summary": tool_input.get("flight_summary"),
                "hotel_summary": tool_input.get("hotel_summary"),
                "total_cost": tool_input.get("total_cost"),
                "calendar_file": path,
            })
            return {"ok": True, "calendar_file_written_to": path}
        return {"ok": False, "error": f"Unknown tool '{name}'"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stream_turn() -> Iterator[Dict]:
    """Runs LLM turns + real tool calls, yielding each event as it happens.

    Stops when the agent is done talking, or when it calls book_trip and we
    must pause for human approval. Callers are responsible for emitting the
    final {"type": "done"} event so the client can always reset its state.
    """
    system = SYSTEM_PROMPT.format(memory_context=memory.as_context_string())
    provider = llm.get_provider()

    while True:
        turn = provider.complete(system=system, tools=TOOL_DEFS,
                                 messages=SESSION["messages"])
        SESSION["messages"].append(turn.raw)

        if turn.text.strip():
            yield {"type": "text", "content": turn.text.strip()}

        if not turn.tool_calls:
            if not turn.text.strip():
                # Thinking models can burn the whole token budget on
                # reasoning and return nothing - surface that honestly.
                yield {"type": "error",
                       "content": "The model returned an empty response (it may have spent "
                                  "its token budget thinking). Try again."}
            return

        results = []
        pending_call = None
        for call in turn.tool_calls:
            yield {"type": "tool_call", "name": call.name, "input": call.input}
            if call.name == "book_trip":
                # Never execute this one directly - pause for human approval.
                pending_call = call
                continue
            result = execute_tool(call.name, call.input)
            event = {"type": "observation", "name": call.name, "result": result}
            if call.name == "remember_preference":
                event["memory"] = memory.data
            yield event
            results.append({"call_id": call.id, "content": json.dumps(result)})

        if pending_call is not None:
            SESSION["pending"] = {"call_id": pending_call.id, "input": pending_call.input}
            SESSION["pending_tool_results"] = results
            yield {
                "type": "approval_required",
                "pending_approval": pending_call.input,
                "memory": memory.data,
            }
            return

        SESSION["messages"].extend(provider.tool_results(results))


def done_event() -> Dict:
    pending_input = SESSION["pending"]["input"] if SESSION.get("pending") else None
    return {"type": "done", "memory": memory.data, "pending_approval": pending_input}


class ChatRequest(BaseModel):
    message: str


class ApproveRequest(BaseModel):
    approved: bool


def guarded_stream(work) -> "Iterator[str]":
    """Runs a turn inside the session lock, streaming NDJSON lines; always
    ends with a done event and always releases the lock, even if the client
    disconnects mid-stream (the generator gets closed -> finally runs)."""
    def generate():
        if not SESSION_LOCK.acquire(blocking=False):
            yield json.dumps({"type": "error",
                              "content": "The agent is already working on a request - wait for it to finish."}) + "\n"
            yield json.dumps({"type": "done", "memory": memory.data, "pending_approval": None}) + "\n"
            return
        try:
            for event in work():
                yield json.dumps(event, default=str) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error",
                              "content": llm.get_provider().friendly_error(e)}) + "\n"
        finally:
            SESSION_LOCK.release()
        yield json.dumps(done_event(), default=str) + "\n"
    return generate()


@app.post("/api/chat")
def chat(req: ChatRequest):
    def work():
        results = []
        if SESSION.get("pending"):
            # A new message while a booking awaits approval implicitly
            # declines it. The pending tool call must still receive a
            # tool result or the conversation state would be invalid.
            pending = SESSION.pop("pending")
            results = SESSION.pop("pending_tool_results", [])
            results.append({
                "call_id": pending["call_id"],
                "content": json.dumps({
                    "ok": False,
                    "error": "User sent a new message instead of approving - "
                             "booking declined. Do not book this trip.",
                }),
            })
            yield {"type": "observation", "name": "book_trip",
                   "result": {"ok": False,
                              "error": "Pending booking declined: the user sent a new message."}}
        provider = llm.get_provider()
        SESSION["messages"].extend(
            provider.tool_results(results, followup_text=req.message))
        yield from stream_turn()
    return StreamingResponse(guarded_stream(work), media_type="application/x-ndjson")


@app.post("/api/approve")
def approve(req: ApproveRequest):
    def work():
        if not SESSION.get("pending"):
            yield {"type": "error", "content": "Nothing is pending approval."}
            return

        pending = SESSION.pop("pending")
        results = SESSION.pop("pending_tool_results", [])

        if req.approved:
            result = execute_tool("book_trip", pending["input"])
        else:
            result = {"ok": False, "error": "User declined approval. Do not book this trip."}

        yield {"type": "observation", "name": "book_trip", "result": result, "memory": memory.data}
        results.append({"call_id": pending["call_id"], "content": json.dumps(result)})
        SESSION["messages"].extend(llm.get_provider().tool_results(results))
        yield from stream_turn()
    return StreamingResponse(guarded_stream(work), media_type="application/x-ndjson")


@app.get("/api/memory")
def get_memory():
    return memory.data


@app.get("/api/config")
def get_config():
    """Lets the frontend show the active model and warn about missing setup."""
    try:
        provider = llm.get_provider()
        active = {"provider": provider.name, "model": provider.model}
    except Exception:
        active = {"provider": None, "model": None}
    return {
        **active,
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "nvidia_key_set": bool(os.environ.get("NVIDIA_API_KEY")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "amadeus_keys_set": bool(os.environ.get("AMADEUS_CLIENT_ID")
                                 and os.environ.get("AMADEUS_CLIENT_SECRET")),
    }


@app.post("/api/reset")
def reset():
    """Starts a fresh conversation. Long-term memory (agent_memory.json) is
    intentionally kept - preferences and past trips carry over."""
    with SESSION_LOCK:
        SESSION["messages"] = []
        SESSION["pending"] = None
        SESSION["pending_tool_results"] = []
    return {"ok": True}


# Serve the frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def missing_frontend():
        return JSONResponse(
            {"error": "frontend/index.html not found - the UI lives in frontend/. See README.md."},
            status_code=404,
        )
