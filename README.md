# Mini Autonomous Travel Agent — Web Version

A real full-stack agent: FastAPI backend + browser frontend. An LLM (Claude
via Anthropic, or GLM / Kimi / DeepSeek via NVIDIA NIM, or any
OpenAI-compatible endpoint) does the planning and reasoning; real Amadeus
APIs provide flights and hotels; Open-Meteo provides live weather;
preferences persist to disk; and a real `.ics` calendar file is written
once a human approves the booking.

## What you see

- **Chat** — talk to the agent like a travel consultant; it replies with a
  concrete, priced plan.
- **Agent activity** (right side) — every tool call and result streams in
  *as it happens*, so you watch the agent search flights, check hotels,
  pull live weather, and reason about the best option in real time.
- **Memory** — the agent's long-term memory panel: preferences and past
  trips, persisted to `agent_memory.json` across restarts.
- **Approval** — when the agent has a final plan, an amber panel asks you
  to **Approve** or **Decline** before anything is booked. Approve → a real
  `.ics` file is written; Decline → nothing is booked and the agent re-plans.
- **Model pill** (header) — which provider and model the agent is running
  on, plus a live "thinking… Ns" counter while it works (thinking models
  on NIM can take 30–90s per turn).

## Architecture

```
 ┌─────────────────────────┐   HTTP, NDJSON streams    ┌───────────────────────────┐
 │        Browser          │ <────────────────────────>│      FastAPI backend       │
 │  frontend/index.html    │   POST /api/chat           │        server.py           │
 │  - chat thread          │   POST /api/approve        │  - runs the LLM tool-use   │
 │  - live activity log    │   GET  /api/memory         │    loop (streaming)        │
 │  - memory panel         │   GET  /api/config         │  - executes real tools     │
 │  - approval panel       │   POST /api/reset          │  - pauses on book_trip     │
 └─────────────────────────┘                            └─────────┬───────────────────┘
                                            llm.py (provider layer)
                                                                   │
                                        ┌─────────────────────────┼─────────────────────┐
                                        ▼                         ▼                     ▼
                              ┌──────────────────┐    ┌──────────────────┐   ┌─────────────────────┐
                              │  Anthropic API    │    │   Amadeus API     │   │   Open-Meteo API     │
                              │        or         │    │  (real flight &   │   │  (real live weather) │
                              │  NVIDIA NIM / any │    │   hotel search)   │   └─────────────────────┘
                              │  OpenAI-compatible │    └──────────────────┘
                              │  endpoint — plans,│
                              │  reasons, decides │
                              │  which tools to   │
                              │  call)            │
                              └────────┬─────────┘
                                       │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
               ┌────────────────────┐        ┌────────────────────┐
               │ agent_memory.json   │        │  itineraries/*.ics │
               │ (persisted prefs   │        │ (real calendar file│
               │  and trip history)  │        │  written on approval)│
               └────────────────────┘        └────────────────────┘
```

**Streaming:** `POST /api/chat` and `POST /api/approve` return
newline-delimited JSON. Each event (`text`, `tool_call`, `observation`,
`approval_required`, `error`) is emitted the moment it happens, and the
stream always ends with a single `done` event.

**Why the backend pauses instead of blocking on input():** a web request
can't sit and wait for a terminal keypress. So when the LLM decides to call
`book_trip`, the backend stores exactly where the conversation is (which
tool call is pending, what results are already collected) and streams an
`approval_required` event to the browser instead of executing it. Your
click on Approve/Decline calls `/api/approve`, which resumes the *same*
conversation from that exact point — a standard human-in-the-loop pattern
for web-based agents. If you type a new message while a booking is pending,
the pending booking is implicitly declined and the conversation stays valid.

**Concurrency:** a single session lock guards the conversation, so a
double-clicked "Send" can't interleave two agent turns.

## 1. Install Python

Python 3.9+ required. Check: `python --version`

## 2. Get the files onto your computer

Keep this structure:

```
travel_agent_web/
├── server.py            # FastAPI app: agent loop, endpoints, static serving
├── llm.py               # LLM provider layer (Anthropic / NVIDIA NIM / OpenAI-compatible)
├── tools.py             # Amadeus flights+hotels, Open-Meteo weather, .ics writer
├── memory.py            # agent_memory.json persistence
├── requirements.txt
├── .env.example
├── frontend/
│   └── index.html       # the whole UI (vanilla HTML/CSS/JS, no build step)
└── (tests)              # smoke_test.py, integration_test.py
```

## 3. Create a virtual environment and install dependencies

```bash
cd travel_agent_web
python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 4. Get your API keys

**LLM provider — one of these (required):**

| Provider | Where to get a key | Notes |
|---|---|---|
| **NVIDIA NIM** | https://build.nvidia.com (free credits) | Default. Set `NVIDIA_API_KEY`. Default model `z-ai/glm-5.3-flash` — solid tool calling at interactive latency; the full `z-ai/glm-5.3` is a heavy thinking model (2+ min/turn). Other good options: `moonshotai/kimi-k3`, `deepseek-ai/deepseek-v4.1-flash`. |
| **Anthropic** | https://console.anthropic.com/settings/keys (paid) | Set `ANTHROPIC_API_KEY`. Default model `claude-sonnet-5-5`. |
| **Any OpenAI-compatible** | any `/v1/chat/completions` endpoint (OpenAI, vLLM, Ollama, LM Studio) | Set `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL`. |

Whichever key is present wins (priority: anthropic → nvidia → openai); set
`LLM_PROVIDER` in `.env` to force one. The model must support function/tool
calling.

**Amadeus (required for real flight/hotel search)** —
https://developers.amadeus.com → free account → create an app in "My
Self-Service Workspace" → copy the API Key (Client ID) and API Secret
(Client Secret) from the free test environment. No card required.

## 5. Configure your environment file

```bash
cp .env.example .env
```

Minimal working setup with NVIDIA:

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=z-ai/glm-5.3-flash
AMADEUS_CLIENT_ID=...
AMADEUS_CLIENT_SECRET=...
```

If anything is missing, the page shows a **Setup needed** banner as soon as
it loads, so you know before typing a word.

## 6. Run it

```bash
uvicorn server:app --reload
```

Open **http://127.0.0.1:8000** and try:

> Plan a 3-day trip from Bangalore to Goa starting Dec 10, budget around $400, I prefer window seats

Watch the right-hand console: you'll see real tool calls (`search_flights`,
`search_hotels`, `check_weather`) fire with their real results underneath
*while the agent works*, your memory panel populate, and — once the agent
has a final plan — an amber **Approval needed** panel appear. Click
**Approve booking** and a real `.ics` file is written to `itineraries/`;
click **Decline** and nothing is booked, and the agent will re-plan or ask
what you'd like instead.

**Reset** (top right) starts a fresh conversation while keeping long-term
memory — your preferences and past trips carry over.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST `{"message": "..."}` | Runs the agent turn; NDJSON stream of events, ends with `done` |
| `/api/approve` | POST `{"approved": true/false}` | Resumes a paused `book_trip`; NDJSON stream like `/api/chat` |
| `/api/memory` | GET | The full memory object (`agent_memory.json`) |
| `/api/config` | GET | Active provider + model, and whether API keys are configured |
| `/api/reset` | POST | Clears the conversation (memory is kept) |

## Tests

No API keys needed — both suites run offline (weather uses the free,
keyless Open-Meteo API; the integration test fakes only the LLM side):

```bash
python smoke_test.py        # memory context string + RFC 5545 .ics output + live weather
python integration_test.py  # full agent loop in BOTH message formats (Anthropic + OpenAI/NVIDIA)
```

The integration test scripts the model responses but exercises everything
else for real: streaming events, the approval pause/resume, implicit
decline on a new message, preference persistence, and the calendar file
written on approval. It also validates conversation-state invariants for
both provider formats.

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app: the streaming LLM tool-use loop, tool execution, approval pause/resume, and static file serving |
| `llm.py` | Provider layer: Anthropic SDK or OpenAI-compatible clients (NVIDIA NIM default), normalized tool calls, friendly errors |
| `tools.py` | Real API calls: Amadeus flights/hotels, Open-Meteo weather, RFC 5545-compliant `.ics` file creation |
| `memory.py` | Reads/writes `agent_memory.json` |
| `frontend/index.html` | The entire UI — chat, activity log, memory panel, approval panel (vanilla HTML/CSS/JS, no build step) |
| `smoke_test.py` | Offline tests: memory context, `.ics` format, live weather (no keys needed) |
| `integration_test.py` | Full agent-loop test with a scripted LLM, in both provider formats (no keys needed) |
| `agent_memory.json` | Created automatically on first run |
| `itineraries/` | Created automatically — `.ics` files from approved bookings |

## Deploying to the internet

**This is a Python backend, not a static site — Netlify/Vercel/GitHub Pages
cannot run it** (they only serve static files, so every `/api/*` call would
404 and you'd see "page not found" or a dead UI). Deploy to a host that
runs Python web services:

**Render (easiest, has a free tier):**

1. Push this repo to GitHub (done).
2. https://dashboard.render.com → **New → Blueprint** → select
   `travel-agent` → the included `render.yaml` configures everything.
3. In the dashboard, fill in the env vars it prompts for:
   `NVIDIA_API_KEY`, `AMADEUS_CLIENT_ID`, `AMADEUS_CLIENT_SECRET`.
   (Keys live in the dashboard, never in git.)
4. Deploy → your app gets a `https://travel-agent-xxxx.onrender.com` URL
   serving both the UI and the API from one origin.

**Anywhere with Docker (Railway, Fly.io, Hugging Face Spaces, a VPS):**
the included `Dockerfile` builds and runs the app; it respects `$PORT`.

Notes for cloud deploys:
- The free Render tier spins the service down after inactivity — the first
  request after a pause takes ~30s to cold-start.
- The in-conversation session and `agent_memory.json` live on the
  instance's ephemeral disk: they reset when the service restarts or
  redeploys. Long-term memory persistence across deploys would need an
  external store (a database) — fine for a demo, worth knowing about.
- A thinking model like GLM on NVIDIA NIM takes 30–90s per turn — the UI's
  "thinking… Ns" counter keeps you informed.

## Troubleshooting

- **Blank page / 404 at localhost:8000** — make sure `frontend/index.html`
  exists at that exact path relative to `server.py`; the backend serves it
  via a static file mount.
- **Setup-needed banner** — the page tells you exactly which keys are
  missing; edit `.env` and restart `uvicorn`.
- **"No LLM provider configured"** — add `NVIDIA_API_KEY` (or
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) to `.env` and restart.
- **"...model was not found"** — set `NVIDIA_MODEL` (or
  `ANTHROPIC_MODEL` / `OPENAI_MODEL`) to a valid model ID. For NVIDIA, the
  catalog is at https://build.nvidia.com.
- **Long waits / empty responses** — the GLM models on NIM are *thinking*
  models: expect 30–90s per turn. If a turn comes back empty, it burned its
  token budget thinking — just retry.
- **Amadeus 401** — double-check you copied the *test* environment
  credentials, not "production" (production needs separate approval).
- **CORS errors if you open `frontend/index.html` directly as a file** —
  don't; always go through `http://127.0.0.1:8000` so frontend and backend
  share an origin.
- **Port already in use** — run `uvicorn server:app --reload --port 8001`
  and open that port instead.

### Honest limitation

No public API lets a script actually purchase a flight or hotel room —
that needs a commercial booking/payment integration individual developers
can't get access to. "Approve booking" here finalizes the plan, logs it,
and generates a real calendar file — it doesn't charge a card. That's the
furthest any honest implementation can go without a business partnership.
