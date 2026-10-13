"""
Real, on-disk memory for the agent.
No mocking here: this reads/writes an actual JSON file in the project
folder, so preferences and trip history genuinely persist between runs.
"""

import copy
import json
import os
from datetime import datetime
from typing import Dict, List

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "agent_memory.json")

DEFAULTS = {
    "preferred_airline": None,
    "seat_class": "economy",
    "max_budget": None,          # in the currency the user quotes
    "hotel_stars_min": 3,
    "home_airport": None,
    "past_trips": [],
}


class Memory:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict:
        # Deep-copy the defaults: a shallow dict() copy would share the
        # past_trips LIST between every Memory instance in the process,
        # so trips logged by one would silently appear in another.
        defaults = copy.deepcopy(DEFAULTS)
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                try:
                    loaded = json.load(f)
                except json.JSONDecodeError:
                    loaded = {}
            if isinstance(loaded, dict):
                defaults.update(loaded)
        return defaults

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self.save()

    def log_trip(self, trip_summary: Dict):
        trip_summary["logged_at"] = datetime.now().isoformat(timespec="seconds")
        self.data.setdefault("past_trips", []).append(trip_summary)
        self.save()

    def as_context_string(self) -> str:
        """A compact summary handed to the LLM so it can use remembered prefs.

        Iterates over whatever is actually stored (including custom keys the
        agent saved via remember_preference), and briefly lists recent trips
        so the agent knows what the user has done before.
        """
        lines = []
        for key, value in self.data.items():
            if key == "past_trips":
                continue
            if value is None or value == "":
                continue
            lines.append(f"- {key}: {value}")
        past = self.data.get("past_trips") or []
        lines.append(f"- number_of_past_trips: {len(past)}")
        for trip in past[-3:]:
            dest = trip.get("destination", "?")
            cost = trip.get("total_cost") or "unknown cost"
            booked = (trip.get("logged_at") or "")[:10]
            lines.append(f"- past trip: {dest} (total {cost}, booked {booked})")
        return "\n".join(lines) if lines else "(nothing remembered yet - this is a fresh memory)"
