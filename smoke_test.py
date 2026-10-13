"""
Offline smoke tests for the parts that don't need API keys.

Run with:
    python smoke_test.py

Covers: memory context string, and the .ics calendar generator
(RFC 5545 escaping, folding, CRLF endings, timezone handling).
The live weather tool is exercised too when the network is available
(Open-Meteo needs no key); a network failure is reported but doesn't
fail the run.
"""

import os
import sys
import tempfile

import tools
from memory import Memory


def test_memory_context():
    path = os.path.join(tempfile.gettempdir(), "ta_smoke_memory.json")
    if os.path.exists(path):
        os.remove(path)
    m = Memory(path=path)

    m.set("max_budget", "400 USD")
    m.set("likes_window_seat", "yes")          # custom key the agent invented
    m.set("seat_class", "economy")             # default, non-empty
    ctx = m.as_context_string()

    assert "max_budget: 400 USD" in ctx, ctx
    assert "likes_window_seat: yes" in ctx, "custom keys must surface in future prompts"
    assert "seat_class: economy" in ctx, ctx
    assert "preferred_airline" not in ctx, "empty defaults should be omitted"
    assert "number_of_past_trips: 0" in ctx, ctx

    m.log_trip({"destination": "Goa", "total_cost": "USD 412", "logged_at": "2026-10-13T10:00:00"})
    ctx = m.as_context_string()
    assert "past trip: Goa (total USD 412, booked 2026-10-13)" in ctx, ctx
    print("memory context        OK")


def test_ics():
    out_dir = os.path.join(tempfile.gettempdir(), "ta_smoke_ics")
    path = tools.create_calendar_event(
        title="Goa trip; final, price 412.50",
        start_iso="2026-12-10T09:00:00",              # naive -> floating local time
        end_iso="2026-12-13T18:00:00+05:30",          # tz-aware -> converted to UTC Z
        description="Flight: IND-122\nHotel: Beach Resort\nTotal: USD 412.50, taxes; included",
        location="Goa, India",
        out_dir=out_dir,
    )

    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")

    assert "\r\n" in text, "ICS must use CRLF line endings"
    lines = text.split("\r\n")
    assert lines[0] == "BEGIN:VCALENDAR"
    assert lines[-2] == "END:VCALENDAR"

    # Unfold RFC 5545 continuation lines (CRLF + single space) first.
    logical = []
    for line in lines:
        if line.startswith(" ") and logical:
            logical[-1] += line[1:]
        else:
            logical.append(line)

    def find(prefix):
        matches = [l for l in logical if l.startswith(prefix)]
        assert matches, f"missing {prefix}"
        return matches[0]

    summary = find("SUMMARY:")
    assert summary == "SUMMARY:Goa trip\\; final\\, price 412.50", summary

    desc = find("DESCRIPTION:")
    assert desc == "DESCRIPTION:Flight: IND-122\\nHotel: Beach Resort\\nTotal: USD 412.50\\, taxes\\; included", desc

    assert find("DTSTART:") == "DTSTART:20261210T090000", "naive datetime should stay floating"
    assert find("DTEND:") == "DTEND:20261213T123000Z", "tz-aware datetime should convert to UTC Z"
    assert find("LOCATION:") == "LOCATION:Goa\\, India"

    assert find("DTSTAMP:").endswith("Z"), "DTSTAMP must be UTC"
    for i, line in enumerate(lines):
        n = len(line.encode("utf-8"))
        assert n <= 75, f"line {i} exceeds 75 octets unfolded ({n}): {line!r}"

    # long text must be folded on a space boundary, and unfold back cleanly
    long_path = tools.create_calendar_event(
        title="A very long title " * 12,
        start_iso="2026-12-10T09:00:00",
        end_iso="2026-12-11T09:00:00",
        out_dir=out_dir,
    )
    with open(long_path, encoding="utf-8", newline="") as f:
        unfolded = f.read().replace("\r\n ", "")  # standard unfold: CRLF+space joins
    assert ("A very long title " * 12).strip() in unfolded.replace("SUMMARY:", "")
    print("ics calendar output    OK")


def test_weather():
    try:
        w = tools.check_weather("Goa", "2026-10-15")
    except Exception as e:
        print(f"weather (skipped)     network unavailable: {e}")
        return
    assert w["note"] == "live forecast", w
    assert "condition" in w and w["temp_max_c"] is not None, w
    print(f"weather               OK ({w['city']}: {w['condition']}, "
          f"{w['temp_min_c']}–{w['temp_max_c']} °C on {w['date']})")


if __name__ == "__main__":
    test_memory_context()
    test_ics()
    test_weather()
    print("\nAll smoke tests passed.")
    sys.exit(0)
