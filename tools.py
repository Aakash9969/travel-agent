"""
Real tools the agent can call.

- Flights & hotels: Amadeus for Developers self-service API (free tier).
  https://developers.amadeus.com  -> sign up -> create an app -> you get a
  Client ID and Client Secret for the TEST environment. Test-environment
  data is drawn from real caches of fare/hotel inventory (not hand-mocked
  by us), which is why this counts as "real" rather than a local mock --
  but note Amadeus's test environment is not the live production feed, and
  no purchase/payment endpoint is exposed on the free tier.

- Weather: Open-Meteo (https://open-meteo.com) - free, no API key, live data.

- Calendar action: writes a real RFC 5545-compliant .ics file to disk that
  you can import into Google Calendar / Outlook / Apple Calendar. This is
  the one part of the pipeline that performs an actual, verifiable action
  on your machine.

All functions raise a clear RuntimeError with guidance if required
environment variables are missing or an upstream API fails, rather than
silently falling back to fake data.
"""

import os
import time
import uuid
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

AMADEUS_BASE = "https://test.api.amadeus.com"

_token_cache = {"access_token": None, "expires_at": 0}

# WMO weather interpretation codes used by Open-Meteo.
# https://open-meteo.com/en/docs  (see "WMO Weather interpretation codes")
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Set it in your .env file (see README.md)."
        )
    return val


def _amadeus_token() -> str:
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    client_id = _require_env("AMADEUS_CLIENT_ID")
    client_secret = _require_env("AMADEUS_CLIENT_SECRET")

    resp = requests.post(
        f"{AMADEUS_BASE}/v1/security/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        hint = ""
        if resp.status_code == 401:
            hint = (" Double-check you copied the Client ID/Secret from the free "
                    "TEST environment (not production, which needs separate approval).")
        raise RuntimeError(
            f"Amadeus authentication failed ({resp.status_code}): "
            f"{resp.text[:200]}{hint}"
        )
    payload = resp.json()
    _token_cache["access_token"] = payload["access_token"]
    _token_cache["expires_at"] = time.time() + payload.get("expires_in", 1800)
    return _token_cache["access_token"]


def _amadeus_get(path: str, params: Dict) -> Dict:
    token = _amadeus_token()
    resp = requests.get(
        f"{AMADEUS_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Amadeus API error ({resp.status_code}) on {path}: {resp.text[:300]}"
        )
    return resp.json()


def resolve_city_or_airport_code(keyword: str) -> Optional[str]:
    """Look up a real IATA city/airport code for a free-text place name."""
    data = _amadeus_get(
        "/v1/reference-data/locations",
        {"keyword": keyword, "subType": "CITY,AIRPORT", "page[limit]": 1},
    )
    results = data.get("data", [])
    if not results:
        return None
    return results[0]["iataCode"]


def search_flights(origin: str, destination: str, departure_date: str,
                   adults: int = 1, max_results: int = 5) -> List[Dict]:
    """
    origin/destination: IATA codes (e.g. 'BLR', 'GOI') OR free-text city
    names (they'll be resolved via Amadeus location search first).
    departure_date: 'YYYY-MM-DD'
    """
    if len(origin) != 3 or not origin.isalpha():
        code = resolve_city_or_airport_code(origin)
        if not code:
            raise RuntimeError(f"Could not resolve an airport code for '{origin}'.")
        origin = code
    if len(destination) != 3 or not destination.isalpha():
        code = resolve_city_or_airport_code(destination)
        if not code:
            raise RuntimeError(f"Could not resolve an airport code for '{destination}'.")
        destination = code

    data = _amadeus_get(
        "/v2/shopping/flight-offers",
        {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
            "currencyCode": "USD",
        },
    )

    offers = []
    for offer in data.get("data", []):
        itin = offer.get("itineraries", [{}])[0]
        segments = itin.get("segments", [])
        if not segments:
            continue
        first_seg = segments[0]
        offers.append({
            "id": offer["id"],
            "airline": first_seg["carrierCode"],
            "origin": first_seg["departure"]["iataCode"],
            "destination": segments[-1]["arrival"]["iataCode"],
            "departure_time": first_seg["departure"]["at"],
            "duration": itin.get("duration"),
            "stops": len(segments) - 1,
            "price": offer.get("price", {}).get("total"),
            "currency": offer.get("price", {}).get("currency"),
        })
    return offers


def search_hotels(city_code: str, check_in: str, check_out: str,
                  max_results: int = 5) -> List[Dict]:
    """city_code: IATA city code, e.g. 'GOI' for Goa, 'PAR' for Paris."""
    if len(city_code) != 3 or not city_code.isalpha():
        resolved = resolve_city_or_airport_code(city_code)
        if not resolved:
            raise RuntimeError(f"Could not resolve a city code for '{city_code}'.")
        city_code = resolved

    hotel_list = _amadeus_get(
        "/v1/reference-data/locations/hotels/by-city",
        {"cityCode": city_code.upper()},
    )
    hotel_ids = [h["hotelId"] for h in hotel_list.get("data", []) if h.get("hotelId")][:20]
    if not hotel_ids:
        return []

    offers_data = _amadeus_get(
        "/v3/shopping/hotel-offers",
        {
            "hotelIds": ",".join(hotel_ids),
            "checkInDate": check_in,
            "checkOutDate": check_out,
            "adults": 1,
        },
    )

    hotels = []
    for entry in offers_data.get("data", []):
        if len(hotels) >= max_results:
            break
        hotel = entry.get("hotel", {})
        offers = entry.get("offers") or []
        if not offers:  # some hotels return no availability for the dates
            continue
        offer = offers[0]
        price = offer.get("price", {})
        hotels.append({
            "name": hotel.get("name"),
            "hotel_id": hotel.get("hotelId") or entry.get("id"),
            "rating": hotel.get("rating"),
            "price": price.get("total") or price.get("base"),
            "currency": price.get("currency"),
            "check_in": check_in,
            "check_out": check_out,
        })
    return hotels


def check_weather(city: str, date: str) -> Dict:
    """Real forecast via Open-Meteo. Free, no API key required."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=15,
    )
    geo.raise_for_status()
    results = geo.json().get("results")
    if not results:
        raise RuntimeError(f"Could not geocode city '{city}' for weather lookup.")
    lat, lon = results[0]["latitude"], results[0]["longitude"]

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": "auto",
        },
        timeout=15,
    )
    forecast.raise_for_status()

    daily = forecast.json().get("daily", {})
    dates = daily.get("time", [])
    if date in dates:
        idx = dates.index(date)
        code = daily["weathercode"][idx]
        return {
            "city": city,
            "date": date,
            "temp_max_c": daily["temperature_2m_max"][idx],
            "temp_min_c": daily["temperature_2m_min"][idx],
            "condition": WMO_CODES.get(code, f"weather code {code}"),
            "note": "live forecast",
        }
    # Open-Meteo's free forecast only covers ~16 days ahead; be honest about that.
    return {
        "city": city,
        "date": date,
        "note": "Requested date is outside Open-Meteo's ~16-day forecast window; "
                "no live data available for it.",
    }


# ---- Calendar (.ics) output -------------------------------------------------

def _ics_escape(text: str) -> str:
    """Escape per RFC 5545 §3.3.11: backslash, semicolon, comma, newlines."""
    if not text:
        return ""
    return (str(text)
            .replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\r\n", "\\n")
            .replace("\n", "\\n")
            .replace("\r", "\\n"))


def _ics_fold(line: str) -> List[str]:
    """RFC 5545 §3.1: content lines SHOULD be <= 75 octets. Fold longer
    lines by inserting CRLF + a single leading space on continuation
    lines, never splitting a UTF-8 sequence."""
    raw = line.encode("utf-8")
    parts = []
    limit = 73  # + 1 leading space on continuations stays within 75 octets
    while len(raw) > limit:
        cut = limit
        while cut > 0 and (raw[cut] & 0xC0) == 0x80:  # don't split a UTF-8 char
            cut -= 1
        parts.append(raw[:cut].decode("utf-8"))
        raw = raw[cut:]
    parts.append(raw.decode("utf-8"))
    return [parts[0]] + [" " + p for p in parts[1:]]


def create_calendar_event(title: str, start_iso: str, end_iso: str,
                          description: str = "", location: str = "",
                          out_dir: Optional[str] = None) -> str:
    """
    Writes a REAL .ics file to disk. This is the concrete, verifiable
    action the agent performs after human approval - you can double-click
    the resulting file to import it into Calendar/Outlook/Google Calendar.
    Returns the file path.
    """
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "itineraries")
    os.makedirs(out_dir, exist_ok=True)
    uid = str(uuid.uuid4())
    filename = os.path.join(out_dir, f"trip_{uid[:8]}.ics")

    def fmt(dt_iso: str) -> str:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Floating local time: the calendar interprets it in the user's zone.
            return dt.strftime("%Y%m%dT%H%M%S")
        return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Mini Autonomous Travel Agent//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}@mini-travel-agent",
        f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{fmt(start_iso)}",
        f"DTEND:{fmt(end_iso)}",
        f"SUMMARY:{_ics_escape(title)}",
        f"DESCRIPTION:{_ics_escape(description)}",
        f"LOCATION:{_ics_escape(location)}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]

    # RFC 5545 requires CRLF line endings; newline="" stops Python from
    # translating them again on Windows.
    folded: List[str] = []
    for line in lines:
        folded.extend(_ics_fold(line))
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(folded) + "\r\n")
    return filename
