import getpass
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from garminconnect import Garmin, GarminConnectAuthenticationError

_GARMIN_TOKENSTORE = str(Path("~/.garminconnect").expanduser())


def get_garmin_client() -> Garmin:
    """
    Return an authenticated Garmin client using a two-stage strategy.

    Stage 1 — Token cache (preferred, no credentials needed):
        Loads OAuth tokens from ~/.garminconnect/. Valid ~30 days; the library
        refreshes them silently before each API call.

    Stage 2 — Fresh login (first run or tokens expired):
        Reads GARMIN_EMAIL from env. Password is taken from GARMIN_PASSWORD
        (headless/CI only) or prompted interactively via getpass() so it is
        never written to disk. Tokens are saved after a successful login so
        Stage 1 succeeds on all future runs.
    """
    os.makedirs(_GARMIN_TOKENSTORE, exist_ok=True)

    try:
        client = Garmin()
        client.login(_GARMIN_TOKENSTORE)
        print("[Garmin] Logged in via cached OAuth tokens.")
        return client
    except (GarminConnectAuthenticationError, Exception):
        pass

    email = os.environ.get("GARMIN_EMAIL")
    if not email:
        raise RuntimeError(
            "GARMIN_EMAIL is not set and no cached Garmin tokens were found.\n"
            "Add  GARMIN_EMAIL=your@email.com  to .env"
        )

    password = os.environ.get("GARMIN_PASSWORD") or getpass.getpass(
        f"Garmin password for {email} (input hidden): "
    )

    client = Garmin(
        email=email,
        password=password,
        prompt_mfa=lambda: input("Garmin MFA code: "),
    )
    client.login(_GARMIN_TOKENSTORE)
    print(f"[Garmin] Login successful. Tokens cached → {_GARMIN_TOKENSTORE}")
    return client


def fetch_health_summary() -> dict[str, Any]:
    """
    Fetch and return a health summary dict from Garmin Connect.

    Metrics collected:
      - body_battery              : most recent reading from today's/yesterday's series
      - sleep_score               : overall score from last night
      - hrv_status                : yesterday's HRV classification
      - resting_hr_bpm            : yesterday's resting heart rate
      - minutes_of_load_last7days : total training minutes across the last 7 days
                                    (derived from the 10 most recent activities)

    Falls back to clearly-labelled mock data so a network blip never kills
    the pipeline.
    """
    today = date.today()
    yesterday_str = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()

    try:
        client = get_garmin_client()

        # Body Battery — 2-day window so we catch today's data even if not yet synced
        bb_raw = client.get_body_battery(yesterday_str, today_str)
        body_battery = 0
        if bb_raw:
            series = bb_raw[-1].get("bodyBatteryValuesArray", [])
            if series:
                body_battery = int(series[-1][1])

        # Sleep score (last night)
        sleep_raw = client.get_sleep_data(yesterday_str) or {}
        sleep_score = (
            sleep_raw.get("dailySleepDTO", {})
                     .get("sleepScores", {})
                     .get("overall", {})
                     .get("value", 0)
        ) or 0

        # HRV status (yesterday)
        hrv_raw = client.get_hrv_data(yesterday_str) or {}
        hrv_status = (
            hrv_raw.get("hrvSummary", {})
                   .get("status", "unknown")
                   .lower()
        )

        # Resting heart rate (yesterday)
        hr_raw = client.get_heart_rates(yesterday_str) or {}
        resting_hr = hr_raw.get("restingHeartRate", 0) or 0

        # Training load — sum minutes from activities within the 7-day window
        activities = client.get_activities(0, 10) or []
        cutoff_str = (today - timedelta(days=7)).isoformat()
        recent_minutes = sum(
            a.get("duration", 0) / 60
            for a in activities
            if a.get("startTimeLocal", "")[:10] >= cutoff_str
        )

        summary: dict[str, Any] = {
            "body_battery":              body_battery,
            "sleep_score":               sleep_score,
            "hrv_status":                hrv_status,
            "resting_hr_bpm":            resting_hr,
            "minutes_of_load_last7days": round(recent_minutes, 1),
        }
        print(f"[Garmin] Live health summary: {summary}")
        return summary

    except Exception as exc:
        print(f"[Garmin] WARNING — fetch failed: {exc}. Using mock fallback.")
        return {
            "body_battery":              30,
            "sleep_score":               45,
            "hrv_status":                "poor",
            "resting_hr_bpm":            58,
            "minutes_of_load_last7days": 420.0,
            "_source":                   "mock_fallback",
        }
