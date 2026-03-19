import pandas as pd
import random 

def safe(x):
    try:
        if pd.isna(x):
            return "could not be clearly identified"
    except Exception:
        pass
    return x

def flt_plan_prompt_v1(flight_df: pd.Series):
    prompt = (
        f"Current time: {safe(flight_df['obs_time'])}. "
        f"Actual airspace entry time for flight {safe(flight_df['callsign_code_icao'])} "
        f"was {safe(flight_df['actual_entry_time'])}. "
        f"This {safe(flight_df['haul'])} flight operated by {safe(flight_df['airline_name_english'])} "
        f"was scheduled to arrive at {safe(flight_df['arr_sched_time_utc'])} UTC on "
        f"{safe(flight_df['date'])} ({safe(flight_df['day_of_week'])}). "
        f"It originated from {safe(flight_df['dep_name_english'])} "
        f"({safe(flight_df['dep_code_icao'])} / {safe(flight_df['dep_code_iata'])}, "
        f"lat: {safe(flight_df['dep_lat'])}, lon: {safe(flight_df['dep_lon'])}, "
        f"alt: {safe(flight_df['dep_altitude'])} ft) "
        f"and was headed for Istanbul Airport "
        f"({safe(flight_df['dest_code_icao'])} / {safe(flight_df['dest_code_iata'])}, "
        f"lat: {safe(flight_df['dest_lat'])}, lon: {safe(flight_df['dest_lon'])}, "
        f"alt: {safe(flight_df['dest_altitude'])} ft). "
        f"The Aircraft type: {safe(flight_df['aircraft_type'])}. "
        f"Registration number: {safe(flight_df['aircraft_registration'])}. "
        f"Wake turbulence category: {safe(flight_df['wake_turbulence_cat'])}. "
        f"Total route distance: {safe(flight_df['distance'])} km."
    )
    return prompt

def flt_plan_prompt_v2(flight_df: pd.Series):
    prompt = (
        f"Scheduled for arrival at {safe(flight_df['arr_sched_time_utc'])} UTC on "
        f"{safe(flight_df['date'])} ({safe(flight_df['day_of_week'])}), "
        f"flight {safe(flight_df['callsign_code_iata'])}/"
        f"{safe(flight_df['callsign_code_icao'])} by {safe(flight_df['airline_name_english'])} "
        f"was set to land at {safe(flight_df['dest_name_english'])} "
        f"({safe(flight_df['dest_code_iata'])}/{safe(flight_df['dest_code_icao'])}, "
        f"lat: {safe(flight_df['dest_lat'])}, lon: {safe(flight_df['dest_lon'])}, "
        f"alt: {safe(flight_df['dest_altitude'])} ft). "
        f"The aircraft originated from {safe(flight_df['dep_name_english'])} "
        f"({safe(flight_df['dep_code_iata'])}/{safe(flight_df['dep_code_icao'])}, "
        f"lat: {safe(flight_df['dep_lat'])}, lon: {safe(flight_df['dep_lon'])}, "
        f"alt: {safe(flight_df['dep_altitude'])} ft), "
        f"and the total route spanned {safe(flight_df['distance'])} km. "
        f"The aircraft was a {safe(flight_df['aircraft_type'])} with registration "
        f"{safe(flight_df['aircraft_registration'])}, and it belonged to the "
        f"{safe(flight_df['wake_turbulence_cat'])} wake turbulence category. "
        f"It was expected to enter airspace or appear via ADS-B at "
        f"{safe(flight_df['actual_entry_time'])}. "
        f"This was a {safe(flight_df['haul'])}-haul flight."
    )
    return prompt

def weather_prompt(row: pd.Series):
    return row["weather_prompt"]

def build_prompt(row: pd.Series, use_weather: bool = True):
    base_prompt = random.choice([
        flt_plan_prompt_v1(row),
        flt_plan_prompt_v2(row),
    ])

    parts = [base_prompt]

    if use_weather:
        wp = row.get("weather_prompt", "")
        if isinstance(wp, str) and wp.strip():
            parts.append(wp.strip())

    return "\n\n".join(parts)


TRAJ_PROMPT_1 = "Airspace is described using three trajectory types. This embedding is for the focus trajectory: {"
TRAJ_PROMPT_2 = "} These embeddings are for other active trajectories: {"
TRAJ_PROMPT_3 = "} These embeddings are for past or inactive trajectories that may still matter: {"
TRAJ_PROMPT_4 = "} Missing types will have no embedding. Based on the above, predict the total time spent in the airspace."
