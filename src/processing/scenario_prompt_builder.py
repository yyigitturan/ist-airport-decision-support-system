import pandas as pd


def flt_plan_prompt(flight_df: pd.Series):

    def safe(x):
        if pd.isna(x):
            return "couldnt be clearly identified"
        return x

    prompt = f"""Current time: {safe(flight_df["date"])} Expected airspace entry time for flight 
     {safe(flight_df["callsign_code_icao"])} was {safe(flight_df["actual_entry_time"])}. This {safe(flight_df["haul"])}
     flight operated by {safe(flight_df["airline_name_english"])} was scheduled to arrive at {safe(flight_df["arr_sched_time_utc"])}
     UTC on {safe(flight_df["date"])} ({safe(flight_df["day_of_week"])}). It originated from {safe(flight_df["dep_name_english"])}
     ({safe(flight_df["dep_code_icao"])} / {safe(flight_df["dep_code_iata"])}, lat: {safe(flight_df["dep_lat"])}, lon: {safe(flight_df["dep_lon"])}, alt: {safe(flight_df["dep_altitude"])} ft) 
     and was headed for Istanbul Airport ({safe(flight_df["dest_code_icao"])} / {safe(flight_df["dest_code_iata"])}, lat: {safe(flight_df["dest_lat"])}, 
     lon: {safe(flight_df["dest_lon"])}, alt: {safe(flight_df["dest_altitude"])} ft). 
     The aircraft type : {safe(flight_df["aircraft_type"])}, 
     registration number : {safe(flight_df["aircraft_registration"])}, 
     and wake turbalence category : {safe(flight_df["wake_turbulence_cat"])}. 
     Total route distance : {safe(flight_df["distance"])} km.
    """

    return prompt

def weather_prompt(row: pd.Series):
    return row["weather_prompt"]