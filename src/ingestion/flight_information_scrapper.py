"""
Istanbul Airport (LTFM) Arrival Flight Data Ingestion Script

"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import pytz
from itertools import cycle
from dotenv import load_dotenv
import os

load_dotenv()
# ===========================
# CONFIGURATION
# ===========================

# API Keys - for rotation
API_KEYS = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in API_KEYS.split(",") if k.strip()]

# API Configuration - UTC ENDPOINT
API_BASE_URL = "https://aerodatabox.p.rapidapi.com/flights/airports/icao/LTFM"
RATE_LIMIT_SEC = 1.2
MAX_RETRIES = 3
RETRY_DELAY = 5

# Time Slots
TIME_SLOTS = [
    ("00:00", "03:59"),
    ("04:00", "07:59"),
    ("08:00", "11:59"),
    ("12:00", "15:59"),
    ("16:00", "19:59"),
    ("20:00", "23:59")
]

# Output Directory
OUTPUT_DIR = "/Users/YGT/1 airline project/ist_flight_data"

# Timezone
ISTANBUL_TZ = pytz.timezone('Europe/Istanbul')
UTC_TZ = pytz.utc

# ===========================
# LOGGING SETUP
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IstanbulFlightIngestion:
    """
    Validator-compliant Istanbul Airport flight data ingestion.
    """

    def __init__(self, api_keys: List[str], output_dir: str):
        """Initialize ingestion with API key rotation."""
        self.api_keys = api_keys
        self.key_cycle = cycle(api_keys)
        self.current_key = next(self.key_cycle)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.last_request_time = 0
        self.request_count = 0
        self.key_rotation_count = 0

        logger.info(f"Initialized with {len(api_keys)} API keys")
        logger.info(f"Output directory: {self.output_dir}")

    def _rotate_api_key(self):
        """Rotate to next API key."""
        self.current_key = next(self.key_cycle)
        self.key_rotation_count += 1
        logger.debug(f"Rotated to API key #{self.key_rotation_count % len(self.api_keys) + 1}")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with current API key."""
        return {
            "X-RapidAPI-Key": self.current_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }

    def _enforce_rate_limit(self):
        """Enforce strict rate limiting - BAN PREVENTION."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_SEC:
            sleep_time = RATE_LIMIT_SEC - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _convert_istanbul_to_utc(self, ist_date: str, ist_time: str) -> str:
        """
        Convert Istanbul local time to UTC.
        
        Args:
            ist_date: Date in Istanbul timezone (YYYY-MM-DD)
            ist_time: Time in Istanbul timezone (HH:MM)
        
        Returns:
            UTC datetime string (YYYY-MM-DDTHH:MM)
        """
        try:
            # Create Istanbul datetime
            ist_datetime_str = f"{ist_date}T{ist_time}"
            ist_dt = ISTANBUL_TZ.localize(datetime.strptime(ist_datetime_str, '%Y-%m-%dT%H:%M'))

            # Convert to UTC
            utc_dt = ist_dt.astimezone(UTC_TZ)

            return utc_dt.strftime('%Y-%m-%dT%H:%M')

        except Exception as e:
            logger.error(f"Time conversion failed: {e}")
            return ""

    def _safe_get(self, data: Dict, path: str, default: Any = "") -> Any:
        """
        Safely get nested dict value with validator compliance.
        Returns empty string ("") for missing values - validator compatible.
        """
        keys = path.split('.')
        result = data

        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default

        # Validator expects empty string for missing values, not None
        return result if result is not None and result != "" else default

    def _get_time_field(self, flight: Dict, primary_path: str,
                        fallback_path: Optional[str] = None) -> str:
        """
        Get time field with optional fallback.
        
        For arr_actual_time_utc: Try actualTime, fallback to runwayTime.
        Returns empty string if both missing (validator compliant).
        """
        value = self._safe_get(flight, primary_path, default="")

        if not value and fallback_path:
            value = self._safe_get(flight, fallback_path, default="")

        return value

    def _convert_to_istanbul_time(self, utc_time_str: str) -> Tuple[str, str, int]:
        """
        Convert UTC to Istanbul time and extract date, weekday, hour.
        
        Returns: (date, day_of_week, hour_ist) or ("", "", "") if invalid
        """
        if not utc_time_str:
            return "", "", ""

        try:
            # Parse ISO 8601 format
            utc_dt = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))

            # Convert to Istanbul (UTC+3)
            ist_dt = utc_dt.astimezone(ISTANBUL_TZ)

            date = ist_dt.strftime('%Y-%m-%d')
            day_of_week = ist_dt.strftime('%A')
            hour_ist = ist_dt.hour

            return date, day_of_week, hour_ist

        except Exception as e:
            logger.warning(f"Time conversion failed for '{utc_time_str}': {e}")
            return "", "", ""

    def _extract_flight_data(self, flight: Dict) -> Dict[str, Any]:
        """
        Extract flight data with 100% validator compliance.
        
        CRITICAL FIX: Runway data extraction prioritizes arrival.runway first,
        then falls back to movement.runway if needed.
        
        Fills ALL required columns. Empty values = "" (not None).
        """
        try:

            # Scheduled arrival time (REQUIRED for validator)
            arr_sched_time_utc = self._safe_get(flight, 'arrival.scheduledTime.utc')

            # Istanbul local time from scheduled arrival
            date, day_of_week, hour_ist = self._convert_to_istanbul_time(arr_sched_time_utc)

            # CRITICAL: arr_actual_time_utc with fallback to runwayTime
            status = self._safe_get(flight, 'status')

            # Build validator-compliant data dictionary
            data = {
                # Derived fields (REQUIRED by validator)
                'date': date,
                'day_of_week': day_of_week,
                'hour_ist': hour_ist if hour_ist != "" else "",

                # Aircraft
                'hex_icao': self._safe_get(flight, 'aircraft.modeS'),

                # Airline (REQUIRED: airline_name_english, callsign_code_iata)
                'airline_name_english': self._safe_get(flight, 'airline.name'),
                'callsign_code_iata': self._safe_get(flight, 'number'),
                'callsign_code_icao': self._safe_get(flight, 'callSign'),
                'airline_iata': self._safe_get(flight, 'airline.iata'),
                'airline_icao': self._safe_get(flight, 'airline.icao'),

                # Departure (REQUIRED: dep_code_iata)
                'dep_code_iata': self._safe_get(flight, 'departure.airport.iata'),
                'dep_code_icao': self._safe_get(flight, 'departure.airport.icao'),
                'dep_name_english': self._safe_get(flight, 'departure.airport.name'),

                # Destination (REQUIRED: dest_code_iata, dest_code_icao - hardcoded)
                'dest_code_iata': 'IST',
                'dest_code_icao': 'LTFM',
                'dest_name_english': 'Istanbul Airport',
                'dest_lat': '41.2751',
                'dest_lon': '28.7519',
                'dest_altitude': '325', #feet

                # Arrival times (REQUIRED: arr_sched_time_utc)
                'arr_sched_time_utc': arr_sched_time_utc,
                'arr_revised_time_utc': self._safe_get(flight, 'arrival.revisedTime.utc'),
                'aircraft_type': self._safe_get(flight, 'aircraft.model'),
                'aircraft_registration': self._safe_get(flight, 'aircraft.reg'),


                # Status (REQUIRED by validator)
                'status': self._safe_get(flight, 'status'),
            }

            return data

        except Exception as e:
            logger.error(f"Flight extraction failed: {e}")

            return data

    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Make API request with retry logic and key rotation.
        """
        for attempt in range(MAX_RETRIES):
            try:
                self._enforce_rate_limit()

                headers = self._get_headers()

                logger.debug(f"Request attempt {attempt + 1}/{MAX_RETRIES}: {url}")
                response = requests.get(url, headers=headers, params=params, timeout=30)

                self.request_count += 1

                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit (429), rotating key and waiting...")
                    self._rotate_api_key()
                    time.sleep(60)
                    continue

                # Handle quota exceeded
                if response.status_code == 403:
                    logger.error(f"API quota exceeded (403), rotating key...")
                    self._rotate_api_key()
                    time.sleep(30)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")

                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    self._rotate_api_key()
                else:
                    logger.error(f"Max retries reached for {url}")
                    return None

        return None

    def fetch_slot(self, date: str, start_time: str, end_time: str) -> List[Dict]:
        """
        Fetch flights for a specific time slot with pagination.
        
        CRITICAL FIX: Converts Istanbul local time to UTC for API request.
        This ensures runway data is properly returned.
        
        Args:
            date: Date string (YYYY-MM-DD) in Istanbul timezone
            start_time: Slot start (HH:MM) in Istanbul timezone
            end_time: Slot end (HH:MM) in Istanbul timezone
        
        Returns:
            List of flight data dictionaries
        """
        # CRITICAL: Convert Istanbul time to UTC for API
        from_datetime_utc = self._convert_istanbul_to_utc(date, start_time)
        to_datetime_utc = self._convert_istanbul_to_utc(date, end_time)

        if not from_datetime_utc or not to_datetime_utc:
            logger.error(f"Time conversion failed for {date} {start_time}-{end_time}")
            return []

        all_flights = []
        cursor = None
        page = 1

        params = {
            'direction': 'Arrival',
            'withLeg': 'true',
            'withLocation': 'true',
            'withCancelled': 'false',
            'withCodeshared': 'false',
            'withPrivate': 'false'
        }

        logger.info(f"  Slot {start_time}-{end_time} IST (UTC: {from_datetime_utc.split('T')[1]}-{to_datetime_utc.split('T')[1]}): Fetching...")

        while True:
            # Use UTC times in URL
            url = f"{API_BASE_URL}/{from_datetime_utc}/{to_datetime_utc}"

            request_params = params.copy()
            if cursor:
                request_params['cursor'] = cursor

            data = self._make_request(url, request_params)

            if not data:
                logger.warning(f"  Slot {start_time}-{end_time}: Request failed, stopping")
                break

            arrivals = data.get('arrivals', [])

            if not arrivals:
                logger.debug(f"  Slot {start_time}-{end_time}: No more arrivals")
                break

            logger.debug(f"  Slot {start_time}-{end_time}: Page {page} - {len(arrivals)} flights")

            # Extract flight data
            for flight in arrivals:
                flight_data = self._extract_flight_data(flight)

                # Skip completely empty flights
                if flight_data.get('callsign_code_iata') or flight_data.get('arr_sched_time_utc'):
                    all_flights.append(flight_data)

            # Check pagination
            next_link = data.get('_links', {}).get('next', {}).get('href', '')
            if 'cursor=' in next_link:
                cursor = next_link.split('cursor=')[1]
                page += 1
            else:
                break

        logger.info(f"  Slot {start_time}-{end_time}: Collected {len(all_flights)} flights")
        return all_flights

    def fetch_day(self, date: str) -> List[Dict]:
        """
        Fetch all flights for a day using time slots.
        
        Args:
            date: Date string (YYYY-MM-DD) in Istanbul timezone
        
        Returns:
            List of all flights for the day
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DATE: {date} (Istanbul time)")
        logger.info(f"{'='*70}")

        all_flights = []

        for slot_start, slot_end in TIME_SLOTS:
            slot_flights = self.fetch_slot(date, slot_start, slot_end)
            all_flights.extend(slot_flights)

            # Small delay between slots
            time.sleep(2)

        logger.info(f"Total for {date}: {len(all_flights)} flights")
        return all_flights

    def save_to_csv(self, flights: List[Dict], date: str):
        """
        Save flights to CSV with validator compliance.
        """
        if not flights:
            logger.warning(f"No flights to save for {date}")
            return

        df = pd.DataFrame(flights)

        # Define column order
        columns = [
            'date', 'day_of_week', 'hour_ist',
            'hex_icao', 'aircraft_type', 'aircraft_registration',
            'airline_name_english', 'callsign_code_iata', 'callsign_code_icao',
            'airline_iata', 'airline_icao',
            'dep_code_iata', 'dep_code_icao', 'dep_name_english',
            'dest_code_iata', 'dest_code_icao', 'dest_name_english',
            'dest_lat', 'dest_lon', 'dest_altitude',
            'arr_sched_time_utc', 'arr_revised_time_utc',
            'status'
        ]

        # Reorder columns
        existing_cols = [col for col in columns if col in df.columns]
        df = df[existing_cols]

        # Remove duplicates
        if 'callsign_code_iata' in df.columns and 'arr_sched_time_utc' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=['callsign_code_iata', 'arr_sched_time_utc'], keep='first')
            after = len(df)
            if before > after:
                logger.info(f"Removed {before - after} duplicates")

        # Save
        filename = self.output_dir / f"arrivals_{date}.csv"
        df.to_csv(filename, index=False)

        total_count = len(df)

        logger.info(f"✅ Saved: {filename.name}")

        return total_count, 0

    def ingest_date_range(self, start_date: str, end_date: str):
        """
        Ingest flights for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD) in Istanbul timezone
            end_date: End date (YYYY-MM-DD) in Istanbul timezone
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        current = start
        total_flights = 0
        total_actual = 0
        days_processed = 0

        logger.info(f"\n{'#'*70}")
        logger.info(f"INGESTION START: {start_date} to {end_date} (Istanbul time)")
        logger.info(f"{'#'*70}")

        while current <= end:
            date_str = current.strftime('%Y-%m-%d')

            # Fetch day
            flights = self.fetch_day(date_str)

            # Save
            if flights:
                day_total, day_actual = self.save_to_csv(flights, date_str)
                total_flights += day_total
                total_actual += day_actual
                days_processed += 1

            # Move to next day
            current += timedelta(days=1)

            # Delay between days
            time.sleep(3)

        # Final summary
        logger.info(f"\n{'#'*70}")
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"{'#'*70}")
        logger.info(f"Days processed: {days_processed}")
        logger.info(f"Total flights: {total_flights}")
        logger.info(f"Flights with actual time: {total_actual}")
        logger.info(f"Total API requests: {self.request_count}")
        logger.info(f"Key rotations: {self.key_rotation_count}")
        logger.info(f"{'#'*70}\n")


def main():
    """Main execution - ingestion."""

    # Initialize ingestion
    ingestion = IstanbulFlightIngestion(
        api_keys=API_KEYS,
        output_dir=OUTPUT_DIR
    )

    try:

        ingestion.ingest_date_range(
            start_date="2025-11-2",
            end_date="2025-11-4"
        )

        logger.info("Ingestion completed successfully!")
        logger.info(f"Check: {OUTPUT_DIR}")

    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()