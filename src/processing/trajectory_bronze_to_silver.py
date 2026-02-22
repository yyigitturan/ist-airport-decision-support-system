import pandas as pd
from pathlib import Path

BRONZE_PATH = Path("/Users/YGT/ist-airport-decision-support-system/data/bronze/trajectories")
SILVER_PATH = Path("/Users/YGT/ist-airport-decision-support-system/data/silver/trajectories")
SILVER_PATH.mkdir(parents=True, exist_ok=True) # Klasör yoksa oluşturur

keep_list = [
    'hex', 'icao', 'date',
    'trajectory_timestamp', 'point_timestamp', 'timestamp_offset',
    'latitude', 'longitude', 'altitude_baro',
    'ground_speed', 'track', 'baro_rate', 'nav_altitude_mcp',
    'distance_km', 'on_ground'
]

def fix_trajectory_timestamps(df):
    # 1. Önce Trajectory Timestamp'i datetime yap (ms'den saniye bazlı UTC'ye)
    df['trajectory_timestamp'] = pd.to_datetime(
        df['trajectory_timestamp'].astype('int64') // 1000, 
        unit='s', utc=True
    ).dt.floor('s').astype('datetime64[s, UTC]')
    
    # 2. Point Timestamp'i hesapla (Trajectory + Offset) ve saniyeye sabitle
    df['point_timestamp'] = (
        df['trajectory_timestamp'] + 
        pd.to_timedelta(df['timestamp_offset'], unit='s')
    ).dt.floor('s').astype('datetime64[s, UTC]')
    
    # 3. Hesaplama bitti, offset'i atabiliriz
    return df.drop(columns=['timestamp_offset'])

for file in BRONZE_PATH.glob("*.parquet"):
    print(f"🚀 İşleniyor: {file.name}")
    
    df_raw = pd.read_parquet(file)
    
    # Sadece dosyada olan sütunları seç (KeyError almamak için)
    available_cols = [col for col in keep_list if col in df_raw.columns]
    temp_df = df_raw[available_cols].copy()
    
    # Zaman düzeltmesini uygula
    if 'trajectory_timestamp' in temp_df.columns and 'timestamp_offset' in temp_df.columns:
        temp_df = fix_trajectory_timestamps(temp_df)
    
    # Kaydet (df.save_parquet diye bir komut yoktur, to_parquet kullanılır)
    output_file = SILVER_PATH / file.name
    temp_df.to_parquet(output_file, index=False, compression='snappy')
    print(f"✅ Kaydedildi: {file.name} -> {temp_df.shape}")

print("\n✨ Tüm dosyalar TAM SANİYE formatında Silver'a taşındı!")