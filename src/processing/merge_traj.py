import pandas as pd
import glob
from pathlib import Path

# Ayarlar
SILVER_FOLDER = "data/silver/trajectories/"
OUTPUT_FILE = "data/silver/trajectories/trajectory_all.parquet"

def merge_raw_silver():
    # 1. Tüm Parquet dosyalarını listele
    files = glob.glob(f"{SILVER_FOLDER}/*.parquet")
    print(f"🚀 {len(files)} dosya birleştirme için bulundu.")

    # 2. Chunk bazlı okuma ve listeye ekleme
    # Bellek yönetimi için sütünları ve tipleri kontrol ederek okuyoruz
    all_data = []

    for i, file in enumerate(files):
        # Dosyayı olduğu gibi oku
        df_chunk = pd.read_parquet(file)

        all_data.append(df_chunk)
        print(f"✅ {i+1}/{len(files)}: {Path(file).name} belleğe alındı. Satır: {len(df_chunk):,}")

    # 3. Tek seferde birleştir
    print("\n📦 Birleştirme (Concat) işlemi başlıyor...")
    master_df = pd.concat(all_data, ignore_index=True)

    print(f"🔥 Toplam satır sayısı: {len(master_df):,}")
    print(f"💾 {OUTPUT_FILE} adresine kaydediliyor...")
    master_df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')

    print("✨ İşlem başarıyla tamamlandı!")
    return master_df

master_trajectory = merge_raw_silver()