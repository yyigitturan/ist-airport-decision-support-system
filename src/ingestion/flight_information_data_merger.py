import pandas as pd
import glob
import os


input_directory = "/Users/YGT/1 airline project/ist_flight_data"
output_file = "/Users/YGT/1 airline project/2025-11_IST_Final.csv"

def merge_raw_data():
    # (arrivals_*.csv)
    all_files = glob.glob(os.path.join(input_directory, "arrivals_*.csv"))

    if not all_files:
        print("error: there is no file")
        return

    print(f"📂 {len(all_files)} ...")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame = frame.sort_values(by=['date', 'arr_sched_time_utc'])

    frame.to_csv(output_file, index=False)

    print("-" * 50)
    print(f"✅ Process completed successfully! All files merged into one.")
    print(f"📊 Total raw count: {len(frame)}")
    print(f"💾 saved file: {output_file}")
    print("-" * 50)

if __name__ == "__main__":
    merge_raw_data()