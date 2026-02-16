import glob
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    pattern = "/Users/YGT/1 airline project/processed/istanbul_2025-09-*_adsb.parquet"
    output_file = "2025-09-adsb.parquet"

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError("No parquet files found!")

    print(f"Found {len(files)} files\n")

    tables = []
    total_rows = 0

    for f in files:
        print(f"Reading: {f}")
        table = pq.read_table(f)
        tables.append(table)
        total_rows += table.num_rows

    print("\nConcatenating tables (no row modification)...")
    combined = pa.concat_tables(tables, promote=True)

    print("Writing merged file...")
    pq.write_table(combined, output_file)

    print("\n========== DONE ==========")
    print(f"Output file: {output_file}")
    print(f"Expected rows: {total_rows}")
    print(f"Written rows:  {combined.num_rows}")

    if total_rows == combined.num_rows:
        print("✔ ZERO DATA LOSS CONFIRMED")
    else:
        print("ROW COUNT MISMATCH")


if __name__ == "__main__":
    main()
