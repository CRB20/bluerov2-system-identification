#!/usr/bin/env python3

import csv
from pathlib import Path

INPUT = Path(
    "/home/hp/rov_system_id_ws/ROV_system_idetification/data/raw/"
    "rov_identification_20260925_181127.csv"
)

OUTPUT = Path(
    "/home/hp/rov_system_id_ws/ROV_system_idetification/data/"
    "rov_identification_20260925_181127_50hz.csv"
)

TARGET_DT = 0.02  # 50 Hz


def process_segment(rows, writer):
    """
    Convert one experiment segment to a 50 Hz time grid.

    The raw ros_time_s timestamps are used directly.
    For each 20 ms target time, the nearest recorded row is selected.
    No interpolation is performed.
    """

    if not rows:
        return 0

    rows.sort(key=lambda r: float(r["ros_time_s"]))

    t0 = float(rows[0]["ros_time_s"])
    t1 = float(rows[-1]["ros_time_s"])

    target_time = t0
    idx = 0
    written = 0

    while target_time <= t1:

        # Move forward until rows[idx] is the last sample
        # before or at the target time.
        while (
            idx + 1 < len(rows)
            and float(rows[idx + 1]["ros_time_s"]) <= target_time
        ):
            idx += 1

        best = idx

        # Compare with the next sample and choose the nearest timestamp.
        if idx + 1 < len(rows):
            t_current = float(rows[idx]["ros_time_s"])
            t_next = float(rows[idx + 1]["ros_time_s"])

            if abs(t_next - target_time) < abs(t_current - target_time):
                best = idx + 1

        writer.writerow(rows[best])
        written += 1

        target_time += TARGET_DT

    return written


def main():

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("50 Hz ROV IDENTIFICATION DATA EXTRACTION")
    print("=" * 80)
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print("Rate  : 50 Hz")
    print()

    input_rows = 0
    output_rows = 0
    segments = 0

    current_segment = None
    segment_rows = []

    with INPUT.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as fin, OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as fout:

        reader = csv.DictReader(fin)

        if reader.fieldnames is None:
            raise RuntimeError("Input CSV has no header.")

        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames
        )

        writer.writeheader()

        for row in reader:

            input_rows += 1
            segment = row["segment_id"]

            if current_segment is None:
                current_segment = segment

            # Finish previous experiment segment.
            if segment != current_segment:

                output_rows += process_segment(
                    segment_rows,
                    writer
                )

                segments += 1
                segment_rows = []
                current_segment = segment

            segment_rows.append(row)

        # Final segment.
        if segment_rows:
            output_rows += process_segment(
                segment_rows,
                writer
            )
            segments += 1

    print()
    print("Extraction complete")
    print("-------------------")
    print(f"Input rows      : {input_rows:,}")
    print(f"Output rows     : {output_rows:,}")
    print(f"Segments        : {segments:,}")

    if output_rows:
        print(
            f"Reduction       : "
            f"{input_rows / output_rows:.1f}x"
        )

    print()
    print(f"Saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
