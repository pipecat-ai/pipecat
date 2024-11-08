import json

with open("logs.json", "r") as file:
    logs = json.load(file)

    # Find IDs with duration less than 245 seconds
    ids_with_short_duration = [
        [entry["id"], entry["mtgSessionId"], entry["duration"]]
        for entry in logs["data"]
        if entry["duration"] < 245
    ]

    print("\n=== Entries with Duration < 245 seconds ===")
    print(f"Total entries found: {len(ids_with_short_duration)} / 144")
    print("-" * 100)
    print(f"{'Recording ID':36} | {'Session ID':36} | {'Duration':10}")
    print("-" * 100)

    for rec_id, session_id, duration in ids_with_short_duration:
        print(f"{rec_id:20} | {session_id:20} | {duration:8} seconds")
    print("-" * 100 + "\n")
