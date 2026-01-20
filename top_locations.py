from collections import defaultdict

path = "data/checkins-gowalla.txt"

location_count = defaultdict(int)

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        loc = parts[1]
        location_count[loc] += 1

# Sort by visits
top10 = sorted(location_count.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 most visited locations:")
for i, (loc, count) in enumerate(top10, 1):
    print(f"{i}. Location {loc} -> {count} visits")
