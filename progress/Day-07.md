import matplotlib.pyplot as plt
from collections import defaultdict

path = "data/checkins-gowalla.txt"

location_count = defaultdict(int)

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        location_count[parts[1]] += 1

top10 = sorted(location_count.items(), key=lambda x: x[1], reverse=True)[:10]

locations = [x[0] for x in top10]
counts = [x[1] for x in top10]

plt.figure(figsize=(10,5))
plt.bar(locations, counts)
plt.title("Top 10 Most Visited Locations")
plt.xlabel("Location ID")
plt.ylabel("Number of Visits")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
