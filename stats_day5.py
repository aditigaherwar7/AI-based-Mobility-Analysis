from collections import defaultdict

path = "data/checkins-gowalla.txt"

total = 0
users = set()
locs = set()

# Read line by line (fast + memory safe)
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        u = parts[0]
        l = parts[1]
        users.add(u)
        locs.add(l)
        total += 1

print("Total check-ins:", total)
print("Unique users:", len(users))
print("Unique locations:", len(locs))
