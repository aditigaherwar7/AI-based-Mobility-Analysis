from dataloader import PoiDataloader

loader = PoiDataloader(0, 101)
loader.read("data/checkins-gowalla.txt")

print("Total locations:", loader.locations())

for i in range(5):
    print(i, "->", loader.get_coord(i))
