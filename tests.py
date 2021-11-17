
instr = "Turn around and enter the in betweeen the house. Take a left. Enter the bedroom on the left. Wait in between the two beds.".lower().strip().strip(".")
all_directions = ["in between the","between the", "left"]

for direction in all_directions:
    if direction in instr:
        times_in = instr.count(direction)
        list_of_idx = []
        start = 0
        for _ in range(times_in):
            idx = instr.index(direction,start)
            list_of_idx.append(idx)
            start = idx +1
        print(direction, list_of_idx)