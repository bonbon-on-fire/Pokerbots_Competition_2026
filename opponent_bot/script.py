with open("python_skeleton/cards.txt", "w") as file:
    suits = ["h", "d", "c", "s"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    for s in suits:
        for r in ranks:
            file.write(f"{r}{s}\n")
