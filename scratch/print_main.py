with open("app/main.py", "rb") as f:
    lines = f.readlines()
for i in range(48, 62):
    print(i+1, repr(lines[i]))

