import os
os.system("python connect-four/self_play.py")

for cycle in range(50):
    print("=== Cycle", cycle, "===")
    os.system("python connect-four/self_play.py")
    os.system("python connect-four/train_model.py")
