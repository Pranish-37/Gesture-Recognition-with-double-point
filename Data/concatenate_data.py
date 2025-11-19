import pandas as pd

# Backward gesture label
g1 = pd.read_csv("gesture_data_Backward.csv")
g1["label"] = 0
g1_1 = pd.read_csv("gesture_data_backward_2.0.csv")
g1_1["label"] = 0

# Forward gesture label
g2 = pd.read_csv("gesture_data_forward_2.0.csv")
g2["label"] = 1

g2_1 = pd.read_csv("gesture_data_Forward.csv")
g2_1["label"] = 1

# Handwave gesture label
g3 = pd.read_csv("gesture_data_wave_hand_2.0.csv")
g3["label"] = 2

g3_1 = pd.read_csv("gesture_data_wave hand.csv")
g3_1["label"] = 2

# Random gesture label
rnd = pd.read_csv("gesture_data_random.csv")
rnd["label"] = 3

full_df = pd.concat([g1, g1_1, g2, g2_1, g3, g3_1, rnd], ignore_index=True)
