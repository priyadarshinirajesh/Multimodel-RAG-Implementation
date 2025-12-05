
# import pandas as pd, os
# df = pd.read_csv("data/raw/final_multimodal_dataset.csv")
# missing = [p for p in df['filename'].tolist() if not os.path.exists(p)]
# print("Missing files:", len(missing))
# # print up to first 20 missing paths
# for p in missing[:20]:
#     print(p)

import os

print(os.listdir("data/raw"))
