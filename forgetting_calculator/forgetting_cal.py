import numpy as np
import json

# Specify the file path
file_path = "log/lander_t10b05.log"

txt = ""
remove_key = ["total", "old", "new"]
list_dict = []
with open(file_path, 'r', encoding="utf8") as file:
    # Iterate through the file line by line
    line: str
    for line in file:
        if line.startswith("CNN: "):
            txt = txt + line.replace("CNN: ", "")
            data_dict = eval(line.replace("CNN: ", ""))
            for k in remove_key:
                data_dict.pop(k)
            list_dict.append(data_dict)

num_task = len(list_dict)
np_array = np.zeros((num_task, num_task))
list_key = list(list_dict[-1].keys())
for i in range(num_task):
    for j in range(i+1):
        np_array[i][j] = list_dict[i][list_key[j]]

forgetting = np.mean(np.max(np_array, axis=1) - np_array[-1])

# for i in range(num_task):
#     t = 0
#     for j in range(i+1):
#         t += np_array[i][j]
#     print(t/(i+1))
# print(np.mean(np_array[-1]))

print(forgetting)