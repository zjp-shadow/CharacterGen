# load json
import json
with open('vroid.json', 'r') as f:
    path_list = json.load(f)

new_list = []
import glob
rendered_list = glob.glob("E:/new_render/*")
rendered_list = [rendered.split("\\")[-1].split("_")[0] for rendered in rendered_list]
for path in path_list:
    id = path.split("/")[-1].split(".")[0]
    if id not in rendered_list:
        new_list.append(path)

path_list = new_list
num_files = 2
# split to files
for i in range(num_files):
    with open('vroid_' + str(i) + '.json', 'w') as f:
        json.dump(path_list[i::num_files], f, indent=4)