import glob, os

path_list = glob.glob('vroid/*/*/*.vrm')
print(path_list.__len__())

# replace \\ to /
path_list = [path.replace('\\', '/') for path in path_list]

# remove exist
exist_list = glob.glob("H:/vrm-render/render_image/*")
# change to set
exist_list = [exist.split('\\')[-1].split('_')[0] for exist in exist_list]
exist_list = set(exist_list)

ans_list = []
for path in path_list[:]:
    if path.split('/')[-1].split('.')[0] in exist_list:
        ans_list.append(path)
print(ans_list.__len__())

import json
with open('vroid.json', 'w') as f:
    json.dump(ans_list, f, indent=4)

#print(path_list)