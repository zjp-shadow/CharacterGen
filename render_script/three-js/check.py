import glob, os
path_list = glob.glob("E:/new_render/*")
for path in path_list:
    new_path_list = glob.glob("E:/new_render/" + path.split('\\')[-1] + "/*")
    if len(new_path_list) != 240:
        print(path, len(new_path_list))
        # remove 
        os.system('rd /s /q "E:/new_render/' + path.split('\\')[-1] + '"')