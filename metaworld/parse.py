import os
import copy

with open("envs/assets_v2/objects/assets/xyz_base.xml", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_line = copy.deepcopy(line)
    if 'rgba="' in line:
        # print(line)
        start = line.find('rgba="') + len('rgba="')
        end = line.find('"', start)
        r, g, b, a = line[start:end].split(" ")
        a = 0
        new_line = line[:start] + f"{r} {g} {b} {a}" + line[end:]
        print(line)
    if "'rgba='" in line:
        # print(line)
        start = line.find("rgba='") + len("rgba='")
        end = line.find("'", start)
        r, g, b, a = line[start:end].split(" ")
        a = 0
        new_line = line[:start] + f"{r} {g} {b} {a}" + line[end:]
    
        
    new_lines.append(new_line)
    
    
string = "".join(new_lines)
print(string)
with open("envs/assets_v2/objects/assets/xyz_base.xml", "w") as f:
    f.write(string)