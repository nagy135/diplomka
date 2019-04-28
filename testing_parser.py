import numpy as np
import re

cat = "/home/infiniter/Diplomka/data/input_sc/E10012B_2_R-0003_d.cat"
cat_mine = "/home/infiniter/Diplomka/data/input_sc/E10012B_2_R-0003_d_mine.cat"

cat_arr = list()
with open(cat, 'r') as c:
    for line in c:
        cutted = line[66:]
        cleaned = re.sub( ' +', ' ', cutted).strip()
        row = cleaned.split(' ')
        try:
            x,y = float(row[0]), float(row[1])
            cat_arr.append((x,y))
        except (IndexError, ValueError) as e:
            pass

cat_mine_arr = list()
with open(cat_mine, 'r') as c:
    for line in c:
        cleaned = re.sub( ' +', ' ', line).strip()
        row = cleaned.split(' ')
        try:
            x,y = float(row[0]), float(row[1])
            cat_mine_arr.append((x,y))
        except (IndexError, ValueError) as e:
            pass

dist_x_all = list()
dist_y_all = list()
for point in cat_arr:
    dist_x = 1024
    dist_y = 1024
    for point2 in cat_mine_arr:
        if abs(point[0] - point2[0]) < dist_x and abs(point[1] - point2[1]) < dist_y:
            dist_x = abs(point[0] - point2[0])
            dist_y = abs(point[1] - point2[1])
    if dist_x < 3 and dist_y < 3:
        dist_x_all.append(dist_x)
        dist_y_all.append(dist_y)
avg_x = round(np.mean(np.array(dist_x_all)), 4)
print('avg_x', avg_x)
avg_y = round(np.mean(np.array(dist_y_all)), 4)
print('avg_y', avg_y)
avg_total = round(np.mean(np.array([avg_x, avg_y])), 4)
print('avg_total', avg_total)
