import os
import random
path_list = "/home/megh/projects/domain-adaptation/SSDA_MME/data/txt/multi_original/unlabeled_target_images_clipart_1.txt"
percent = 0.5
path_list = open(path_list, "r")
path_list = [image for image in path_list]
random.shuffle(path_list)
take = int(len(path_list) * percent)
leave = len(path_list) - take
print(take,leave)
take_list = path_list[0:take]
leave_list = path_list[take:]
print(len(take_list),len(leave_list))
take_file = open("take.txt","w")
leave_file = open("leave.txt","w")
for image in take_list:
    take_file.write(image)
for image in leave_list:
    leave_file.write(image)
