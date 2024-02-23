# from ..map_generator import *
import numpy as np
import copy
import skimage.measure
import math
from scipy.signal import convolve2d


#trackable datatype
#world[][]
#num_visible_squares
#num_total_squares
#num_nodes
#average_degree_per_node

def find_hidden_squares(world):
    hidden_space = {}
    for x in len(world):
        for y in len(world[0]):
            if(world[x][y] != -1):
                hidden_space[(x,y)] = 1

    return hidden_space

def ray_cast(world, new_pos, fog_dist, isFirstNode):
    x, y = new_pos

    #if node placed on obstacle
    if(world[x][y] == -1):
        return -1, world, 0
    
    #node places on another node
    if(world[x][y] == 1):
        return -2, world, 0
    
    world_copy = world.copy()

    update_list = set()
    visible_nodes = set()
    another_Node_Found = False
    visible_nodes_count = 0

    # Cast rays in 360 degrees
    for angle in range(0, 360, 10):
        radian_angle = math.radians(angle)
        true_x, true_y = new_pos
        x = round(true_x)
        y = round(true_y)

        is_cornered_ray = False

        while ((0 <= x < len(world_copy)) and
               (0 <= y < len(world_copy[0])) and
                world_copy[x][y] != -1 and 
                is_cornered_ray == False and
               (math.sqrt((x-new_pos[0])*(x-new_pos[0]) + (y-new_pos[1])*(y-new_pos[1])) <= fog_dist)):
            
            last_x = x
            last_y = y

            true_x = true_x + math.cos(radian_angle)
            true_y = true_y + math.sin(radian_angle)
            x = round(true_x)
            y = round(true_y)

            if((0 <= x < len(world_copy)) and (0 <= y < len(world_copy[0]))):
                if(world_copy[x][last_y] == -1 and world_copy[last_x][y] == -1):
                    is_cornered_ray = True
                elif(world_copy[x][y] == 0):
                    update_list.add((x,y))
                elif(world_copy[x][y] == 1):
                    another_Node_Found = True
                    visible_nodes.add((x,y))
            
    if another_Node_Found or isFirstNode:
        world_copy[new_pos[0]][new_pos[1]] = 1  
        for attr in update_list:
            x, y = attr 
            world_copy[x][y] = 2
    #no line of sight with other nodes
    else:
        return -3, world, 0

    return 1, world_copy, len(visible_nodes)
            

def padding_world(world):
    labeled_image, count = skimage.measure.label(world, background=-1, connectivity=1, return_num=True)
    num_comp = np.zeros(count)
    for i in range(labeled_image.shape[0]):
        for j in range(labeled_image.shape[1]):
            for seg in range(1, count + 1):
                if labeled_image[i, j] == seg:
                    num_comp[seg - 1] = num_comp[seg - 1] + 1
    list_num_comp = num_comp.tolist()
    max_index = list_num_comp.index(max(list_num_comp))
    padding_world = copy.deepcopy(labeled_image)
    for seg in range(1, count + 1):
        for i in range(labeled_image.shape[0]):
            for j in range(labeled_image.shape[1]):
                if labeled_image[i, j] == 0:
                    padding_world[i, j] = -1
                elif labeled_image[i, j] == max_index + 1:
                    padding_world[i, j] = 0
                else:
                    padding_world[i, j] = -1
    return padding_world
    

                
            

        
