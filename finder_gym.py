import gym
from gym import spaces
import numpy as np
import pygame
from map_tester import *

fog_dist = 7

class nodeFindEnv(gym.Env):
    def __init__(self):
        super(nodeFindEnv, self).__init__()

        # self.world = np.array(world)  # world represented as a 2D numpy array
        # # self.start_pos = np.where(self.world == 'S')  # Starting position
        # # self.goal_pos = np.where(self.world == 'G')  # Goal position
        # self.start_pos = start_pos
        # self.current_pos = start_pos #starting position is current posiiton of agent

        self.num_rows, self.num_cols = 20, 20

        # row*col number of actions
        self.action_space = spaces.Discrete(self.num_rows*self.num_cols)  

        # Observation space is grid of size:rows x columns
        #self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        self.observation_space = spaces.Box(
            np.ones((1, 20, 20)) * -1,
            np.ones((1, 20, 20)),
            (1, 20, 20),
            dtype=np.float32,
        )

        prob = np.random.randint(15,45)*0.01
        size = 20

        world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        #for PRIMAL1 map
        # world = random_generator(SIZE_O=self.SIZE, PROB_O=self.PROB)
        world = padding_world(world)

        # for index, x in np.ndenumerate(world):
        #         if x == 0:
        #             result, self.world = ray_cast(world, index, 7, True)
        #             break

        self.density_square = [[0]*self.num_rows]*self.num_cols

        while True:
            x = np.random.randint(0, self.num_rows)
            y = np.random.randint(0, self.num_cols)

            if(world[x][y] == 0):
                result, self.world, visible_nodes = ray_cast(world, (x,y), fog_dist, True)
                break

        self.iteration = 0

        # Initialize Pygame
        pygame.init()
        self.cell_size = 30

        # setting display size
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
        # prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
        #                             self.PROB[1])  # sample a value from triangular distribution
        # size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
        #                         p=[.5, .25, .25])  # sample a value according to the given probability
        prob = np.random.rand()*0.3
        size = 20
        # prob = self.PROB
        # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        # here is the map without any agents nor goals
        world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        #for PRIMAL1 map
        # world = random_generator(SIZE_O=self.SIZE, PROB_O=self.PROB)
        world = padding_world(world)

        # for index, x in np.ndenumerate(world):
        #         if x == 0:
        #             result, self.world = ray_cast(world, index, 7, True)
        #             break

        self.density_square = [[0]*self.num_rows]*self.num_cols

        while True:
            x = np.random.randint(0, self.num_rows)
            y = np.random.randint(0, self.num_cols)

            if(world[x][y] == 0):
                result, self.world, visible_nodes = ray_cast(world, (x,y), fog_dist, True)
                break
            

        self.iteration = 0

        pygame.display.update()
        return np.expand_dims(self.world,0), {}

    def step(self, action):
        #choose new position based on the selected action
        new_pos = [0,0]

        # new_pos[0] = action % self.num_rows
        # new_pos[1] = action // self.num_rows

        new_pos[1] = action % self.num_rows
        new_pos[0] = action // self.num_rows

        green_count = 0
        empty_space = 0
        node_space = 0
        for index, x in np.ndenumerate(self.world):
            if x == 2:
                green_count -= 1

        #check if new node is in valid location
        if(self.world[new_pos[0]][new_pos[1]] == 2):
            result = 1
            _, self.world, visible_nodes_count = ray_cast(self.world,new_pos, fog_dist, False)
        elif(self.world[new_pos[0]][new_pos[1]] == -1):
            result = -1
        elif(self.world[new_pos[0]][new_pos[1]] == 1):
            result = -2
        elif(self.world[new_pos[0]][new_pos[1]] == 0):
            result = -3   

        # # Check if the new position is valid
        # if self._is_valid_position(new_pos):
        #     self.current_pos = new_pos
        
        #TODO:
        #increase penalty for invalid moves
        #increase reward for completed view
        #reduce fog distance
        #reward based on node connectivity

        # Reward function
        #node is placed on obstacle
        if result == -1:
            reward = -8.0
        #node placed on another node 
        elif result == -2:
            reward = -8.0
        #node is not connected
        elif result == -3:
            reward = -8.0
        #node placed successfully
        else:

            #reward based on increasing map coverage and node-space density ratio

            for index, x in np.ndenumerate(self.world):
                if x == 2:
                    #green_count starts at negative index (see above)
                    #only counts the new green nodes
                    green_count += 1
                
                if x == 2 or x == 0:
                    empty_space += 1
                elif x == 1:
                    node_space += 1

            #account for new node taking up 1 green space
            green_count += 1
            reward = green_count*0.5 - 7
            # density = node_space / (node_space + empty_space)

            # if reward > 0:
            #     reward = reward * (1-density)
            # else:
            #     reward = reward * density

            #discourage nodes from being placed next to each other
            x, y = new_pos
            node_proximity_penalty = 3
            nearby_node_count = 0
            if (x > 0 and self.world[x-1][y] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x-1][y]
            if (x < len(self.world)-1 and self.world[x+1][y] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x+1][y]
            if (y > 0 and self.world[x][y-1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x][y-1]
            if (y < len(self.world[0])-1 and self.world[x][y+1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x][y+1]
            if (x > 0 and y > 0 and self.world[x-1][y-1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x-1][y-1]
            if (x < len(self.world)-1 and y > 0 and self.world[x+1][y-1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x-1][y-1]
            if (x > 0 and y < len(self.world[0])-1 and self.world[x-1][y+1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x-1][y+1]
            if (x < len(self.world)-1 and y < len(self.world[0])-1 and self.world[x+1][y+1] == 1):
                reward -= node_proximity_penalty
                nearby_node_count += 1 + self.density_square[x+1][y+1]

            self.density_square[x][y] = nearby_node_count

            #encourage nodes from having line of sight with one another
            reward += (visible_nodes_count - nearby_node_count)*1.2
            
            
        #print("result = ", result)
        
        if 0 in self.world:
            done = False
        else:
            done = True
            reward += 150

        self.current_pos = new_pos

        self.iteration += 1
        if(self.iteration > 250):
            done = True


        #clip reward
        if reward < -2000:
            reward = -2000

        return np.expand_dims(self.world,0), reward, done,False, {'isValid': result}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.world[row, col] == -1:
            return False
        return True

    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        #print(self.world)

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                # try:
                #     print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                # except Exception as e:
                #     print('Initial state')

                if self.world[row, col] == -1:  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.world[row, col] == 1:  # node
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.world[row, col] == 2:  # visible range
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                # if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1,1)):  # Agent position
                #     pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display