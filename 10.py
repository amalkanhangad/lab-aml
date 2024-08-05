import numpy as np

grid_size = 4
actions = ['up', 'down', 'left', 'right']
rewards = np.zeros((grid_size, grid_size))
rewards[2, 2] = 1

def move(x, y, action):
    if action == 'up': return max(x-1, 0), y
    if action == 'down': return min(x+1, grid_size-1), y
    if action == 'left': return x, max(y-1, 0)
    if action == 'right': return x, min(y+1, grid_size-1)

def value_iteration(gamma=0.9, theta=1e-6):
    V = np.zeros((grid_size, grid_size)) #optimal value function
    policy = np.zeros((grid_size, grid_size), dtype=int)
    
    while True:
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) == (2, 2): continue
                
                v = V[x, y] #current value of state (x, y)
                q_values = [rewards[x, y] + gamma * V[*move(x, y, a)] for a in actions] #reward + discounted future value
                V[x, y] = max(q_values)
                policy[x, y] = np.argmax(q_values)
                delta = max(delta, abs(v - V[x, y]))
        
        if delta < theta:
            break
    
    return policy, V

def print_policy(policy):
    arrows = ['U', 'D', 'L', 'R']
    for x in range(grid_size):
        for y in range(grid_size):
            print('G' if (x, y) == (2, 2) else arrows[policy[x, y]], end=' ')
        print()

policy, _ = value_iteration()
print("Optimal Policy:")
print_policy(policy)