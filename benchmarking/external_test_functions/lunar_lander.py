import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def objective(x):
    env = gym.make("LunarLander-v3")
    a = time.time()
    reward_sum = 0.
    num_seeds = 500 
    for i in range(num_seeds): # Average over 100 seeds as in the TurBO paper
        observation, info = env.reset(seed=i)

        episode_over = False
        while not episode_over:
            action = env.action_space.sample()
            action = heuristic_Controller(observation, x)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            episode_over = terminated or truncated

        env.close()
    b = time.time()
    elapsed_time = b-a
    print(f"Time for lunar lander call: {elapsed_time}")
    sys.stdout.flush()
    return -reward_sum/num_seeds

# Controller taken from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py
# But parameterized as in
def heuristic_Controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


if __name__ == "__main__":

    good_controller = np.array([0.5, 1.0, 0.4, 0.55, 0.5, 1.0, 0.5, 0.5, 0., 0.5, 0.05, 0.05])
    print(objective(good_controller))
    bad_controller = np.zeros(12,)
    print(objective(bad_controller))

    # Given good_controller
    good_controller = np.array([0.5, 1.0, 0.4, 0.55, 0.5, 1.0, 0.5, 0.5, 0., 0.5, 0.05, 0.05])

    # Step 1: Choose a random direction in parameter space
    random_direction = np.random.randn(len(good_controller))
    random_direction /= np.linalg.norm(random_direction)  # Normalize to unit length

    # Step 2: Define a range of values along this direction
    t_values = np.linspace(-1, 1, num=100)   # Adjust range if needed
    points = good_controller + t_values[:, np.newaxis] * random_direction

    # Clip points to stay within the unit cube [0,1]
    points = np.clip(points, 0, 1)

    # Step 3: Calculate the objective function for each point
    objectives = np.array([objective(point) for point in points])

    # Step 4: Plotting the results
    plt.figure(figsize=(10*2/3,6*2/3))
    plt.plot(t_values, objectives)
    plt.title('Objective Function Along Random Direction')
    plt.xlabel('Parameter Space Direction (t)')
    plt.ylabel('Objective Value')
    plt.grid()
    plt.show(block=False)
    plt.savefig("fig2.pdf", format="pdf", bbox_inches="tight", pad_inches=0.0)
    #plt.savefig('objective_function_plot__.pdf', format='pdf', dpi=300)