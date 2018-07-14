import gym

#Create the environment
env = gym.make('CartPole-v0')


#Reset the env to the default
observation = env.reset()

for i in range(1000):

    #Visual aspect to see what happen to the env
    env.render()

    #Grab 4 variables of the observation
    cart_position, cart_velocity, pol_angle, ang_velocity = observation

    #Learning Agent(Policy) ... do some logic with 2 actions 0-left, 1-right
    if pol_angle > 0:
        action = 1
    else:
        action = 0

    #Feed the action to the env
    #Return back 4 variables: observation, reward, done and info
    observation, reward, done, info = env.step(action)
