import gym
import random
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.externals import joblib

#max_steps in game_env
max_steps = 100000

# threshold score for selecting a game as a part of
# a training set.
intial_score_threshold = 100

# how many games should I play for
# getting my initial sample data.
initial_game_num = 50000

# number of games to play using the trained nn
trained_games = 1000

# make the training process verbose
make_verbose = False

def get_avg(input_list):
    """
    Function for getting the average value in a list
    """
    return sum(input_list)/len(input_list)

def get_inputs(initial_data_list):
    input_list = []
    output_list = []
    for i in initial_data_list:
        input_list = input_list + i[0]
        output_list = output_list + i[1]

    return input_list, output_list
def get_random_moves(game_env):
    """
    Function for getting random moves for a specific score
    """
    initial_data = []
    score_list = []

    for _ in range(initial_game_num):

        # reset the game env after every game is done.
        game_env.reset()

        # intialize the current score to 0
        score = 0

        # complete play history of a game
        prev_observation = []
        observation_list = []
        action_list = []

        # while we only iterate 10000 times, in realty the
        # game will finish much quicker because the agent will loose.
        for i in range(max_steps):

            # let supply a random action to the
            # game
            action = random.randrange(0,2)

            # the current state of the game
            observation, reward, done, info = game_env.step(action)

            # use the prev_state of the game
            # to give a action now.
            if len(prev_observation) > 0:
                observation_list.append(prev_observation)
                action_list.append(action)

            # action made, current state becomes previous state now
            prev_observation = observation

            # maintain the record of the score
            score = score + reward

            # incase we loose
            if done:
                break

        if score > intial_score_threshold:
            initial_data.append([observation_list, action_list])


    score_list.append(score)

    # the average score by our random agent
    print("Average Score during random play: {}".format(get_avg(score_list)))

    # return the intial training data
    return initial_data


def play_game(game_env, curr_nn):
    """
    Function to play the cartpole game using
    trained nn
    """
    final_score = []
    for K in range(trained_games):
        game_env.reset()
        score = 0
        prev_observation = []

        for i in range(max_steps):
            game_env.render()

            # if this the first time the game is running
            # then make the first move randomly, since we
            # don't have a previous state.
            if len(prev_observation) == 0:
                action = random.randrange(0,2)
            else:
            # for all the subsequent moves, since there
            # are observations for the environment, predict
            # using the neural network.
                currs = np.array(prev_observation)
                currs.reshape(-1,1)
                action = curr_nn.predict([currs])[0]


            observation, reward, done, info = game_env.step(action)
            prev_observation = observation
            score = score + reward

            # end game incase the agent loses.
            if done:
                break

        print("Game Number {} Score: {}".format(K,score))
        final_score.append(score)

    print("Average Score for Trained Agent: {}".format(get_avg(final_score)))

def train(input_list, output_list):
    """
    Function for intializing and training a neural network
    """
    clf = MLPClassifier(solver='adam', activation='relu', verbose=make_verbose, learning_rate='adaptive', hidden_layer_sizes=(500,))
    clf.fit(input_list, output_list)
    return clf


def run_simulation():
    """
    Function for running a complete simulation of the cart pole problem
    """
    # setup the environment
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_steps * 100
    env.reset()

    # initial dataset to train the neural net on
    initial_data = get_random_moves(env)

    input_list, output_list =  get_inputs(initial_data)

    # get the trained instance of the neural network back
    curr_nn = train(input_list, output_list)

    # play game using the trained curr_nn
    play_game(env, curr_nn)


run_simulation()
