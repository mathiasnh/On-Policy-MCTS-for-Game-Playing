from topp import TOPP

from math import sqrt


if __name__ == "__main__":
    
    EPISODES = 300
    SIZE = 5
    M = 5
    G = 6
    NN_LEANRING_RATE = 0.001
    NN_HIDDEN_LAYERS = [32, 64, 128, 64, 32]
    NN_ACTIVATION = "RELU HER"
    NN_OPTIMIZER = "ADAM HER"
    NN_LOSS_FUNCTION = "BCELoss HER"
    EPSILON = 0.9
    MC_EXPLORATION_CONSTANT = sqrt(2)
    MC_NUMBER_SEARCH_GAMES = 500


    topp = TOPP(M, G, EPISODES, SIZE, NN_HIDDEN_LAYERS)

    topp.tournament()