from visualization import HexMapVisualizer
from AI import NeuralNetActor
from hexgame import HexGame

class TOPP:
    def __init__(self, M, G, E, size, nn_layers):
        self.vs = [(a, b) for a in range(M) for b in range(M) if a < b]
        self.model_postfixes = [int(E/(M-1)) * i for i in range(M)]
        self.number_of_games = G
        self.size = size
        self.layers = nn_layers
        self.AI = {}

    def init_AI(self):
        for i, pf in enumerate(self.model_postfixes):
            ai = ToppPlayer(NeuralNetActor(self.size**2, self.layers, 0), pf, i)
            ai.load()
            self.AI[i] = ai

    def tournament(self):

        self.init_AI()
        
        for players in self.vs:
            print(f"Model {self.model_postfixes[players[0]]} vs. model {self.model_postfixes[players[1]]}")
            starter = 1
            game = HexGame(self.size, starter)
            visualizer = HexMapVisualizer(game.board.cells.values(), True, self.size, game_type="hex")

            p1_wins = 0
            
            for i in range(self.number_of_games):
                
                player = self.AI[players[game.playing - 1]]
                #print(f"Player {game.playing} (model {player.name}) is starting!")


                while not game.is_terminal_state():
                    #print(f"Player {player.name} moving")
                    visualizer.draw(game.get_state(), 0.0001)

                    state = game.get_simple_state()
                    legal_moves = game.get_reversed_binary()

                    possible_states = game.generate_possible_child_states()

                    pred = player.model.forward(state, legal_moves, dense=True)
                    best_index = pred.index(max(pred))

                    data = possible_states[best_index]

                    game.do_action(data["action"])

                    next_player = 0 if game.playing - 1 == 1 else 1
                    player = self.AI[players[next_player]]
                
                print(f"Player {starter} started, and player {game.playing} (model {player.name}) won!")
                
                p1 = self.model_postfixes[players[0]]
                if player.name == p1:
                    p1_wins += 1

                visualizer.draw(game.get_state(), 1)

                starter = 2 if starter == 1 else 1

                game.reset_map(starter, hard=True)
            
            print(f"Model {p1} won {p1_wins} out of {self.number_of_games} games ({p1_wins/self.number_of_games*100}%)")


class ToppPlayer:
    def __init__(self, model, name, index):
        self.model = model
        self.name = name
        self.index = index

    def load(self):
        print(self.name)
        self.model.load_model(f"models1/checkpoint{self.name}.pth.tar")

                    
