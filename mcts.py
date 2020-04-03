from math import sqrt, log2
from random import randrange, choice
from operator import attrgetter
from copy import copy

class MCTS:
    def __init__(self, game, exploration_constant):
        self.game = game
        self.exploration_constant = exploration_constant

    def tree_search(self, root, resources):
        for i in range(resources):
            # Traverse the tree to find a node that is not fully expanded
            leaf = self.traverse(root)

            # Perform a simulation/rollout on an unvisited leaf node
            simulation_result = self.rollout(leaf)

            # Backpropogate result from leaf node to the root node
            self.backpropogate(leaf, simulation_result)

        # Select the best move based on Q(s,a)
        return self.best_child(root)

    ### Traversing
    def traverse(self, node):
        """ 
            Traversing the tree using UCT until a not fully expanded node
            is encountered.
        """
        while node.is_fully_expanded():
            node = self.best_uct(node)

        # If the node has never been encountered, expand 
        if node.children == [] and self.non_terminal(node):
            self.expand_node(node)
            
        # Either pick an unvisited child, or the node is terminal 
        return self.pick_unvisited(node.children) or node

    def best_uct(self, node):
        """
            Returns the child node with best UCT.

            UCT is computed relative to the player who moves from the node.
            Player 1 aims to maximize UCT, while player 2 aims to minimize it.
        """
        for child in node.children:
            child.UCT(node, self.exploration_constant)
        if node.player_to_move.name == "1":
            uct = max(node.children, key=attrgetter("uct"))
        else:
            uct = min(node.children, key=attrgetter("uct"))
        return choice([n for n in node.children if n.uct == uct.uct])

    def expand_node(self, node):
        """ 
            Creates child nodes for every legal state that can be reached 
            from node 
        """
        for data in self.game.generate_possible_child_states(node):
            node.add_child(Node(data["state"], data["player"], node, data["result"]))

    def pick_unvisited(self, children):
        """
            Picks a random unvisited child to do rollout on.
        """
        if children == []:
            return False
        unvisited = [node for node in children if node.visits == 0]
        return choice(unvisited)

    ### Rollout 
    def rollout(self, node):
        """
            Simulation: a game is played out from node. Actions are chosen 
            based on the rollout policy. 

            Returns 1 if player 1 wins, 0 if player 2 wins
        """
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return self.result(node)
    
    def rollout_policy(self, node):
        """ 
            Pick random state and add it as a child of node
        """
        possible_states = self.game.generate_possible_child_states(node)
        data = choice(possible_states)
        return Node(data["state"], data["player"], node, data["result"])

    def non_terminal(self, node):
        return not self.game.env.is_terminal_state(node.get_state())

    def result(self, node):
        return 1 if node.parent.player_to_move.name == "1" else 0

    ### Backprop
    def backpropogate(self, node, result):
        """
            Backpropogate the result from the rollout from leaf node 
            to root node, updating the statistics of each node on the 
            path.
        """
        node.update_stats(result)
        if node.is_root():
            return
        self.backpropogate(node.parent, result)

    ### End
    def best_child(self, node):
        """ 
            Returns the child with the best Q(s,a) relative to the
            player moving from node.
        """
        if node.player_to_move.name == "1":
            cmp = max(node.children, key=attrgetter("q"))
        else:
            cmp = min(node.children, key=attrgetter("q"))
        return choice([n for n in node.children if n.q == cmp.q])


class Node:
    def __init__(self, state, player, parent, action):
        self.state = state
        self.player_to_move = player
        self.parent = parent
        self.p1_wins = 0
        self.visits = 0
        self.q = 0
        self.children = []
        self.uct = 0
        self.action = action

    def is_root(self):
        return self.parent == None

    def get_state(self):
        return copy(self.state)

    def update_stats(self, result):
        self.visits += 1
        if result == 1:
            self.p1_wins += 1
        self.q = self.p1_wins / self.visits

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        """ 
            Returns True if all of node's children have been visited 
        """
        for child in self.children:
            if child.visits == 0:
                return False
        return self.children != []

    def UCT(self, parent, c):
        """
            Core function of Monte Carlo Tree Search 

            Calculate Upper Confidence Bound relative to player that 
            moves from this node
        """
        if parent.player_to_move.name == "2":
            c = -c
        self.uct = self.q + c * sqrt(log2(parent.visits) / self.visits)
        return self.uct

    def reset(self):
        """
            Set current node to root of tree
        """
        self.parent = None
        self.p1_wins = 0
        self.visits = 0
        self.uct = 0
        self.children = []