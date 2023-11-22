# Script that implements AlphaZero's MCTS for a 11x11 Gomoku game

# Author: Azmyin Md. Kamal
# Date: 11/21/2023

"""
Notes:
* Network adopted from "AlphaZero Gomoku" report by Liang et al.
* MCTS tutorial: https://www.youtube.com/watch?v=NjeYgIbPMmg&ab_channel=SkowstertheGeek
* Tutorial on CNN https://www.youtube.com/watch?v=ZBfpkepdZlw&ab_channel=JamesBriggs
* https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/src/neural_network.py
* https://github.com/michaelnny/alpha_zero/blob/main/network.py
* Good tutorial on CNN: https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=532s&ab_channel=PatrickLoeber
* Policy Value architecture taken from https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
* Forward pass implementation idea https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
* Excellent introductory tutorial on tree data structure: https://www.youtube.com/watch?v=qH6yxkw0u78&ab_channel=mycodeschool

* standardState - (2,11,11) a 3D matrix containing the placement for the whole board for both players

* Understanding alphazero's MCTS
https://www.youtube.com/watch?v=HikhrP5sgQo&ab_channel=ThePrincipalComponent

"""


# Imports
import numpy as np
import math
import sys
from MCTSBase import TreeNode, MCTSBase

import torch # Requried to create tensors to store all numerical values
import torch.nn as nn # Required for weight and bias tensors
import torch.nn.functional as F # Required for the activation functions


# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpuct = 1.05 # A hyper parameter for UCB formulation
board_sz = 11 # Default


class PolicyValueNetworkWithNoTrain(nn.Module):
    """
    Class that applies the DNN network to compute policy logits and value for a given board state
    * Logits are internally scaled to 0 - 1 as we are not doing cross entropy loss
    """
    def __init__(self, num_players = 2, board_sz = 11):
        super(PolicyValueNetworkWithNoTrain, self).__init__()
        
        # Initialize class specific parameters
        self.num_players = num_players
        self.board_sz = board_sz
        self.num_positions = self.board_sz * self.board_sz # 1D array of how many unique moves possible
        
        #* Feature extraction with some convolution layers
        self.conv1 = nn.Conv2d(2,32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
       
        # Map features to a linear layer
        self.fc1 = nn.Linear(64*7*7, 256) # Why 64*7*7 had to be found by trial in advance.
        # Follow 18:00 from this video https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=532s&ab_channel=PatrickLoeber
        
        #* Policy head
        self.policy_fc = nn.Linear(256, self.num_positions) # From fc1 to another fully connected layer of 121 neurons
        self.softmax = nn.Softmax(dim = 1) # Scale logits between 0 - 1

        #* Value head
        self.value_fc = nn.Linear(256, 1) # From FC1 to scalar
        self.tanh = nn.Tanh()


    def forward(self, standardState):
        """Given the board state boardState, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        
        # Preprocess, convert standardState from numpy 3D tensor to pytorch tensor
        boardTensor = torch.from_numpy(standardState).to(device).float()
        
        s = F.relu(self.conv1(boardTensor))
        s = F.relu(self.conv2(s))
        #* Before passing to the Fully Connected layer, it needs to be flattened
        s = s.view(-1, 64*7*7) # -1 corresponds to number of batch size
        s = F.relu(self.fc1(s))

        pi = self.softmax(self.policy_fc(s)).detach().cpu() # Probability of choosing an action, pull out tensor from computation graph
        v = self.tanh(self.value_fc(s)).detach().cpu() # Value of the current state, pull out tensor from computation graph

        # Post process to return as numpy arrary and a scalar value
        pi = pi.numpy().flatten()
        v = v.item()
        return pi, v


class Node(TreeNode):
    """
    A node object in the Monte Carlo Tree
    * policyvaluednn is a pointer, its not a copy of the whole neural network
    * standardState = 2,11,11 board state
    * game_state = None, 0 for draw and 1 for a player win, used to determine if a node is terminal
    """
    #def __init__(self, unique_id, policyvalue_DNN, board_state, parent = None, action_taken = None, game_state = None):
    def __init__(self, policyvalue_DNN, standardState, game_state = None, parent = None):
        super(Node).__init__() # ensures TreeNode class is declared before MonteCarloTreeNode is declared
        
        # self.id = unique_id # This node's unique id
        self.polvaldnn = policyvalue_DNN # Keeps a pointer to the neural network
        
        # Node specific variables
        self.standardState = standardState.copy()
        self.game_state = game_state # Required
        self.move_tuple = None # (x,y) tuple, selected after the fact??
        self.children_nodes = [] # List of nodes that are children to this node
        self.parent = parent # Node that becomes its parent
        
        # # Redundant ??
        # self.leaf_flag = False # Set true if this node has no children
        # self.terminal_flag = False # Set True based on game condition
        
        # Monte Carlo Tree Search algorithm specific variables
        self.N = 0 # Int, number of times action "a" has been taken from state "s" to come to this node, visit count
        self.W = 0.0 # Float, "total value of next state" ?? Still not very clear
        self.Q = 0.0 # Float, W/N, same concept from Q value
        self.P = 0.0 # Float, prior value, 0 - 1, prior proability of the action "a" being selected that lead to this node

        #* Incremented in the self.update() function when this node is in the path of the current simulation
        #* Everytime a leaf node is found, we use the Policy Head to find the value of this state. 
        #* On self.update(), this self.W is added to all the nodes in the current path
        #* Update Q = W/N in self.update()

    def is_terminal(self):
        '''
        :return: True if this node is a terminal node, False otherwise.
        * We also update the terminal_flag in this function
        '''
        if (self.game_state is not None):
            return True
        else:
            return False

    def value(self):
        '''
        :return: the value of the node from the current player's point of view
        * No simulation, the value head directly reports the reward
        '''
        _,v = self.polvaldnn(self.standardState) #* Value head, returns numpy scalar
        return v # Returns 

    def find_action(self):
        '''
        Find the action with the highest upper confidence bound to take from the state represented by this MC tree node.
        :return: action as a tuple (x, y), i.e., putting down a piece at location x, y
        '''
        """
        * Useful notes on UCB https://www.youtube.com/watch?v=UXW2yZndl7U&ab_channel=JohnLevine
        """
        # Initialize work variables
        sum__Nsb = 0 # Sum of visits of action "b" of children nodes connected to this node
        ucb_scores = None # 1D numpy array
        
        P_sa,_ = self.polvaldnn(self.standardState) #* P(s,a) for given state s, find prior probabilities of all actions "a"

        #* P_sa here is for all actions (legal and illegal) we zero out probabilities for illegal moves
        P_sa_legal = np.copy(P_sa)
        P_sa_legal = P_sa_legal.reshape((11, 11)) # 2D array now

        #* Mask out illegal moves
        currentState = self.standardState.copy()
        player1Board = currentState[0,:,:]
        player2Board = currentState[1,:,:]

        #* Cycle through Player 1 moves
        for i in range(player1Board.shape[0]): # row
            for j in range(player1Board.shape[1]): # column
                if (player1Board[i,j]!=0):
                    P_sa_legal[i,j] = 0.0 # Apply mask
                else:
                    # Don't apply mask
                    pass
        
        #* Cycle through Player 2 moves
        for i in range(player2Board.shape[0]): # row
            for j in range(player2Board.shape[1]): # column
                if (player2Board[i,j]!=0):
                    P_sa_legal[i,j] = 0.0 # Apply mask
                else:
                    # Don't apply mask
                    pass
        
        P_sa_legal = P_sa_legal.flatten() # Revert back to 1D array

        # Cycle through the child nodes, recover their visit counts and add them up
        for chd in self.children_nodes:
            sum__Nsb = sum__Nsb + chd.N
        
        #* Upper confidence bound scores for the entire action space
        ucb_scores = self.Q + cpuct * P_sa_legal * (math.sqrt(sum__Nsb)/(1 + self.N)) # 1D numpy array of UCB scores

        #* Tree Policy: Select action that maximizes the upper confidence obund on the Q-values
        x,y = self.map_to_board(np.argmax(ucb_scores)) # Choose board position for the move that has the maximum UCB score

        move_tuple = (x,y)

        # Record this move into this node
        self.move_tuple = move_tuple

        return (x,y)

    def update(self, v):
        '''
        Update the statistics/counts and values for this node
        :param v: value backup following the action that was selected in the last call of "find_action"
        :return: None
        '''
        # Update this node first
        self.N = self.N + 1 # Increment count of number of visits
        self.W = self.W + v
        self.Q = self.W / self.N

        # # Then update its parent node
        # if (self.parent is not None):
        #     self.parent.N = self.parent.N + 1 # Increment count of number of visits
        #     self.parent.W = self.parent.W + v
        #     self.parent.Q = self.parent.W / self.parent.N
        
        return None


    #* Helper methods
    def map_to_board(self, one_d_pos):
        """
        Funtion that maps the 
        returns x,y row and col position on the board
        """
        row = one_d_pos // board_sz
        col = one_d_pos % board_sz

        # print(f"x,y move {row}, {col}") # Debug
        return (row, col)


class MCTS(MCTSBase):
    # Implements Monte Carlo Tree Search with DNNs
    # Note the game board will be represented by a numpy array of size [2, board_size[0], board_size[1]]
    def __init__(self, game):
        """
        game: Gomoku object
        state: 3d numpy array of size [n_player, board_height, board_width]
        """
        super().__init__(game) # MCTS now has access to self.game variable
        
        # Encoding 2D positions of the board into 1D numbers
        self.actionable_moves_1d_mapping = [move for move in range(board_sz * board_sz)] # 1D map of the 2D game board
        
        # Initialize a DNN Policy and Value network
        self.policyvalue_dnn = PolicyValueNetworkWithNoTrain()
        self.policyvalue_dnn.to(device) # Puts to GPU or CPU
        self.policyvalue_dnn = self.policyvalue_dnn.eval() # Evaluation mode
        
        # Algorithm related variables
        self.insert_root_node = True # Set false once the root node has been built
        
        # Variables to execute the full MCTS pipeline
        self.node_count = 0 # Keeps a count of how many nodes are present in the monte carlo tree
        self.root_node = None # Pointer to the node in Tree graph that will become the parent of the current child node, essentially the "root" node
        self.standardState = None # Numpy 3d array (2,11,11), a work variable that records the board state for the i^th MCTS iteration
        self.move_count_matrix = np.zeros((board_sz,board_sz)) # A 11x11 matrix where each cell counts how many times that move was selected. Set 0 for unavailable or illegal moves
        self.search_tree = [] # Chain of nodes used in this iteration of MCTS

        self.current_root = None # As we move down in the search tree we need to change focus 

    def reset(self):
        '''
        Clean up the internal states and make the class ready for a new tree search
        :return: None
        '''
        # Reset
        self.node_count = 0 # Keeps a count of how many nodes are present in the monte carlo tree
        self.root_node = None # Pointer to the node in Tree graph that will become the parent of the current child node, essentially the "root" node
        
        self.standardState = None 
        self.insert_root_node = True
        self.search_tree = [] # Chain of nodes used in this iteration of MCTS
        self.move_count_matrix = np.zeros((board_sz,board_sz))
        
        return None

    def get_treenode(self, standardState):
        '''
        Find and return the node corresponding to the standardState in the search tree
        :param standardState: board state
        :return: tree node (None if the state is new, i.e., we need to expand the tree by adding a node corresponding to the state)
        '''
        self.standardState = standardState.copy() # copy the latest state
        
        # Iteration 1, no root node, no nodes in the tree graph
        if (self.node_count == 0):
            return None # No nodes in the tree graph, we need to insert the root node, in iteration 1
        else:
            # Iteration 2 and onwwards
            found_node = None
            found_node = self.search_for_matching_node(self.standardState)
            # print(f"get_treenode -- found_node: {found_node}")
             
            return found_node # either None or the found_node
        
    def new_tree_node(self, standardState, game_end):
        '''
        Create a new tree node for the search
        :param standardState: board state
        :param game_end: whether game ends after last move, takes 3 values: None-> game not end; 0 -> game ends with a tie; 1-> player who made the last move win
        :return: a new tree node
        '''
        workState = standardState.copy() # Redundant?

        #* Iteration 1, no node in the tree graph, add root node
        if (self.insert_root_node):
            # root = Node(self.unique_node_id, self.policyvalue_dnn, self.board_state, parent=None, action_taken=None, game_state=None)
            root_n = Node(self.policyvalue_dnn, workState, game_end, parent = None)
            self.insert_root_node = False # Will not execute this again
            self.search_tree.append(root_n) # push root node into the current search tree
            self.node_count = self.node_count + 1
            self.current_root = root_n # Level L = 0
            return root_n

        else:
            # Child nodes created here from Iteration 2 and onwards
            # print(f"In new_tree_node, creating a new child node\n")
            chd_node = Node(self.policyvalue_dnn, workState, game_end, parent = self.current_root) 
            self.current_root.children_nodes.append(chd_node) # This node is now the child of the previous root level
            
            self.search_tree.append(chd_node) # push root node into the current search tree
            self.node_count = self.node_count + 1

            #* EXPERIMENTAL
            # This node is now the root for the next level 
            self.current_root = chd_node # Level L = 1 and onwards
        
            return chd_node   

    def get_visit_count(self, state):
        '''
        Obtain number of visits for each valid (state, a) pairs [i.e childrens] from this state during the search
        :param state: the state represented by this node
        :return: a board_size[0] X board_size[1] matrix of visit counts. 
        It should have zero at locations corresponding to invalid moves at this state.
        '''

        # # Psuedocode
        #* Find node corresponding to the given state. state represents the root node
        #* Recover the child nodes and their child nodes in a recursive manner
        
        out_mat = np.zeros((11,11))
        # print(f"Number of nodes in the search tree -- {len(self.search_tree)}")

        # Initialize work variables
        workState = state.copy()
        root_n = self.search_for_matching_node(workState) # Get the root [L0]
        #root_n = self.search_tree[0] # Get the root [L0]
        root_childs = root_n.children_nodes # Who are the childrens of the root node [L1]

        if not root_childs:
            # list is empty
            pass
        else:
            # root has child nodes, select the one from this level
            for chd in root_childs:
                print(f"Root child's move: {chd.move_tuple[0],chd.move_tuple[1]} and count: {chd.N}")
                x,y = chd.move_tuple[0],chd.move_tuple[1]
                out_mat[x,y] = chd.N
                
                #* Implemented a mask in the get_action() method
                # # # TODO a function to check for invalid move
                # # #* If True, either Player 1 or Player 2 has placed a token in this place
                # if (self.is_invalid_move((x,y),chd.standardState)):
                #     print(f"pass")
                #     continue 
                # else:
                #     #* else, add count
                #     print(f"set")
                #     out_mat[x,y] = chd.N
                
                #! DOES NOT WORK
                #* In lower levels, if a leaf node is reached, then that node will not have any child nodes
                #print(f"Number of child nodes in this child node: {len(chd.children_nodes)}")
                # if not chd.children_nodes:
                #     pass # list is empty
                # else:
                #     for chdinchd in chd.children_nodes:
                #         #* check for invalid move
                #         try:
                #             print(f"child nodes child move: {chdinchd.move_tuple[0],chdinchd.move_tuple[1]} and count: {chdinchd.N}")
                
                #             x_in, y_in = chdinchd.move_tuple[0], chdinchd.move_tuple[1]
                #             out_mat[x_in,y_in] = chdinchd.N
                #             # #* check for invalid move
                #             # if(self.is_invalid_move((x_in, y_in),chdinchd.standardState)):
                #             #     continue
                #             # else:
                #             #     out_mat[x_in,y_in] = chdinchd.N
                #         except:
                #             continue

        return out_mat

    #* Helper functions
    def is_invalid_move(self, move_tuple, board):
        """
        Checks if the chosen position is already occupied by either Player 1 or Player 2 
        returns: True if invalid, False otherwise
        """
        x = move_tuple[0]
        y = move_tuple[1]
        a = board[0,x,y]!=0 or board[1,x,y]!=0
        return a

    def search_for_matching_node(self, standardState):
        """
        returns True if a node's standardState was matched with query standardState, else False
        """
        
        # Cycle through all the available nodes in the search tree
        for node in self.search_tree:
            chx_flag = False # Set True if node's state and query state are one to one match
            chx_flag = np.array_equal(standardState, node.standardState)
            if(chx_flag):
                return node # This node matched up
        
        return None # no node matched up

    

   
    
# ---------------------------------------------------- EOF -----------------------------------------
  
  
  
  # else:
        #     # this function call follows v = self.search(next_state, end) and a self.new_tree_node
        #     # We create the new node, update parent-child relations, do backprop and then return the new node 
        #     print(f"Creating new node ..... \n")
            
        #     # This follows a self.execute_move method so both standardState and game_end states have updated values
        #     # Here we haven't moved into a new 
        #     new_node = Node(self.unique_node_id, self.policyvalue_dnn, standardState, parent=self.last_parent, action_taken=self.a, game_state = game_end)
            
        #     # Add this new_node into the "child" list of the last parent
        #     self.last_parent.children.append(new_node)

        #     # Query value for this new node
        #     _,v = new_node.polvaldnn(new_node.boardState)

        #     # Backpropagation, update the statistics/count after finding an action 
        #     new_node.update(v)

        #     # Push this node into the tree graph, update unique identifier count and store this node's boardstate
        #     self.add_node_to_tree_graph(new_node)

        #     return new_node   
    
    
# root.game_end_state = game_end # Remember the game end state
            # self.last_parent = root # Root at iteration 1 is the parent for the next children
            # self.add_node_to_tree_graph(root) # Push root node to the tree graph
            # # print(f"The tree graph now has {len(self.tree_graph)} nodes ") # DEBUG

            # # Iteration 1, create the first leaf node
            # move_tuple = root.find_action() # Find action that maximizes the winning chance
            
            # #! Be careful, execute_move overrides standardState
            # # Do a bit of cleanup
            # self.work_state = None
            # next_state = None
            # end = None

            # self.work_state = root.boardState.copy()
            # next_state, end = self.execute_move(self.work_state, move_tuple) # Execute this move, here in "simulation"
            # # TODO write a better comment for this line

            # #* Make the node with the next_state, the root node is its parent, this is now the leaf node
            
            # new_node = Node(self.unique_node_id, self.policyvalue_dnn, next_state, parent=self.last_parent, action_taken=move_tuple, game_state = end)
            
            # # Add this new_node into the "child" list of the last parent
            # self.last_parent.children.append(new_node)

            # # Query value for this new node
            # _,v = new_node.polvaldnn(new_node.boardState)

            # # Backpropagation, update the statistics/count after finding an action 
            # new_node.update(v)
            
            # # Push this node into the tree graph, update unique identifier count
            # self.add_node_to_tree_graph(new_node)

            # #* This ends iteration 1
            # self.do_first_iteration = False # Never repeats again    
    
    

    

# TODO delete these lines    
# print()
# print(f"Number of nodes: {len(self.tree_graph)}, {len(self.tree_graph_boardstates)}")
# print(f"state: {self.board_state}")
# print(f"in_tree_graph -- {in_tree_graph}")

# print()
# print(f"root's state: {self.tree_graph[0].boardState}")
    

# print(f"move x,y: {move_tuple[0]}, {move_tuple[1]}")


 # def find_node_corresponding_to_state(self, state):
    #     """
    #     Helper function, given a state, find node corresponding to this state
    #     """
    #     # Initialize some work variables
    #     in_tree_graph = False 
    #     in_idx = -1
        
    #     in_tree_graph, in_idx = self.search_for_matching_node(state) # Is this state recorded as a node in the tree graph?
        
    #     if(in_tree_graph):
    #         # Extract the matched node
    #         found_node = self.tree_graph[in_idx]
            
    #         return found_node
    #     else:
    #         return None