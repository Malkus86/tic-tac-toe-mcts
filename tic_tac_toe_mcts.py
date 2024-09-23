import math
import random
from graphviz import Digraph

"""Tic-Tac-Toe game implementation using Monte Carlo Tree Search (MCTS) algorithm."""

PLAYER = {"HUMAN": 0, "MACHINE": 1}

def get_other_player(player):
    """Returns the opposite player.

    Args:
        player (int): Current player (PLAYER["HUMAN"] or PLAYER["MACHINE"]).

    Returns:
        int: The opposite player.
    """
    return PLAYER["MACHINE"] if player == PLAYER["HUMAN"] else PLAYER["HUMAN"]

class TicTacToeBoard:
    """Represents the Tic-Tac-Toe game board and game state."""

    def __init__(self):
        """Initializes the Tic-Tac-Toe board."""
        self.grid = ["" for _ in range(9)]
        self.current_player = PLAYER["HUMAN"]  # Start with human player

    def make_move(self, move):
        """Applies a move to the board if it's legal.

        Args:
            move (GameMove): The move to be applied.

        Returns:
            bool: True if the move was successfully applied, False otherwise.
        """
        if move is None:
            return False  # No move to make
        if self.grid[move.position] == "":
            self.grid[move.position] = "h" if move.player == PLAYER["HUMAN"] else "m"
            self.current_player = get_other_player(self.current_player)
            return True
        return False

    def get_legal_positions(self):
        """Returns a list of empty positions on the board.

        Returns:
            list: Indices of empty positions on the board.
        """
        return [i for i, pos in enumerate(self.grid) if pos == ""]

    def has_legal_positions(self):
        """Checks if there are any empty positions on the board.

        Returns:
            bool: True if there are empty positions, False otherwise.
        """
        return len(self.get_legal_positions()) > 0

    def check_win(self):
        """Checks if the game has been won or ended in a draw.

        Returns:
            str: 'h' for human win, 'm' for machine win, 'v' for draw, or '' if the game is not over.
        """
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        # Check for win combinations
        for combo in win_combinations:
            if self.grid[combo[0]] != "" and self.grid[combo[0]] == self.grid[combo[1]] == self.grid[combo[2]]:
                return self.grid[combo[0]]

        # Check for draw
        if all(pos != "" for pos in self.grid):
            return "v"  # Draw

        return ""  # Game not over


    def copy(self):
        """Creates a deep copy of the current board state.

        Returns:
            TicTacToeBoard: A new instance with the same state as the current board.
        """
        new_board = TicTacToeBoard()
        new_board.grid = self.grid.copy()
        new_board.current_player = self.current_player
        return new_board

class GameMove:
    """Represents a move in the Tic-Tac-Toe game."""

    def __init__(self, player, position):
        """Initializes a game move.

        Args:
            player (int): The player making the move (PLAYER["HUMAN"] or PLAYER["MACHINE"]).
            position (int): The position on the board where the move is made (0-8).
        """
        self.player = player
        self.position = position

    def copy(self):
        """Creates a copy of the current move.

        Returns:
            GameMove: A new instance with the same state as the current move.
        """
        return GameMove(self.player, self.position)

class GameNode:
    """Represents a node in the Monte Carlo Tree Search."""

    def __init__(self, move, board_state):
        """Initializes a game node.

        Args:
            move (GameMove): The move that led to this node.
            board_state (TicTacToeBoard): The board state at this node.
        """
        self.move = move
        self.board_state = board_state
        self.wins = 0
        self.draws = 0
        self.simulations = 0

    def copy(self):
        """Creates a deep copy of the current game node.

        Returns:
            GameNode: A new instance with the same state as the current node.
        """
        new_node = GameNode(self.move.copy() if self.move else None, self.board_state.copy())
        new_node.wins = self.wins
        new_node.draws = self.draws
        new_node.simulations = self.simulations
        return new_node

class Node:
    """Represents a node in the game tree."""

    def __init__(self, data, tree, id=-1, children_id=None, parent_id=-1):
        """Initializes a node in the game tree.

        Args:
            data (GameNode): The game data stored in this node.
            tree (Tree): The tree this node belongs to.
            id (int, optional): The unique identifier for this node. Defaults to -1.
            children_id (list, optional): List of child node ids. Defaults to None.
            parent_id (int, optional): The id of the parent node. Defaults to -1.
        """
        self.data = data
        self.tree = tree
        self.id = id
        self.children_id = children_id if children_id is not None else []
        self.parent_id = parent_id

    def copy(self):
        """Creates a copy of the current node.

        Returns:
            Node: A new instance with the same state as the current node.
        """
        return Node(self.data.copy(), self.tree, self.id, self.children_id.copy(), self.parent_id)

    def is_leaf(self):
        """Checks if the node is a leaf node (has no children).

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return len(self.children_id) == 0

    def is_root(self):
        """Checks if the node is the root node of the tree.

        Returns:
            bool: True if the node is the root, False otherwise.
        """
        return self.id == 0

class Tree:
    """Represents the game tree for Monte Carlo Tree Search."""

    def __init__(self, root_data):
        """Initializes the game tree with a root node.

        Args:
            root_data (GameNode): The data for the root node.
        """
        self.nodes = [Node(root_data, self, 0)]

    def get(self, id):
        """Retrieves a node by its id.

        Args:
            id (int): The id of the node to retrieve.

        Returns:
            Node: The node with the given id.
        """
        return self.nodes[id]

    def insert(self, node_data, parent):
        """Inserts a new node into the tree.

        Args:
            node_data (GameNode): The data for the new node.
            parent (Node): The parent node for the new node.

        Returns:
            Node: The newly created and inserted node.
        """
        new_node = Node(node_data, self, len(self.nodes), parent_id=parent.id)
        self.nodes.append(new_node)
        parent.children_id.append(new_node.id)
        return new_node

    def get_parent(self, node):
        """Retrieves the parent of a given node.

        Args:
            node (Node): The node whose parent is to be retrieved.

        Returns:
            Node: The parent node.
        """
        return self.nodes[node.parent_id]

    def get_children(self, node):
        """Retrieves all children of a given node.

        Args:
            node (Node): The node whose children are to be retrieved.

        Returns:
            list: A list of child nodes.
        """
        return [self.nodes[child_id] for child_id in node.children_id]

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm for Tic-Tac-Toe."""

    def __init__(self, model, player=PLAYER["MACHINE"]):
        """Initializes the MCTS algorithm.

        Args:
            model (TicTacToeBoard): The initial game state.
            player (int, optional): The player to optimize for. Defaults to PLAYER["MACHINE"].
        """
        self.model = model
        initial_board = model.copy()
        root_node = GameNode(None, initial_board)  # Root node has no move
        self.tree = Tree(root_node)

    def run_search(self, iterations):
        """Runs the Monte Carlo Tree Search for a specified number of iterations.

        Args:
            iterations (int, optional): The number of search iterations to perform.

        Returns:
            dict: A dictionary containing the best move and a trace of the search process.
        """
        trace = []
        for _ in range(iterations):
            iteration_trace = self.run_search_iteration()
            trace.append(iteration_trace)

        best_move_node = max(self.tree.get_children(self.tree.get(0)), key=lambda x: x.data.simulations)
        trace.append([{"kind": "finish", "node_id": best_move_node.id, "old_data": None, "new_data": None}])
        return {"move": best_move_node.data.move, "trace": trace}

    def run_search_iteration(self):
        """Performs a single iteration of the Monte Carlo Tree Search.

        Returns:
            list: A trace of the actions performed during this iteration.
        """
        select_res = self.select(self.model.copy())
        expand_res = self.expand(select_res["node"], select_res["model"])
        simulation = self.simulate(expand_res["node"], expand_res["model"])
        backpropagated = self.backpropagate(expand_res["node"], simulation["winner_icon"], simulation["is_draw"])

        return (select_res["actions"] + expand_res["actions"] + 
                simulation["actions"] + backpropagated["actions"])

    def get_best_child_ucb1(self, node):
        """Selects the best child node based on the UCB1 formula.

        Args:
            node (Node): The parent node.

        Returns:
            Node: The child node with the highest UCB1 value.
        """
        return max(self.tree.get_children(node), key=lambda c: ucb1(c, node))

    def select(self, model):
        """Selects a leaf node in the tree to expand.

        Args:
            model (TicTacToeBoard): The current game state.

        Returns:
            dict: A dictionary containing the selected node, updated model, and a trace of actions.
        """
        node = self.tree.get(0)
        actions = [{"kind": "selection", "node_id": node.id, "old_data": None, "new_data": None}]

        while not node.is_leaf() and self.is_fully_explored(node, model):
            node = self.get_best_child_ucb1(node)
            if node.data.move is not None:
                success = model.make_move(node.data.move)
                if not success:
                    print(f"Selection Error: Failed to make move at position {node.data.move.position}")
                actions.append({"kind": "selection", "node_id": node.id, "old_data": None, "new_data": None})

        return {"node": node, "model": model, "actions": actions}

    def expand(self, node, model):
        """Expands the selected node by adding a child node.

        Args:
            node (Node): The node to expand.
            model (TicTacToeBoard): The current game state.

        Returns:
            dict: A dictionary containing the expanded node, updated model, and a trace of actions.
        """
        actions = []
        if model.check_win() == "":
            legal_positions = self.get_available_plays(node, model)
            if legal_positions:
                random_pos = random.choice(legal_positions)
                other_player = get_other_player(node.data.move.player if node.data.move else PLAYER["HUMAN"])
                random_move = GameMove(other_player, random_pos)
                success = model.make_move(random_move)
                if not success:
                    print(f"Expansion Error: Failed to make move at position {random_move.position}")

                new_board_state = model.copy()
                expanded_node = self.tree.insert(GameNode(random_move, new_board_state), node)
                actions = [{"kind": "expansion", "node_id": expanded_node.id, "old_data": None, "new_data": None}]
            else:
                expanded_node = node
        else:
            expanded_node = node

        return {"node": expanded_node, "model": model, "actions": actions}

    def simulate(self, node, model):
        """Simulates a random play-out from the given node to the end of the game.

        Args:
            node (Node): The starting node for the simulation.
            model (TicTacToeBoard): The current game state.

        Returns:
            dict: A dictionary containing the simulation result and a trace of actions.
        """
        current_player = node.data.move.player if node.data.move else PLAYER["HUMAN"]
        while model.check_win() == "":
            current_player = get_other_player(current_player)
            legal_moves = model.get_legal_positions()
            if legal_moves:
                random_pos = random.choice(legal_moves)
                random_move = GameMove(current_player, random_pos)
                success = model.make_move(random_move)
                if not success:
                    print(f"Simulation Error: Failed to make move at position {random_move.position}")
            else:
                break

        winner_icon = model.check_win()
        is_draw = winner_icon == "v"

        return {
            "winner_icon": winner_icon,
            "is_draw": is_draw,
            "actions": [{
                "kind": "simulation",
                "node_id": node.id,
                "old_data": None,
                "new_data": {"result": winner_icon, "board": model}
            }]
        }

    def backpropagate(self, node, winner, is_draw):
        """Performs backpropagation in the Monte Carlo Tree Search.

        Updates the statistics of nodes in the path from the given node
        to the root of the tree based on the game result.

        Args:
            node (Node): The leaf node to start backpropagation from.
            winner (str): The winner of the game. 'm' for machine win,
                'h' for human win, 'v' for draw.
            is_draw (bool): Indicates whether the game ended in a draw.

        Returns:
            dict: A dictionary containing a list of actions performed during
            backpropagation. Each action is a dictionary with keys:
            - 'kind': Always 'backpropagation'
            - 'node_id': The ID of the node being updated
            - 'old_data': Node's statistics before update
            - 'new_data': Node's statistics after update
        """
        actions = []
        current_node = node

        while True:
            action = {
                "kind": "backpropagation",
                "node_id": current_node.id,
                "old_data": {"old_wins": current_node.data.wins, "old_draws": current_node.data.draws, "old_visits": current_node.data.simulations},
                "new_data": None
            }

            current_node.data.simulations += 1
            
            if winner == "m":  # machine wins
                current_node.data.wins += 1
                # print(f"Node {current_node.id}: AI wins, wins incremented to {current_node.data.wins}")
            elif is_draw:  # draw
                if current_node.data.simulations > 0:  # only increment draws if the node has been visited
                    current_node.data.draws += 1 
            #         print(f"Node {current_node.id}: Draw, draws incremented to {current_node.data.draws}")
            # else:  # human wins
            #     print(f"Node {current_node.id}: Human wins, wins and draws not incremented")

            action["new_data"] = {
                "new_wins": current_node.data.wins,
                "new_draws": current_node.data.draws,
                "new_visits": current_node.data.simulations
            }

            actions.insert(0, action)

            if current_node.is_root():
                break

            current_node = self.tree.get_parent(current_node)

        return {"actions": actions}

    def is_fully_explored(self, node, model):
        """Checks if all possible moves from the current node have been explored.

        Args:
            node (Node): The node to check.
            model (TicTacToeBoard): The current game state.

        Returns:
            bool: True if all moves have been explored, False otherwise.
        """
        return len(self.get_available_plays(node, model)) == 0

    def get_available_plays(self, node, model):
        """Gets all legal moves that haven't been explored from the current node.

        Args:
            node (Node): The current node.
            model (TicTacToeBoard): The current game state.

        Returns:
            list: A list of legal positions that haven't been explored.
        """
        children = self.tree.get_children(node)
        return [pos for pos in model.get_legal_positions() 
                if not any(child.data.move.position == pos for child in children)]

def ucb1(node, parent):
    """Calculates the UCB1 value for a node.

    Args:
        node (Node): The node to calculate the UCB1 value for.
        parent (Node): The parent of the node.

    Returns:
        float: The UCB1 value of the node.
    """
    if node.data.simulations == 0:
        return float('inf')
    exploitation = (node.data.wins + node.data.draws) / node.data.simulations
    exploration = math.sqrt(2 * math.log(parent.data.simulations) / node.data.simulations)
    return exploitation + exploration

def format_board(board):
    """Formats the Tic-Tac-Toe board as a string for console output.

    Args:
        board (TicTacToeBoard): The board to format.

    Returns:
        str: A string representation of the board.
    """
    if board is None:
        return "None"
    symbols = {'h': 'X', 'm': 'O', '': ' '}
    lines = []
    for i in range(3):
        row = ' | '.join([symbols[board.grid[i*3 + j]] for j in range(3)])
        lines.append(f" {row} ")
    board_str = '\n-----------\n'.join(lines)
    return board_str

def format_board_for_graphviz(board):
    """Formats the Tic-Tac-Toe board as a string for Graphviz visualization.

    Args:
        board (TicTacToeBoard): The board to format.

    Returns:
        str: A string representation of the board suitable for Graphviz.
    """
    if board is None:
        return "None"
    symbols = {'h': 'X', 'm': 'O', '': ' '}
    lines = []
    for i in range(3):
        row = ' | '.join([symbols[board.grid[i*3 + j]] for j in range(3)])
        lines.append(f" {row} ")
    board_str = '\\n-----------\\n'.join(lines)
    return board_str

def draw_tree(root):
    """Draws the MCTS tree using Graphviz.

    Args:
        root (Node): The root node of the MCTS tree.
    """
    dot = Digraph(comment='MCTS Tree', engine='dot')
    dot.attr('node', shape='box', style='filled', fontsize='10', fontname='Courier')

    def add_nodes_edges(node, parent_id=None, depth=0, max_depth=5):
        """Recursively adds nodes and edges to the Graphviz diagram.

        Args:
            node (Node): The current node to add.
            parent_id (str, optional): The ID of the parent node. Defaults to None.
            depth (int, optional): The current depth in the tree. Defaults to 0.
            max_depth (int, optional): The maximum depth to visualize. Defaults to 2.
        """
        if depth > max_depth:
            return
        node_id = str(id(node))
        board_str = format_board_for_graphviz(node.data.board_state)
        
        if node.is_root():
            label = f"{board_str}\\nRoot\\nV:{node.data.simulations}"
        else:
            parent = node.tree.get_parent(node)
            ucb_value = ucb1(node, parent)
            win_rate = (node.data.wins + node.data.draws) / node.data.simulations if node.data.simulations > 0 else 0
            move_str = f"Move: {node.data.move.position}"
            label = f"{board_str}\\n{move_str}\\nV:{node.data.simulations}\\nW:{node.data.wins:.1f}\\nD:{node.data.draws:.1f}\\nWR:{win_rate:.2f}\\nUCB1:{ucb_value:.4f}"

        dot.node(node_id, label=label, fillcolor='white')
        if parent_id:
            dot.edge(parent_id, node_id)
        for child in node.tree.get_children(node):
            add_nodes_edges(child, node_id, depth + 1, max_depth)

    add_nodes_edges(root)
    dot.render('mcts_tree', view=True, format='pdf')

def play_game():
    """Runs the Tic-Tac-Toe game, allowing a human player to play against the AI."""
    board = TicTacToeBoard()
    mcts = MCTS(board)

    while board.check_win() == "":
        print(format_board(board))
        if board.current_player == PLAYER["HUMAN"]:
            while True:
                try:
                    position = int(input("Enter your move (0-8): "))
                    if 0 <= position <= 8 and position in board.get_legal_positions():
                        move = GameMove(PLAYER["HUMAN"], position)
                        board.make_move(move)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 8.")
        else:
            print("AI is thinking...")
            result = mcts.run_search(iterations=1000)
            board.make_move(result["move"])
            print(f"AI moved to position {result['move'].position}")
            draw_tree(mcts.tree.get(0))

        mcts = MCTS(board)  # Reset the MCTS tree for the new board state

    print(format_board(board))
    winner = board.check_win()
    if winner == "h":
        print("You win!")
    elif winner == "m":
        print("AI wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()