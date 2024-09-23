# Tic-Tac-Toe with Monte Carlo Tree Search

This project implements an AI player for the classic game of Tic-Tac-Toe using the Monte Carlo Tree Search (MCTS) algorithm. The AI provides a challenging opponent by utilizing MCTS to make intelligent decisions based on simulated game plays.

![](./mcts_tree.png)

## Features

- Play Tic-Tac-Toe against an AI opponent powered by MCTS
- Visualize the MCTS decision tree as a PDF
- Adjustable search iterations for balancing decision quality and speed

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Malkus86/tic-tac-toe-mcts.git
   cd tic-tac-toe-mcts
   ```

2. Install the required dependencies:
   ```
   pip install graphviz
   ```

   Note: You also need to install Graphviz on your system. Visit the [Graphviz website](https://graphviz.org/download/) for installation instructions specific to your operating system.

## Usage

Run the game using Python:

```
python tic_tac_toe_mcts.py
```

Follow the on-screen prompts to play the game. Enter your moves by selecting a position from 0 to 8, corresponding to the Tic-Tac-Toe board layout:

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

The AI will use MCTS to determine its moves. After each AI turn, a visualization of the decision tree will be generated as a PDF file named 'mcts_tree.pdf' in the same directory. This PDF will show the current state of the MCTS tree, including node values and UCB1 scores.

## Dependencies

- Python 3.x
- Graphviz (for generating the MCTS tree visualization PDF)

Enjoy playing against the MCTS-powered AI and exploring the decision-making process through the generated PDFs!