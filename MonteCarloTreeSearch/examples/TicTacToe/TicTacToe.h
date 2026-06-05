#ifndef MCTS_TICTACTOE_H
#define MCTS_TICTACTOE_H

#include "../../mcts/include/state.h"
#include <deque>

using namespace std;


class TicTacToe_state : public MCTS_state {
    char board[3][3]{};
    bool player_won(char player) const;
    char calculate_winner() const;
    char turn, winner;
    void change_turn();
    
    // Heuristic helper methods
    int find_best_heuristic_move(TicTacToe_state* state, const deque<int>& available) const;
    double count_winning_lines(char player) const;
    bool can_win_line(char pos1, char pos2, char pos3, char player) const;
    
public:
    TicTacToe_state();
    TicTacToe_state(const TicTacToe_state &other);
    char get_turn() const;
    char get_winner() const;
    bool is_terminal() const override;
    MCTS_state *next_state(const MCTS_move *move) const override;
    queue<MCTS_move *> *actions_to_try() const override;
    double rollout() const override;                        // the rollout simulation in MCTS
    void print() const override;
    bool is_self_side_turn() const override { return turn == 'x'; }
    MCTS_state* clone() const override { return new TicTacToe_state(*this); }
    
    // Heuristic rollout methods
    double heuristic_rollout() const override;
    double evaluate_move(const MCTS_move* move) const override;
    double evaluate_position() const override;
};


struct TicTacToe_move : public MCTS_move {
    int x, y;
    char player;
    TicTacToe_move(int x, int y, char p) : x(x), y(y), player(p) {}
    bool operator==(const MCTS_move& other) const override;
    std::string sprint() const override;
    std::vector<double> to_numpy() const override;
    std::vector<int> to_env_action() const override;
};

#endif
