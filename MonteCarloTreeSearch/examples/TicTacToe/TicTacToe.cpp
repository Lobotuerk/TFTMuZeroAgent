#include <iostream>
#include "TicTacToe.h"
#include <ctime>
#include <string>
#include <random>
#include <thread>


using namespace std;


TicTacToe_state::TicTacToe_state() : MCTS_state(), turn('x') {
    // initialize board as empty
    for (int i = 0 ; i < 9 ; i++) {
        board[i / 3][i % 3] = ' ';
    }
    // calculate winner
    winner = calculate_winner();
}

TicTacToe_state::TicTacToe_state(const TicTacToe_state &other)
        : MCTS_state(other), turn(other.turn), winner(other.winner) {
    // copy board
    for (int i = 0 ; i < 9 ; i++) {
        board[i / 3][i % 3] = other.board[i / 3][i % 3];
    }
}

bool TicTacToe_state::player_won(char player) const {
    if (player != 'x' && player != 'o') cerr << "Warning: check winner for unknown player" << endl;
    for (int i = 0 ; i < 3 ; i++) {
        if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
        if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
    }
    return (board[0][0] == player && board[1][1] == player && board[2][2] == player) ||
           (board[0][2] == player && board[1][1] == player && board[2][0] == player);
}

bool TicTacToe_state::is_terminal() const {
    return winner != ' ';
}

char TicTacToe_state::get_turn() const { return turn; }

char TicTacToe_state::get_winner() const { return winner; }

void TicTacToe_state::change_turn() {
    turn = (turn == 'x') ? 'o' : 'x';
}

MCTS_state *TicTacToe_state::next_state(const MCTS_move *move) const {
    // Note: We have to manually cast it to its correct type
    TicTacToe_move *m = (TicTacToe_move *) move;
    TicTacToe_state *new_state = new TicTacToe_state(*this);  // create new state from current
    if (new_state->board[m->x][m->y] == ' ') {
        new_state->board[m->x][m->y] = m->player;             // play move
        new_state->winner = new_state->calculate_winner();    // check again for a winner
        new_state->change_turn();
    } else {
        cerr << "Warning: Illegal move (" << m->x << ", " << m->y << ")" << endl;
        return NULL;
    }
    return new_state;
}

queue<MCTS_move *> *TicTacToe_state::actions_to_try() const {
    queue<MCTS_move *> *Q = new queue<MCTS_move *>();
    for (int i = 0 ; i < 9 ; i++) {
        if (board[i / 3][i % 3] == ' ') {
            Q->push(new TicTacToe_move(i / 3, i % 3, turn));
        }
    }
    return Q;
}

double TicTacToe_state::rollout() const {
    if (is_terminal()) return (winner == 'x') ? 1.0 : (winner == 'd') ? 0.5 : 0.0;
    // Simulate a completely random game
    // Note: dequeue is not very efficient for random accesses but vector is not efficient for deletes
    deque<int> available;
    for (int i = 0 ; i < 9 ; i++){
        if (board[i / 3][i % 3] == ' ') {
            available.push_front(i);
        }
    }
    long long r;
    int a;
    TicTacToe_state *curstate = (TicTacToe_state *) this;   // TODO: ignore const...
    
    // Thread-safe random number generation
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    
    bool first = true;
    do {
        if (available.empty()) {
            cerr << "Warning: Ran out of available moves and state is not terminal?";
            return 0.0;
        }
        std::uniform_int_distribution<> dis(0, available.size() - 1);
        r = dis(gen);
        a = available[r];
        TicTacToe_move move(a / 3, a % 3, curstate->turn);
        available.erase(available.begin() + r);    // delete from available moves
        TicTacToe_state *old = curstate;
        curstate = (TicTacToe_state *) curstate->next_state(&move);
        if (!first) {
            delete old;
        }
        first = false;
    } while (!curstate->is_terminal());
    double res = (curstate->winner == 'x') ? 1.0 : (curstate->winner == 'd') ? 0.5 : 0.0;
    delete curstate;
    return res;
}

double TicTacToe_state::heuristic_rollout() const {
    if (is_terminal()) return (winner == 'x') ? 1.0 : (winner == 'd') ? 0.5 : 0.0;
    
    // Heuristic-guided simulation: prioritize winning, blocking, center, corners
    deque<int> available;
    for (int i = 0 ; i < 9 ; i++){
        if (board[i / 3][i % 3] == ' ') {
            available.push_front(i);
        }
    }
    
    TicTacToe_state *curstate = (TicTacToe_state *) this;
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    
    bool first = true;
    do {
        if (available.empty()) {
            cerr << "Warning: Ran out of available moves and state is not terminal?";
            return 0.0;
        }
        
        // Find best move using heuristics
        int best_move = find_best_heuristic_move(curstate, available);
        
        // Remove chosen move from available
        for (auto it = available.begin(); it != available.end(); ++it) {
            if (*it == best_move) {
                available.erase(it);
                break;
            }
        }
        
        TicTacToe_move move(best_move / 3, best_move % 3, curstate->turn);
        TicTacToe_state *old = curstate;
        curstate = (TicTacToe_state *) curstate->next_state(&move);
        if (!first) {
            delete old;
        }
        first = false;
    } while (!curstate->is_terminal());
    
    double res = (curstate->winner == 'x') ? 1.0 : (curstate->winner == 'd') ? 0.5 : 0.0;
    delete curstate;
    return res;
}

int TicTacToe_state::find_best_heuristic_move(TicTacToe_state* state, const deque<int>& available) const {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    
    // Priority 1: Win if possible
    for (int pos : available) {
        TicTacToe_move test_move(pos / 3, pos % 3, state->turn);
        TicTacToe_state* test_state = (TicTacToe_state*)state->next_state(&test_move);
        if (test_state->winner == state->turn) {
            delete test_state;
            return pos;
        }
        delete test_state;
    }
    
    // Priority 2: Block opponent's win
    char opponent = (state->turn == 'x') ? 'o' : 'x';
    for (int pos : available) {
        TicTacToe_move test_move(pos / 3, pos % 3, opponent);
        TicTacToe_state* test_state = (TicTacToe_state*)state->next_state(&test_move);
        if (test_state->winner == opponent) {
            delete test_state;
            return pos;
        }
        delete test_state;
    }
    
    // Priority 3: Take center (position 4)
    for (int pos : available) {
        if (pos == 4) return pos;
    }
    
    // Priority 4: Take corners (0, 2, 6, 8)
    vector<int> corners = {0, 2, 6, 8};
    for (int corner : corners) {
        for (int pos : available) {
            if (pos == corner) return pos;
        }
    }
    
    // Priority 5: Random choice from remaining
    std::uniform_int_distribution<> dis(0, available.size() - 1);
    return available[dis(gen)];
}

double TicTacToe_state::evaluate_move(const MCTS_move* move) const {
    const TicTacToe_move* ttt_move = static_cast<const TicTacToe_move*>(move);
    int pos = ttt_move->x * 3 + ttt_move->y;
    
    // Test if this move wins the game
    TicTacToe_state* test_state = (TicTacToe_state*)next_state(move);
    if (test_state->winner == turn) {
        delete test_state;
        return 1.0;  // Winning move
    }
    delete test_state;
    
    // Test if this move blocks opponent's win
    char opponent = (turn == 'x') ? 'o' : 'x';
    TicTacToe_move opponent_move(ttt_move->x, ttt_move->y, opponent);
    TicTacToe_state* opponent_test = (TicTacToe_state*)next_state(&opponent_move);
    if (opponent_test->winner == opponent) {
        delete opponent_test;
        return 0.8;  // Blocking move
    }
    delete opponent_test;
    
    // Positional preferences
    if (pos == 4) return 0.6;  // Center
    if (pos == 0 || pos == 2 || pos == 6 || pos == 8) return 0.4;  // Corners
    return 0.2;  // Edges
}

double TicTacToe_state::evaluate_position() const {
    if (is_terminal()) {
        if (winner == 'x') return 1.0;
        if (winner == 'o') return 0.0;
        return 0.5;  // Draw
    }
    
    // Simple evaluation: count winning opportunities
    double x_score = count_winning_lines('x');
    double o_score = count_winning_lines('o');
    double total = x_score + o_score;
    
    if (total == 0) return 0.5;
    return x_score / total;
}

double TicTacToe_state::count_winning_lines(char player) const {
    double count = 0.0;
    
    // Check all lines (rows, columns, diagonals)
    // Rows
    for (int i = 0; i < 3; i++) {
        if (can_win_line(board[i][0], board[i][1], board[i][2], player)) count += 1.0;
    }
    
    // Columns
    for (int j = 0; j < 3; j++) {
        if (can_win_line(board[0][j], board[1][j], board[2][j], player)) count += 1.0;
    }
    
    // Diagonals
    if (can_win_line(board[0][0], board[1][1], board[2][2], player)) count += 1.0;
    if (can_win_line(board[0][2], board[1][1], board[2][0], player)) count += 1.0;
    
    return count;
}

bool TicTacToe_state::can_win_line(char pos1, char pos2, char pos3, char player) const {
    char opponent = (player == 'x') ? 'o' : 'x';
    // A line can be won if it contains the player's pieces and no opponent pieces
    int player_count = 0;
    int opponent_count = 0;
    
    if (pos1 == player) player_count++;
    else if (pos1 == opponent) opponent_count++;
    
    if (pos2 == player) player_count++;
    else if (pos2 == opponent) opponent_count++;
    
    if (pos3 == player) player_count++;
    else if (pos3 == opponent) opponent_count++;
    
    return opponent_count == 0;  // Can win if no opponent pieces
}

void TicTacToe_state::print() const {
    printf(" %c | %c | %c\n---+---+---\n %c | %c | %c\n---+---+---\n %c | %c | %c\n",
           board[0][0], board[0][1], board[0][2],
           board[1][0], board[1][1], board[1][2],
           board[2][0], board[2][1], board[2][2]);
}

char TicTacToe_state::calculate_winner() const {
    if (player_won('x')) return 'x';
    else if (player_won('o')) return 'o';
    bool all_taken = true;
    for (int i = 0 ; i < 9 ; i++) {
        if (board[i / 3][i % 3] == ' ') {
            all_taken = false;
            break;
        }
    }
    if (all_taken) return 'd';   // draw
    else return ' ';             // no-one yet
}

bool TicTacToe_move::operator==(const MCTS_move &other) const {
    const TicTacToe_move &o = (const TicTacToe_move &) other;        // Note: Casting necessary
    return x == o.x && y == o.y && player == o.player;
}

std::string TicTacToe_move::sprint() const {
    return std::string("(") + std::to_string(x) + "," + std::to_string(y) + "," + player + ")";
}

std::vector<double> TicTacToe_move::to_numpy() const {
    // Convert move to numpy-like representation: [x, y, player_as_double]
    double player_val = (player == 'x') ? 1.0 : 0.0;
    return {static_cast<double>(x), static_cast<double>(y), player_val};
}

std::vector<int> TicTacToe_move::to_env_action() const {
    // Convert move to environment action format: [x, y, player_as_int]
    int player_val = (player == 'x') ? 1 : 0;
    return {x, y, player_val};
}
