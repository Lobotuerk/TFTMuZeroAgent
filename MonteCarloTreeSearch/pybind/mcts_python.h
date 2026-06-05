#ifndef MCTS_PYTHON_H
#define MCTS_PYTHON_H

#include "state.h"
#include <vector>
#include <queue>
#include <iomanip>
#include <thread>
#include <future>
#include <algorithm>

#define STARTING_NUMBER_OF_CHILDREN 32   // expected number so that we can preallocate this many pointers
// #define PARALLEL_ROLLOUTS                // Enable parallel rollouts with std::thread (DISABLED due to destructor issues)
#define DEFAULT_NUMBER_OF_THREADS 1      // Default number of parallel rollout threads (disabled)

using namespace std;

/** Ideas for improvements:
 * - state should probably be const like move is (currently problematic because of Quoridor's example)
 * - Instead of a FIFO Queue use a Priority Queue with priority on most probable (better) actions to be explored first
  or maybe this should just be an iterable and we let the implementation decide but these have no superclasses in C++ it seems
 * - vectors, queues and these structures allocate data on the heap anyway so there is little point in using the heap for them
 * so use stack instead?
 */

class MCTS_node {
    bool terminal;
    unsigned int size;
    unsigned int number_of_simulations;
    double score;                       // e.g. number of wins (could be int but double is more general if we use evaluation functions)
    double prior_probability;           // prior probability for PUCT
    MCTS_state *state;                  // current state
    const MCTS_move *move;              // move to get here from parent node's state
    mutable vector<MCTS_node *> children;
    MCTS_node *parent;
    queue<MCTS_move *> untried_actions;
    vector<double> action_probabilities; // stored probabilities for untried actions
    bool owns_state;                    // true if this node should delete the state in destructor
    void backpropagate(double w, int n);
    
    // Configuration for parallel rollouts
    static unsigned int num_rollout_threads;
    
public:
    MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move, bool owns_state = true, double prior_probability = 1.0);
    ~MCTS_node();
    bool is_fully_expanded() const;
    bool is_terminal() const;
    const MCTS_move *get_move() const;
    unsigned int get_size() const;
    double get_prior_probability() const { return prior_probability; }
    void expand();
    void rollout();
    MCTS_node *select_best_child(double c) const;
    MCTS_node *advance_tree(const MCTS_move *m);
    const MCTS_state *get_current_state() const;
    void print_stats() const;
    double calculate_winrate(bool player1turn) const;
    
    // Static method to configure parallel rollouts
    static void set_rollout_threads(unsigned int num_threads);
    static unsigned int get_rollout_threads();
};

class MCTS_tree {
    MCTS_node *root;
public:
    MCTS_tree(MCTS_state *starting_state);
    ~MCTS_tree();
    MCTS_node *select(double c=1.41);        // select child node to expand according to tree policy (UCT)
    MCTS_node *select_best_child();          // select the most promising child of the root node
    void grow_tree(int max_iter, double max_time_in_seconds);
    void advance_tree(const MCTS_move *move);      // if the move is applicable advance the tree, else start over
    unsigned int get_size() const;
    const MCTS_state *get_current_state() const;
    void print_stats() const;
};

class MCTS_agent {                           // example of an agent based on the MCTS_tree. One can also use the tree directly.
    MCTS_tree *tree;
    int max_iter, max_seconds;
public:
    MCTS_agent(MCTS_state *starting_state, int max_iter = 100000, int max_seconds = 30);
    ~MCTS_agent();
    const MCTS_move *genmove(const MCTS_move *enemy_move);
    const MCTS_state *get_current_state() const;
    void feedback() const { tree->print_stats(); }
    
    // Configure parallel rollouts for this agent's tree
    void set_rollout_threads(unsigned int num_threads);
    unsigned int get_rollout_threads() const;
};

// Utility functions for parallel rollouts
namespace ParallelRollouts {
    // Perform a single rollout simulation (thread-safe)
    double perform_rollout(const MCTS_state* state);
    
    // Get optimal number of threads based on hardware
    unsigned int get_optimal_thread_count();
}

#endif