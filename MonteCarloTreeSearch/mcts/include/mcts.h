#ifndef MCTS_H
#define MCTS_H

#include "state.h"
#include <vector>
#include <queue>
#include <iomanip>

#define STARTING_NUMBER_OF_CHILDREN 32   // expected number so that we can preallocate this many pointers
#define PARALLEL_ROLLOUTS                // whether or not to do multiple parallel rollouts

#ifdef PARALLEL_ROLLOUTS
#include "JobScheduler.h"
#endif

using namespace std;

// Rollout strategy enumeration
enum class RolloutStrategy {
    RANDOM,           // Pure random rollouts (default)
    HEURISTIC,        // Use heuristic_rollout() method
    MIXED,            // Mix of random and heuristic (configurable ratio)
    HEAVY             // Deeper heuristic evaluation
};

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
    void backpropagate(double w, int n);
    
    // Static rollout configuration
    static RolloutStrategy rollout_strategy;
    static double heuristic_ratio;      // For MIXED strategy: ratio of heuristic vs random rollouts
    
public:
    MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move, double prior_probability = 1.0);
    ~MCTS_node();
    bool is_fully_expanded() const;
    bool is_terminal() const;
    const MCTS_move *get_move() const;
    unsigned int get_size() const;
    double get_prior_probability() const { return prior_probability; }
    void expand();
    void rollout();
    void rollout_with_strategy(RolloutStrategy strategy);
    MCTS_node *select_best_child(double c) const;
    MCTS_node *advance_tree(const MCTS_move *m);
    const MCTS_state *get_current_state() const;
    void print_stats() const;
    double calculate_winrate(bool self_side_turn) const;
    
    // Static configuration methods
    static void set_rollout_strategy(RolloutStrategy strategy);
    static RolloutStrategy get_rollout_strategy();
    static void set_heuristic_ratio(double ratio);
    static double get_heuristic_ratio();
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
    
    // Rollout strategy configuration
    void set_rollout_strategy(RolloutStrategy strategy);
    RolloutStrategy get_rollout_strategy() const;
    void set_heuristic_ratio(double ratio);
    double get_heuristic_ratio() const;
};


#ifdef PARALLEL_ROLLOUTS
class RolloutJob : public Job {             // class for performing parallel simulations using a thread pool
    double *score;
    const MCTS_state *state;
    RolloutStrategy strategy;
public:
    RolloutJob(const MCTS_state *state, double *score, RolloutStrategy strat = RolloutStrategy::RANDOM) 
        : Job(), state(state), score(score), strategy(strat) {}
    void run() override {
        // Execute rollout based on strategy
        switch (strategy) {
            case RolloutStrategy::HEURISTIC:
                *score = state->heuristic_rollout();
                break;
            case RolloutStrategy::MIXED:
                // Use heuristic_ratio to decide which rollout to use
                if (static_cast<double>(rand()) / RAND_MAX < MCTS_node::get_heuristic_ratio()) {
                    *score = state->heuristic_rollout();
                } else {
                    *score = state->rollout();
                }
                break;
            case RolloutStrategy::HEAVY:
                *score = state->heuristic_rollout();
                break;
            case RolloutStrategy::RANDOM:
            default:
                *score = state->rollout();
                break;
        }
    }
};
#endif


#endif
