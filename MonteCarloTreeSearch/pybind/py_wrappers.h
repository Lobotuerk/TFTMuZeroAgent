#ifndef PY_WRAPPERS_H
#define PY_WRAPPERS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../mcts/include/state.h"
#include "mcts_python.h"  // Use Python-specific header that conditionally includes JobScheduler

#include <vector>
#include <memory>
#include <queue>

// Forward declaration of RolloutStrategy
enum class RolloutStrategy {
    RANDOM,           // Pure random rollouts (default)
    HEURISTIC,        // Use heuristic_rollout() method
    MIXED,            // Mix of random and heuristic (configurable ratio)
    HEAVY             // Deeper heuristic evaluation
};

namespace py = pybind11;

/**
 * Internal C++ move wrapper that stores Python move data
 * This allows C++ MCTS to work with moves without exposing Python objects
 */
class PythonMoveWrapper : public MCTS_move {
private:
    py::object python_move;  // Keep the Python move alive
    std::string move_string; // Cached string representation
    
public:
    PythonMoveWrapper(py::object py_move) : python_move(py_move) {
        try {
            move_string = py_move.attr("sprint")().cast<std::string>();
        } catch (const std::exception& e) {
            move_string = "PythonMove";
        }
    }
    
    bool operator==(const MCTS_move& other) const override {
        const PythonMoveWrapper* other_wrapper = dynamic_cast<const PythonMoveWrapper*>(&other);
        if (other_wrapper) {
            try {
                // Use Python's __eq__ method for comparison
                return python_move.attr("__eq__")(other_wrapper->python_move).cast<bool>();
            } catch (const std::exception& e) {
                return false;
            }
        }
        return false;
    }
    
    std::string sprint() const override {
        return move_string;
    }
    
    std::vector<double> to_numpy() const override {
        try {
            // Call Python move's to_numpy() method
            py::list py_result = python_move.attr("to_numpy")();
            std::vector<double> result;
            for (auto item : py_result) {
                result.push_back(item.cast<double>());
            }
            return result;
        } catch (const std::exception& e) {
            // Fallback to empty vector if method fails
            return std::vector<double>();
        }
    }
    
    std::vector<int> to_env_action() const override {
        try {
            // Call Python move's to_env_action() method
            py::list py_result = python_move.attr("to_env_action")();
            std::vector<int> result;
            for (auto item : py_result) {
                result.push_back(item.cast<int>());
            }
            return result;
        } catch (const std::exception& e) {
            // Fallback to empty vector if method fails
            return std::vector<int>();
        }
    }
    
    py::object get_python_move() const {
        return python_move;
    }
};

/**
 * C++ state class that holds a Python game state object
 * This enables full C++ ownership while preserving Python game logic
 */
class SerializedPythonState : public MCTS_state {
private:
    py::object python_state;  // Store the Python state object
    mutable std::vector<py::object> cached_python_moves; // Keep Python moves alive
    
public:
    SerializedPythonState(py::object python_state);
    ~SerializedPythonState() override = default;
    
    // MCTS_state interface
    std::queue<MCTS_move*>* actions_to_try() const override;
    MCTS_state* next_state(const MCTS_move* move) const override;
    double rollout() const override;
    bool is_terminal() const override;
    void print() const override;
    bool is_self_side_turn() const override;
    MCTS_state* clone() const override;
    std::vector<double> get_action_probabilities() const override;
    
    // Helper to find original Python move from C++ pointer
    py::object find_python_move(const MCTS_move* cpp_move) const;
};

namespace py = pybind11;

/**
 * Trampoline class for MCTS_move to enable Python inheritance
 * Uses py::trampoline_self_life_support for safe lifetime management
 */
class PyMCTS_move : public MCTS_move, public py::trampoline_self_life_support {
public:
    using MCTS_move::MCTS_move;

    bool operator==(const MCTS_move& other) const override {
        PYBIND11_OVERRIDE_PURE(
            bool,           /* Return type */
            MCTS_move,      /* Parent class */
            operator==,     /* Name of function in C++ (must match Python name) */
            other           /* Arguments */
        );
    }

    std::string sprint() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string,    /* Return type */
            MCTS_move,      /* Parent class */
            sprint,         /* Name of function in C++ (must match Python name) */
                            /* No arguments for this function */
        );
    }

    std::vector<double> to_numpy() const override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, /* Return type */
            MCTS_move,           /* Parent class */
            to_numpy,            /* Name of function in C++ (must match Python name) */
                                 /* No arguments for this function */
        );
    }

    std::vector<int> to_env_action() const override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<int>,    /* Return type */
            MCTS_move,           /* Parent class */
            to_env_action,       /* Name of function in C++ (must match Python name) */
                                 /* No arguments for this function */
        );
    }
};

/**
 * Trampoline class for MCTS_state to enable Python inheritance
 * Uses py::trampoline_self_life_support for safe lifetime management
 */
class PyMCTS_state : public MCTS_state, public py::trampoline_self_life_support {
public:
    using MCTS_state::MCTS_state;

    ~PyMCTS_state() override = default;

    std::queue<MCTS_move*>* actions_to_try() const override {
        PYBIND11_OVERRIDE_PURE(
            std::queue<MCTS_move*>*,  /* Return type */
            MCTS_state,               /* Parent class */
            actions_to_try,           /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    MCTS_state* next_state(const MCTS_move* move) const override {
        PYBIND11_OVERRIDE_PURE(
            MCTS_state*,              /* Return type */
            MCTS_state,               /* Parent class */
            next_state,               /* Name of function in C++ (must match Python name) */
            move                      /* Arguments */
        );
    }
    
    double rollout() const override {
        PYBIND11_OVERRIDE_PURE(
            double,                   /* Return type */
            MCTS_state,               /* Parent class */
            rollout,                  /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    bool is_terminal() const override {
        PYBIND11_OVERRIDE_PURE(
            bool,                     /* Return type */
            MCTS_state,               /* Parent class */
            is_terminal,              /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    void print() const override {
        PYBIND11_OVERRIDE_PURE(
            void,                     /* Return type */
            MCTS_state,               /* Parent class */
            print,                    /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    bool is_self_side_turn() const override {
        PYBIND11_OVERRIDE_PURE(
            bool,                     /* Return type */
            MCTS_state,               /* Parent class */
            is_self_side_turn,        /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    MCTS_state* clone() const override {
        PYBIND11_OVERRIDE_PURE(
            MCTS_state*,              /* Return type */
            MCTS_state,               /* Parent class */
            clone,                    /* Name of function in C++ (must match Python name) */
                                      /* No arguments */
        );
    }

    std::vector<double> get_action_probabilities() const override {
        PYBIND11_OVERRIDE(
            std::vector<double>,      /* Return type */
            MCTS_state,               /* Parent class */
            get_action_probabilities, /* Name of function in C++ */
                                      /* No arguments */
        );
    }
};

/**
 * Helper function to convert queue<MCTS_move*>* to vector for Python
 * This takes ownership of the queue and all moves in it
 */
std::vector<MCTS_move*> queue_to_vector(std::queue<MCTS_move*>* q);

/**
 * Helper function to convert vector to queue<MCTS_move*>*
 * This creates new queue and transfers ownership of moves
 */
std::queue<MCTS_move*>* vector_to_queue(const std::vector<MCTS_move*>& vec);

/**
 * Safe wrapper for MCTS_agent that handles move ownership
 */
class SafeMCTS_agent {
private:
    MCTS_agent* agent;
    
public:
    SafeMCTS_agent(MCTS_state* starting_state, int max_iter = 100000, int max_seconds = 30);
    ~SafeMCTS_agent();
    
    // Returns nullptr if no move available (game ended)
    const MCTS_move* genmove(const MCTS_move* enemy_move = nullptr);
    const MCTS_state* get_current_state() const;
    void feedback() const;
};

#endif // PY_WRAPPERS_H