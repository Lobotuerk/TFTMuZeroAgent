#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <sstream>
#include <thread>
#include "py_wrappers.h"
#include "../mcts/include/state.h"
#include "mcts_python.h"  // Use Python-specific header
#include "../examples/TicTacToe/TicTacToe.h"

namespace py = pybind11;

PYBIND11_MODULE(pymcts, m, py::mod_gil_not_used()) {
    m.doc() = "Python bindings for Monte Carlo Tree Search C++ library with smart_holder support";

    // Abstract base classes with trampolines using py::smart_holder
    py::class_<MCTS_move, PyMCTS_move, py::smart_holder>(m, "MCTS_move")
        .def(py::init<>())
        .def("__eq__", &MCTS_move::operator==)
        .def("sprint", &MCTS_move::sprint, "Get string representation of the move")
        .def("__str__", &MCTS_move::sprint)
        .def("to_numpy", &MCTS_move::to_numpy, "Convert move to numpy array representation")
        .def("to_env_action", &MCTS_move::to_env_action, "Convert move to environment action format");

    py::class_<MCTS_state, PyMCTS_state, py::smart_holder>(m, "MCTS_state")
        .def(py::init<>())
        .def("actions_to_try", [](const MCTS_state& self) {
            // Convert queue to vector for Python
            auto* queue = self.actions_to_try();
            return queue_to_vector(queue);
        }, "Get list of possible moves from this state")
        .def("next_state", &MCTS_state::next_state, 
             "Get the state that results from applying the given move",
             py::return_value_policy::take_ownership)
        .def("rollout", &MCTS_state::rollout, 
             "Perform a random rollout simulation and return win probability for self side")
        .def("is_terminal", &MCTS_state::is_terminal, "Check if this is a terminal state")
        .def("print", &MCTS_state::print, "Print the current state")
        .def("is_self_side_turn", &MCTS_state::is_self_side_turn, "Check if it's the self side's turn")
        .def("clone", &MCTS_state::clone, "Create a deep copy of this state", py::return_value_policy::take_ownership)
        .def("get_action_probabilities", &MCTS_state::get_action_probabilities, "Get prior probabilities for possible moves");

    // Core MCTS classes
    py::class_<MCTS_node>(m, "MCTS_node")
        .def("is_fully_expanded", &MCTS_node::is_fully_expanded, 
             "Check if all possible moves from this node have been tried")
        .def("is_terminal", &MCTS_node::is_terminal, "Check if this node represents a terminal state")
        .def("get_move", &MCTS_node::get_move, 
             "Get the move that led to this node", py::return_value_policy::reference)
        .def("get_size", &MCTS_node::get_size, "Get the number of nodes in the subtree")
        .def_property_readonly("prior_probability", &MCTS_node::get_prior_probability, "Get the prior probability for PUCT")
        .def("expand", &MCTS_node::expand, "Expand this node by adding a new child")
        .def("rollout", &MCTS_node::rollout, "Perform a rollout simulation from this node")
        .def("select_best_child", &MCTS_node::select_best_child, 
             "Select the best child using UCT", py::arg("c"))
        .def("get_current_state", &MCTS_node::get_current_state, 
             "Get the game state represented by this node", py::return_value_policy::reference)
        .def("print_stats", &MCTS_node::print_stats, "Print statistics about this node")
        .def("calculate_winrate", &MCTS_node::calculate_winrate, 
             "Calculate win rate for the specified side", py::arg("self_side_turn"));

    py::class_<MCTS_tree>(m, "MCTS_tree")
        .def(py::init<MCTS_state*>(), "Create a new MCTS tree with the given starting state",
             py::arg("starting_state"))
        .def("select", &MCTS_tree::select, 
             "Select a node to expand using UCT", py::arg("c") = 1.41)
        .def("select_best_child", &MCTS_tree::select_best_child, 
             "Select the best child of the root node")
        .def("grow_tree", &MCTS_tree::grow_tree, 
             "Grow the tree for the specified iterations or time",
             py::arg("max_iter"), py::arg("max_time_in_seconds"))
        .def("advance_tree", &MCTS_tree::advance_tree, 
             "Advance the tree by applying the given move", py::arg("move"))
        .def("get_size", &MCTS_tree::get_size, "Get the total number of nodes in the tree")
        .def("get_current_state", &MCTS_tree::get_current_state, 
             "Get the current root state", py::return_value_policy::reference)
        .def("print_stats", &MCTS_tree::print_stats, "Print tree statistics");

    // High-level agent interface (recommended for most users)
    py::class_<SafeMCTS_agent>(m, "MCTS_agent")
        .def(py::init<MCTS_state*, int, int>(), 
             "Create an MCTS agent with the given starting state and parameters",
             py::arg("starting_state"), py::arg("max_iter") = 100000, py::arg("max_seconds") = 30)
        .def("genmove", &SafeMCTS_agent::genmove, 
             "Generate the next move, optionally considering an enemy move first",
             py::arg("enemy_move") = nullptr, py::return_value_policy::reference)
        .def("get_current_state", &SafeMCTS_agent::get_current_state, 
             "Get the current game state", py::return_value_policy::reference)
        .def("feedback", &SafeMCTS_agent::feedback, "Print feedback about the agent's thinking");

    // TicTacToe example implementation with py::smart_holder
    py::class_<TicTacToe_move, MCTS_move, py::smart_holder>(m, "TicTacToe_move")
        .def(py::init<int, int, char>(), 
             "Create a TicTacToe move", py::arg("x"), py::arg("y"), py::arg("player"))
        .def_readwrite("x", &TicTacToe_move::x, "X coordinate (0-2)")
        .def_readwrite("y", &TicTacToe_move::y, "Y coordinate (0-2)")
        .def_readwrite("player", &TicTacToe_move::player, "Player ('x' or 'o')")
        .def("__eq__", &TicTacToe_move::operator==)
        .def("__str__", [](const TicTacToe_move& move) {
            return "TicTacToe_move(" + std::to_string(move.x) + ", " + 
                   std::to_string(move.y) + ", '" + move.player + "')";
        });

    py::class_<TicTacToe_state, MCTS_state, py::smart_holder>(m, "TicTacToe_state")
        .def(py::init<>(), "Create a new TicTacToe game state")
        .def(py::init<const TicTacToe_state&>(), "Copy constructor")
        .def("get_turn", &TicTacToe_state::get_turn, "Get whose turn it is ('x' or 'o')")
        .def("get_winner", &TicTacToe_state::get_winner, 
             "Get the winner ('x', 'o', 'd' for draw, or ' ' for ongoing)")
        .def("actions_to_try", [](const TicTacToe_state& self) {
            auto* queue = self.actions_to_try();
            return queue_to_vector(queue);
        }, "Get list of possible moves")
        .def("next_state", &TicTacToe_state::next_state, 
             "Get state after applying move", py::return_value_policy::take_ownership)
        .def("rollout", &TicTacToe_state::rollout, "Perform random rollout simulation")
        .def("is_terminal", &TicTacToe_state::is_terminal, "Check if game is finished")
        .def("print", &TicTacToe_state::print, "Print the board")
        .def("is_self_side_turn", &TicTacToe_state::is_self_side_turn, "Check if it's the self side's turn")
        .def("clone", &TicTacToe_state::clone, "Create a deep copy of this state", py::return_value_policy::take_ownership)
        .def("__str__", [](const TicTacToe_state& state) {
            // Capture print output for Python string representation
            std::ostringstream oss;
            std::streambuf* old_cout = std::cout.rdbuf(oss.rdbuf());
            state.print();
            std::cout.rdbuf(old_cout);
            return oss.str();
        });

    // Create an alias for easy access to C++ TicTacToe state
    m.def("cpp_TicTacToeState", []() {
        return new TicTacToe_state();
    }, "Create a C++ TicTacToe state instance", py::return_value_policy::take_ownership);

    // Utility functions
    m.def("queue_to_vector", &queue_to_vector, 
          "Convert a queue of moves to a vector (for internal use)");
    m.def("vector_to_queue", &vector_to_queue, 
          "Convert a vector of moves to a queue (for internal use)");
    
    // Python state wrapper for seamless Python game integration
    py::class_<SerializedPythonState, MCTS_state, py::smart_holder>(m, "SerializedPythonState")
        .def(py::init<py::object>(), "Wrap a Python game state object for C++ MCTS",
             py::arg("python_state"));
    
    // Thread configuration functions
    m.def("set_rollout_threads", [](unsigned int num_threads) {
        MCTS_node::set_rollout_threads(num_threads);
    }, "Set the global number of parallel rollout threads", py::arg("num_threads"));
    
    m.def("get_rollout_threads", []() {
        return MCTS_node::get_rollout_threads();
    }, "Get the current number of parallel rollout threads");
    
    m.def("get_optimal_thread_count", []() {
        return ParallelRollouts::get_optimal_thread_count();
    }, "Get the optimal number of threads based on hardware");
    
    m.def("get_hardware_concurrency", []() {
        return std::thread::hardware_concurrency();
    }, "Get the number of concurrent threads supported by the hardware");
}