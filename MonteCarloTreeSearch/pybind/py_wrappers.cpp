#include "py_wrappers.h"
#include <iostream>
#include <stdexcept>

// SerializedPythonState implementation
SerializedPythonState::SerializedPythonState(py::object python_state) 
    : python_state(python_state) {
    // Simple approach: just store the Python object directly
    // C++ owns this object and will manage its lifetime
}

std::queue<MCTS_move*>* SerializedPythonState::actions_to_try() const {
    try {
        py::list py_moves = python_state.attr("actions_to_try")();
        
        // Clear previous cache and rebuild it
        cached_python_moves.clear();
        
        std::queue<MCTS_move*>* queue = new std::queue<MCTS_move*>();
        for (auto item : py_moves) {
            // Create C++ wrapper for each Python move
            PythonMoveWrapper* cpp_move = new PythonMoveWrapper(py::reinterpret_borrow<py::object>(item));
            queue->push(cpp_move);
            // Cache the Python object to keep it alive
            cached_python_moves.push_back(py::reinterpret_borrow<py::object>(item));
        }
        return queue;
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::actions_to_try: " << e.what() << std::endl;
        return new std::queue<MCTS_move*>();
    }
}

MCTS_state* SerializedPythonState::next_state(const MCTS_move* move) const {
    try {
        // Extract the Python move from the wrapper
        const PythonMoveWrapper* wrapper = dynamic_cast<const PythonMoveWrapper*>(move);
        py::object py_move;
        
        if (wrapper) {
            py_move = wrapper->get_python_move();
        } else {
            // Fallback for non-wrapper moves (shouldn't happen in normal usage)
            py_move = py::cast(move, py::return_value_policy::reference);
        }
        
        py::object new_py_state = python_state.attr("next_state")(py_move);
        return new SerializedPythonState(new_py_state);
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::next_state: " << e.what() << std::endl;
        return new SerializedPythonState(python_state);
    }
}

double SerializedPythonState::rollout() const {
    try {
        return python_state.attr("rollout")().cast<double>();
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::rollout: " << e.what() << std::endl;
        return 0.5;
    }
}

bool SerializedPythonState::is_terminal() const {
    try {
        return python_state.attr("is_terminal")().cast<bool>();
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::is_terminal: " << e.what() << std::endl;
        return true;
    }
}

void SerializedPythonState::print() const {
    try {
        python_state.attr("print")();
    } catch (const std::exception& e) {
        std::cout << "SerializedPythonState (print error: " << e.what() << ")" << std::endl;
    }
}

bool SerializedPythonState::is_self_side_turn() const {
    try {
        return python_state.attr("is_self_side_turn")().cast<bool>();
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::is_self_side_turn: " << e.what() << std::endl;
        return true;
    }
}

MCTS_state* SerializedPythonState::clone() const {
    try {
        py::object cloned = python_state.attr("clone")();
        return new SerializedPythonState(cloned);
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::clone: " << e.what() << std::endl;
        return new SerializedPythonState(python_state);
    }
}

std::vector<double> SerializedPythonState::get_action_probabilities() const {
    try {
        if (py::hasattr(python_state, "get_action_probabilities")) {
            py::list py_probs = python_state.attr("get_action_probabilities")();
            std::vector<double> result;
            for (auto item : py_probs) {
                result.push_back(item.cast<double>());
            }
            return result;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in SerializedPythonState::get_action_probabilities: " << e.what() << std::endl;
    }
    return std::vector<double>();
}

py::object SerializedPythonState::find_python_move(const MCTS_move* cpp_move) const {
    // Search through cached Python moves to find the one that matches using value comparison
    for (const auto& py_move : cached_python_moves) {
        try {
            MCTS_move* cached_cpp_move = py_move.cast<MCTS_move*>();
            // Use the move's operator== for value comparison instead of pointer comparison
            if (cached_cpp_move && cpp_move && *cached_cpp_move == *cpp_move) {
                return py_move;
            }
        } catch (const std::exception& e) {
            // Skip this move if casting fails
            continue;
        }
    }
    
    // Fallback: try to convert the C++ pointer directly (risky but better than crashing)
    std::cerr << "Warning: Could not find cached Python move, attempting direct conversion" << std::endl;
    return py::cast(cpp_move, py::return_value_policy::reference);
}

std::vector<MCTS_move*> queue_to_vector(std::queue<MCTS_move*>* q) {
    std::vector<MCTS_move*> result;
    if (q == nullptr) {
        return result;
    }
    
    while (!q->empty()) {
        result.push_back(q->front());
        q->pop();
    }
    delete q;  // Clean up the queue
    return result;
}

std::queue<MCTS_move*>* vector_to_queue(const std::vector<MCTS_move*>& vec) {
    auto* q = new std::queue<MCTS_move*>();
    for (MCTS_move* move : vec) {
        q->push(move);
    }
    return q;
}

SafeMCTS_agent::SafeMCTS_agent(MCTS_state* starting_state, int max_iter, int max_seconds) {
    agent = new MCTS_agent(starting_state, max_iter, max_seconds);
}

SafeMCTS_agent::~SafeMCTS_agent() {
    delete agent;
}

const MCTS_move* SafeMCTS_agent::genmove(const MCTS_move* enemy_move) {
    return agent->genmove(enemy_move);
}

const MCTS_state* SafeMCTS_agent::get_current_state() const {
    return agent->get_current_state();
}

void SafeMCTS_agent::feedback() const {
    agent->feedback();
}