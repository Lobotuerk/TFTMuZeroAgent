import pytest
import math

def test_puct_biased_selection(pymcts_module):
    """
    Test that PUCT correctly uses prior probabilities to bias selection.
    We create a simple state with two moves: 'move_A' and 'move_B'.
    Both moves lead to identical terminal states with score 0.5.
    However, we give 'move_B' a much higher prior probability.
    PUCT should prefer 'move_B' initially.
    """
    
    class SimpleMove(pymcts_module.MCTS_move):
        def __init__(self, name):
            super().__init__()
            self.name = name
        def __eq__(self, other):
            return isinstance(other, SimpleMove) and self.name == other.name
        def sprint(self):
            return self.name
        def to_numpy(self):
            return [1.0] if self.name == "move_A" else [2.0]
        def to_env_action(self):
            return [0] if self.name == "move_A" else [1]

    class BiasedState(pymcts_module.MCTS_state):
        def __init__(self, is_terminal=False):
            super().__init__()
            self._is_terminal = is_terminal
            self._is_self_turn = True

        def actions_to_try(self):
            if self._is_terminal:
                return []
            return [SimpleMove("move_A"), SimpleMove("move_B")]

        def get_action_probabilities(self):
            # Give move_B a 0.9 probability and move_A a 0.1 probability
            return [0.1, 0.9]

        def next_state(self, move):
            return BiasedState(is_terminal=True)

        def rollout(self):
            return 0.5

        def is_terminal(self):
            return self._is_terminal

        def is_self_side_turn(self):
            return self._is_self_turn

    # Force single thread for deterministic-ish behavior
    original_threads = pymcts_module.get_rollout_threads()
    pymcts_module.set_rollout_threads(1)
    
    try:
        state = BiasedState()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Run with few iterations
        # Iteration 1: Expand move_B (it has higher priority in untried_actions)
        # Iteration 2: Expand move_A
        # Iteration 3: Both are expanded. Select based on PUCT.
        # Q(A) = 0.5, Q(B) = 0.5.
        # P(A) = 0.1, P(B) = 0.9.
        # Exploration(A) = c * 0.1 * sqrt(2) / 2
        # Exploration(B) = c * 0.9 * sqrt(2) / 2
        # move_B should be selected again.
        
        agent = pymcts_module.MCTS_agent(wrapped_state, max_iter=10, max_seconds=1)
        move = agent.genmove(None)
        
        assert move is not None
        assert move.sprint() == "move_B"
        
    finally:
        pymcts_module.set_rollout_threads(original_threads)

def test_puct_overcomes_priors(pymcts_module):
    """
    Test that PUCT eventually overcomes a wrong prior if the other move is better.
    'move_A' is a win (1.0), 'move_B' is a loss (0.0).
    'move_B' has a high prior (0.9).
    With enough iterations, MCTS should discover that 'move_A' is better.
    """
    
    class SimpleMove(pymcts_module.MCTS_move):
        def __init__(self, name):
            super().__init__()
            self.name = name
        def __eq__(self, other):
            return isinstance(other, SimpleMove) and self.name == other.name
        def sprint(self):
            return self.name
        def to_numpy(self):
            return [1.0] if self.name == "move_A" else [2.0]
        def to_env_action(self):
            return [0] if self.name == "move_A" else [1]

    class WrongPriorState(pymcts_module.MCTS_state):
        def __init__(self, value=0.5, is_terminal=False):
            super().__init__()
            self.value = value
            self._is_terminal = is_terminal

        def actions_to_try(self):
            if self._is_terminal:
                return []
            return [SimpleMove("move_A"), SimpleMove("move_B")]

        def get_action_probabilities(self):
            return [0.1, 0.9] # move_B looks better to the prior

        def next_state(self, move):
            if move.name == "move_A":
                return WrongPriorState(value=1.0, is_terminal=True)
            else:
                return WrongPriorState(value=0.0, is_terminal=True)

        def rollout(self):
            return self.value

        def is_terminal(self):
            return self._is_terminal

        def is_self_side_turn(self):
            return True

    original_threads = pymcts_module.get_rollout_threads()
    pymcts_module.set_rollout_threads(1)
    
    try:
        state = WrongPriorState()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # With many iterations, it should find move_A
        agent = pymcts_module.MCTS_agent(wrapped_state, max_iter=100, max_seconds=2)
        move = agent.genmove(None)
        
        assert move is not None
        assert move.sprint() == "move_A"
        
    finally:
        pymcts_module.set_rollout_threads(original_threads)

def test_untried_actions_sorted_by_prior(pymcts_module):
    """
    Verify that untried actions are sorted by prior probability in the constructor.
    """
    class SimpleMove(pymcts_module.MCTS_move):
        def __init__(self, name):
            super().__init__()
            self.name = name
        def __eq__(self, other):
            return isinstance(other, SimpleMove) and self.name == other.name
        def sprint(self):
            return self.name
        def to_numpy(self): return [0.0]
        def to_env_action(self): return [0]

    class MultiActionState(pymcts_module.MCTS_state):
        def __init__(self):
            super().__init__()
        def actions_to_try(self):
            return [SimpleMove("low"), SimpleMove("high"), SimpleMove("med")]
        def get_action_probabilities(self):
            return [0.1, 0.8, 0.4]
        def next_state(self, move): return MultiActionState()
        def rollout(self): return 0.5
        def is_terminal(self): return False
        def is_self_side_turn(self): return True

    state = MultiActionState()
    wrapped_state = pymcts_module.SerializedPythonState(state)
    
    # In MCTS_node constructor, it should sort untried actions: high (0.8), med (0.4), low (0.1)
    # The first call to expand() should pick "high".
    
    agent = pymcts_module.MCTS_agent(wrapped_state, max_iter=1, max_seconds=1)
    # genmove(None) will grow tree by 1 iteration.
    # Root will expand "high".
    agent.genmove(None)
    
    # We can't easily inspect the internal tree from Python without more bindings,
    # but we can check if it returns "high" if we only give it 1 iteration and it's the best.
    # Actually genmove returns the best child of the root.
    # If we only have 1 iteration, only one child is expanded. It MUST be the best.
    
    # Wait, if only 1 child is expanded, select_best_child() returns that child.
    # And expand() picks from the front of untried_actions.
    
    # Let's try to verify it by seeing what move was selected.
    # If iterations=1, it expands the first untried action.
    # If sorting works, the first untried action is "high".
    
    # Reset agent
    agent = pymcts_module.MCTS_agent(wrapped_state, max_iter=1, max_seconds=1)
    move = agent.genmove(None)
    assert move.sprint() == "high"
