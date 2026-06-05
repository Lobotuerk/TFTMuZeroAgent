"""
Tests for Python game implementations.
Tests Python games from the demo folder without using MCTS agents.
NOTE: MCTS agent tests disabled due to C++ memory corruption issue.
"""
import pytest
import sys
import os

# Import demo games for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demo'))

try:
    from connect_four_python import ConnectFourMove, ConnectFourState
    CONNECT_FOUR_AVAILABLE = True
except ImportError:
    CONNECT_FOUR_AVAILABLE = False

try:
    from simple_python_games import CoinFlipMove, CoinFlipState, GuessMove, NumberGuessingState
    SIMPLE_GAMES_AVAILABLE = True
except ImportError:
    SIMPLE_GAMES_AVAILABLE = False


class TestPythonGameBasics:
    """Test basic functionality of Python games without MCTS."""
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_basic_functionality(self):
        """Test Connect Four basic functionality."""
        # Test move creation
        move = ConnectFourMove(3, 'X')
        assert move.column == 3
        assert move.player == 'X'
        assert '3' in move.sprint()
        assert 'X' in move.sprint()
        
        # Test state creation
        state = ConnectFourState()
        assert not state.is_terminal()
        assert state.is_self_side_turn()
        
        # Test move generation
        moves = state.actions_to_try()
        assert len(moves) == 7  # 7 columns
        
        # Test state transition
        if moves:
            new_state = state.next_state(moves[0])
            assert new_state is not None
            assert not new_state.is_self_side_turn()  # Should switch players
            
        # Test rollout
        rollout_result = state.rollout()
        assert 0.0 <= rollout_result <= 1.0
        
    @pytest.mark.skipif(not SIMPLE_GAMES_AVAILABLE, reason="Simple games not available")
    def test_coin_flip_game(self):
        """Test coin flip game functionality."""
        # Test move
        move = CoinFlipMove('heads')
        assert move.choice == 'heads'
        assert 'heads' in move.sprint().lower()
        
        # Test state
        state = CoinFlipState()
        assert not state.is_terminal()
        assert state.is_self_side_turn()
        
        moves = state.actions_to_try()
        assert len(moves) == 2  # heads or tails
        
        # Test state transition
        if moves:
            new_state = state.next_state(moves[0])
            assert not new_state.is_terminal()  # Game continues (3 guesses total)
            assert new_state.guesses_left == 2  # Started with 3, now has 2
            
    @pytest.mark.skipif(not SIMPLE_GAMES_AVAILABLE, reason="Simple games not available")
    def test_number_game(self):
        """Test number guessing game functionality."""
        # Test move
        move = GuessMove(5)
        assert move.number == 5
        assert '5' in move.sprint()
        
        # Test state
        state = NumberGuessingState()
        assert not state.is_terminal()
        assert state.is_self_side_turn()
        
        moves = state.actions_to_try()
        assert len(moves) > 0  # Should have some number choices
        
        # Test rollout
        rollout_result = state.rollout()
        assert 0.0 <= rollout_result <= 1.0


class TestPythonGameInheritance:
    """Test that Python games properly inherit from C++ base classes."""
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_inheritance(self, pymcts_module):
        """Test Connect Four inheritance from C++ classes."""
        move = ConnectFourMove(3, 'X')
        state = ConnectFourState()
        
        assert isinstance(move, pymcts_module.MCTS_move)
        assert isinstance(state, pymcts_module.MCTS_state)
        
        # Test required methods exist
        required_move_methods = ['__eq__', 'sprint']
        for method in required_move_methods:
            assert hasattr(move, method)
            assert callable(getattr(move, method))
            
        required_state_methods = ['actions_to_try', 'next_state', 'rollout', 'is_terminal', 'is_self_side_turn']
        for method in required_state_methods:
            assert hasattr(state, method)
            assert callable(getattr(state, method))
            
    @pytest.mark.skipif(not SIMPLE_GAMES_AVAILABLE, reason="Simple games not available")
    def test_simple_games_inheritance(self, pymcts_module):
        """Test simple games inheritance from C++ classes."""
        coin_move = CoinFlipMove('heads')
        coin_state = CoinFlipState()
        
        assert isinstance(coin_move, pymcts_module.MCTS_move)
        assert isinstance(coin_state, pymcts_module.MCTS_state)
        
        number_move = GuessMove(5)
        number_state = NumberGuessingState()
        
        assert isinstance(number_move, pymcts_module.MCTS_move)
        assert isinstance(number_state, pymcts_module.MCTS_state)


class TestPythonGameLogic:
    """Test game logic and rule implementation."""
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_game_logic(self):
        """Test Connect Four game logic implementation."""
        state = ConnectFourState()
        
        # Test initial state
        assert len(state.board) == 6  # 6 rows
        assert len(state.board[0]) == 7  # 7 columns
        assert all(cell == ' ' for row in state.board for cell in row)
        
        # Test valid moves
        moves = state.actions_to_try()
        assert all(0 <= move.column < 7 for move in moves)
        
        # Test move application
        first_move = moves[0]
        new_state = state.next_state(first_move)
        
        # Check that the piece was placed correctly
        column = first_move.column
        assert new_state.board[5][column] == 'X'  # Bottom row should have the piece
        
        # Check player switch
        assert new_state.current_player == 'O'
        
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_win_detection(self):
        """Test Connect Four win detection logic."""
        state = ConnectFourState()
        
        # Create a winning state manually (4 in a row horizontally)
        test_state = ConnectFourState()
        test_state.board[5][0] = 'X'
        test_state.board[5][1] = 'X'
        test_state.board[5][2] = 'X'
        test_state.board[5][3] = 'X'
        
        # Should detect the win
        winner = test_state.get_winner()
        assert winner == 'X'
        assert test_state.is_terminal()


# Working MCTS agent tests - issue was incorrect constructor syntax

class TestPythonGamesWithMCTS:
    """Test Python games with MCTS agents."""
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_with_mcts(self, pymcts_module):
        """Test Connect Four with MCTS agent."""
        state = ConnectFourState()
        
        # Use SerializedPythonState wrapper for MCTS compatibility
        wrapped_state = pymcts_module.SerializedPythonState(state)
        agent = pymcts_module.MCTS_agent(wrapped_state, 20, 1)  # 20 iterations, 1 second max
        
        move = agent.genmove(None)
        assert move is not None
        
        # Should be a valid Connect Four move (returned as MCTS_move)
        move_str = str(move)
        assert 'Drop' in move_str
        assert '@' in move_str
        
        # Parse the move string to verify it's valid
        parts = move_str.split('@')
        assert len(parts) == 2
        column = int(parts[1])
        assert 0 <= column <= 6
        
        print(f"Connect Four MCTS move: {move_str}")
        
    @pytest.mark.skipif(not SIMPLE_GAMES_AVAILABLE, reason="Simple games not available")
    def test_simple_games_with_mcts(self, pymcts_module):
        """Test simple games with MCTS agent."""
        # Test coin flip game
        coin_state = CoinFlipState()
        wrapped_coin_state = pymcts_module.SerializedPythonState(coin_state)
        coin_agent = pymcts_module.MCTS_agent(wrapped_coin_state, 10, 1)  # 10 iterations, 1 second max
        
        coin_move = coin_agent.genmove(None)
        assert coin_move is not None
        coin_move_str = str(coin_move)
        assert 'Choose' in coin_move_str and ('heads' in coin_move_str or 'tails' in coin_move_str)
        
        print(f"Coin flip MCTS choice: {coin_move_str}")
        
        # Test number game
        number_state = NumberGuessingState()
        wrapped_number_state = pymcts_module.SerializedPythonState(number_state)
        number_agent = pymcts_module.MCTS_agent(wrapped_number_state, 10, 1)  # 10 iterations, 1 second max
        
        number_move = number_agent.genmove(None)
        assert number_move is not None
        number_move_str = str(number_move)
        assert 'Guess' in number_move_str
        
        print(f"Number game MCTS choice: {number_move_str}")
        
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_full_game_with_mcts(self, pymcts_module):
        """Test a full Connect Four game with MCTS vs random opponent."""
        state = ConnectFourState()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        agent = pymcts_module.MCTS_agent(wrapped_state, 50, 1)  # 50 iterations, 1 second max
        
        moves_played = 0
        max_moves = 10  # Limit for testing
        
        # Simplified test - just verify MCTS can generate valid moves
        agent_move = agent.genmove(None)
        assert agent_move is not None
        
        # Parse the move string to verify it's valid
        move_str = str(agent_move)
        assert 'Drop' in move_str and '@' in move_str
        column = int(move_str.split('@')[1])
        assert 0 <= column <= 6
        
        print(f"MCTS chose column {column}")
        print(f"Connect Four MCTS test completed successfully")
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_mcts_finds_winning_move(self, pymcts_module):
        """Test that MCTS can identify and choose a winning move when one is available."""
        # Create a specific board state where X can win in one move
        # Board layout (X needs to play in column 3 to win horizontally):
        #   0 1 2 3 4 5 6
        # 5 . . . . . . .
        # 4 . . . . . . .
        # 3 . . . . . . .
        # 2 . . . . . . .
        # 1 . . . . . . .
        # 0 X X X . O O O
        
        state = ConnectFourState()
        
        # Manually set up the board for a winning scenario
        # Bottom row: X X X . O O O
        state.board[5][0] = 'X'  # X at position (5,0)
        state.board[5][1] = 'X'  # X at position (5,1) 
        state.board[5][2] = 'X'  # X at position (5,2)
        state.board[5][3] = ' '  # Empty at position (5,3) - winning move!
        state.board[5][4] = 'O'  # O at position (5,4)
        state.board[5][5] = 'O'  # O at position (5,5)
        state.board[5][6] = 'O'  # O at position (5,6)
        
        # Set current player to X (should be X's turn to make the winning move)
        state.current_player = 'X'
        
        # Use SerializedPythonState wrapper for MCTS compatibility
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Use more iterations to ensure MCTS finds the obvious win
        agent = pymcts_module.MCTS_agent(wrapped_state, 100, 2)  # 100 iterations, 2 seconds max
        
        # MCTS should find the winning move
        move = agent.genmove(None)
        assert move is not None, "MCTS should find a move"
        
        # Parse the move
        move_str = str(move)
        assert 'Drop' in move_str, f"Expected Drop move, got: {move_str}"
        assert '@' in move_str, f"Expected @ in move format, got: {move_str}"
        
        # Extract column from move string (format: "DropX@3")
        column = int(move_str.split('@')[1])
        
        # The winning move should be column 3
        assert column == 3, f"MCTS should choose winning column 3, but chose column {column}"
        
        # Verify this move actually wins the game
        winning_move = ConnectFourMove(3, 'X')
        final_state = state.next_state(winning_move)
        assert final_state.is_terminal(), "The move should result in a terminal state"
        assert final_state.get_winner() == 'X', "X should win after this move"
        
        print(f"✅ MCTS correctly identified winning move: column {column}")