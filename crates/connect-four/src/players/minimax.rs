//! Minimax player implementation using heuristics and recursive min-maxing.

use std::fmt::Debug;

use game::{Board, GameResult, Player, Team};

/// Type for heuristic function.
type HeuristicFn = Box<dyn Fn(&Board, Team) -> f64>;

/// Minimax player with a custom heuristic.
pub struct MinimaxPlayer {
	/// Deepness to do minimax search to.
	deepness: usize,
	/// Heuristic function to compute the value of board positions. 0.0 should
	/// be a draw, anything above is winning, below zero is losing position. The
	/// strength of is shown by the absolute number.
	heuristic: HeuristicFn,
}

impl MinimaxPlayer {
	/// Create new minimax player with custom heuristic.
	fn new<H>(deepness: usize, heuristic: H) -> Self
	where
		H: Fn(&Board, Team) -> f64 + 'static,
	{
		Self { deepness, heuristic: Box::new(heuristic) }
	}

	/// Create new minimax player with heuristic 1.
	pub fn new_1(deepness: usize) -> Self {
		Self::new(deepness, Board::heuristic_1)
	}

	/// Our turn, take the best value out of our turns.
	fn max_value(&self, board: &Board, me: Team, current_deepness: usize) -> f64 {
		match board.game_result() {
			Some(GameResult::Draw) => return 0.0,
			Some(GameResult::Winner(team)) => return if team == me { f64::MAX } else { f64::MIN },
			None => {}
		}

		if current_deepness + 1 < self.deepness {
			board
				.possible_moves()
				.into_iter()
				.map(|column| {
					let mut test_board = *board;
					test_board.put_tile(column, me).expect("Possible move was in fact impossible");
					self.min_value(&test_board, me, current_deepness + 1)
				})
				.max_by(|val_a, val_b| {
					val_a.partial_cmp(val_b).expect("Heuristic value comparison failed")
				})
				.expect("No possible moves")
		} else {
			(self.heuristic)(board, me)
		}
	}

	/// Other player's turn, minimize the heuristic value to take the other
	/// player's best turn into account.
	fn min_value(&self, board: &Board, me: Team, current_deepness: usize) -> f64 {
		match board.game_result() {
			Some(GameResult::Draw) => return 0.0,
			Some(GameResult::Winner(team)) => return if team == me { f64::MAX } else { f64::MIN },
			None => {}
		}

		if current_deepness + 1 < self.deepness {
			board
				.possible_moves()
				.into_iter()
				.map(|column| {
					let mut test_board = *board;
					test_board
						.put_tile(column, me.other())
						.expect("Possible move was in fact impossible");
					self.max_value(&test_board, me, current_deepness + 1)
				})
				.min_by(|val_a, val_b| {
					val_a.partial_cmp(val_b).expect("Heuristic value comparison failed")
				})
				.expect("No possible moves")
		} else {
			(self.heuristic)(board, me)
		}
	}
}

impl Player for MinimaxPlayer {
	fn make_move(&self, board: &Board, me: Team) -> usize {
		board
			.possible_moves()
			.into_iter()
			.map(|column| {
				let mut test_board = *board;
				test_board.put_tile(column, me).expect("Possible move was in fact impossible");
				let value = self.min_value(&test_board, me, 1);
				(column, value)
			})
			.max_by(|(_, value_a), (_, value_b)| {
				value_a.partial_cmp(value_b).expect("Heuristic value comparison failed")
			})
			.expect("No possible move")
			.0
	}
}

impl Debug for MinimaxPlayer {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("MinimaxPlayer")
			.field("deepness", &self.deepness)
			.field("heuristic", &"<fn>")
			.finish()
	}
}
