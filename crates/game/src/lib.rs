//! Implementation of the connect four game, making it performant and simple to
//! simulate or run games.

mod board;
mod error;
mod player;

pub use self::{
	board::{Board, GameResult, Team},
	error::Error,
	player::Player,
};

/// An instance of a connect four game.
#[derive(Debug, Clone, typed_builder::TypedBuilder)]
pub struct Game<'a> {
	/// Game board.
	#[builder(setter(skip), default)]
	board: Board,
	/// Player for team X, starting player.
	player_x: &'a dyn Player,
	/// Player for team O, second player.
	player_o: &'a dyn Player,
}

impl<'a> Game<'a> {
	/// Return the current board position.
	#[must_use]
	pub fn board(&self) -> &Board {
		&self.board
	}

	/// Run the game to completion using the players as actors. Returns the game
	/// result.
	pub fn run(&mut self) -> Result<GameResult, Error> {
		loop {
			let move_x = self.player_x.make_move(&self.board, Team::X);
			self.board.put_tile(move_x, Team::X)?;

			if let Some(result) = self.board.game_result_on_change(move_x) {
				return Ok(result);
			}

			let move_o = self.player_o.make_move(&self.board, Team::O);
			self.board.put_tile(move_o, Team::O)?;

			if let Some(result) = self.board.game_result_on_change(move_o) {
				return Ok(result);
			}
		}
	}

	/// Run the game with conversion of player errors to game loss.
	pub fn run_error_loss(&mut self) -> GameResult {
		loop {
			let move_x = self.player_x.make_move(&self.board, Team::X);
			match self.board.put_tile(move_x, Team::X) {
				Err(Error::FieldFullAtColumn(team)) => return GameResult::Winner(team.other()),
				Err(err) => panic!("Player made non-game related error: {err}"),
				Ok(_) => {}
			}

			if let Some(result) = self.board.game_result_on_change(move_x) {
				return result;
			}

			let move_o = self.player_o.make_move(&self.board, Team::O);
			match self.board.put_tile(move_o, Team::O) {
				Err(Error::FieldFullAtColumn(team)) => return GameResult::Winner(team.other()),
				Err(err) => panic!("Player made non-game related error: {err}"),
				Ok(_) => {}
			}

			if let Some(result) = self.board.game_result_on_change(move_o) {
				return result;
			}
		}
	}
}
