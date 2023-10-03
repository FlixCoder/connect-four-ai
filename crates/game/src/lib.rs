//! Implementation of the connect four game, making it performant and simple to
//! simulate or run games.

mod board;
mod builder;
mod error;
mod player;

use std::sync::Arc;

use self::builder::GameBuilder;
pub use self::{
	board::{Board, GameResult, Team},
	error::Error,
	player::Player,
};

/// An instance of a connect four game.
#[derive(Debug, Clone)]
pub struct Game {
	/// Game board.
	board: Board,
	/// Player for team X, starting player.
	player_x: Arc<dyn Player>,
	/// Player for team O, second player.
	player_o: Arc<dyn Player>,
}

impl Game {
	/// Start building a game.
	#[must_use]
	pub fn builder() -> GameBuilder {
		GameBuilder::default()
	}

	/// Run the game to completion using the players as actors. Returns the game
	/// result.
	pub fn run(mut self) -> Result<GameResult, Error> {
		loop {
			let move_x = self.player_x.make_move(&self.board);
			self.board.put_tile(move_x, Team::X)?;

			if let Some(result) = self.board.game_result() {
				return Ok(result);
			}

			let move_o = self.player_o.make_move(&self.board);
			self.board.put_tile(move_o, Team::O)?;

			if let Some(result) = self.board.game_result() {
				return Ok(result);
			}
		}
	}
}
