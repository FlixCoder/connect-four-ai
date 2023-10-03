//! Builder for the connect four game instance.

use std::sync::Arc;

use crate::{error::Error, player::Player, Board, Game};

/// Builder for the connect four [`Game`].
#[derive(Debug, Default)]
pub struct GameBuilder {
	/// Player for team X, starting player.
	player_x: Option<Arc<dyn Player>>,
	/// Player for team O, second player.
	player_o: Option<Arc<dyn Player>>,
}

impl GameBuilder {
	/// Set the player to play for team X.
	pub fn player_x<P: Player + 'static>(mut self, player: P) -> Self {
		self.player_x = Some(Arc::new(player));
		self
	}

	/// Set the player to play for team O.
	pub fn player_o<P: Player + 'static>(mut self, player: P) -> Self {
		self.player_o = Some(Arc::new(player));
		self
	}

	/// Finalize build of the game.
	pub fn build(self) -> Result<Game, Error> {
		let Some(player_x) = self.player_x else {
			return Err(Error::BuilderMissingField("player_x"));
		};
		let Some(player_o) = self.player_o else {
			return Err(Error::BuilderMissingField("player_o"));
		};

		let board = Board::default();

		Ok(Game { board, player_x, player_o })
	}
}
