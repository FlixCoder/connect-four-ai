//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use std::sync::Arc;

use game::{Error, Game, GameResult};
use players::{IoPlayer, MinimaxPlayer};

fn main() -> Result<(), Error> {
	let mut game = Game::builder()
		.player_x(Arc::new(IoPlayer))
		.player_o(Arc::new(MinimaxPlayer::new_1(7)))
		.build();
	let result = game.run()?;

	match result {
		GameResult::Draw => {
			println!("Good game! That's a draw!");
		}
		GameResult::Winner(winner) => {
			println!("Congratulations {winner:?}, you won!");
		}
	}

	Ok(())
}
