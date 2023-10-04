//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use connect_four::players::{IoPlayer, RandomPlayer};
use game::{Error, Game, GameResult};

fn main() -> Result<(), Error> {
	let mut game = Game::builder().player_x(IoPlayer).player_o(RandomPlayer).build()?;
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
