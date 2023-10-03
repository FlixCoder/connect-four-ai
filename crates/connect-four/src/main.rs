//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use connect_four::IoPlayer;
use game::{Error, Game, GameResult};

fn main() -> Result<(), Error> {
	let game = Game::builder().player_x(IoPlayer).player_o(IoPlayer).build()?;
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
