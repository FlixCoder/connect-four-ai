//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use game::{Game, GameResult};
use players::{IoPlayer, ModelPlayer, WgpuBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let ai = ModelPlayer::<WgpuBackend>::init().load("model")?;

	let mut game = Game::builder().player_x(&IoPlayer).player_o(&ai).build();
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
