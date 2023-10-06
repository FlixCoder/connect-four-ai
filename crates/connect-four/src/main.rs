//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use game::{Game, GameResult};
use players::{IoPlayer, ModelPlayer, NdArrayBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let ai = ModelPlayer::<NdArrayBackend>::init().load("model").unwrap_or_else(|err| {
		println!("Failed loading model: {err}");
		println!("Starting with new model");
		ModelPlayer::init()
	});

	let mut game = Game::builder().player_x(&IoPlayer).player_o(&ai).build();
	let result = game.run()?;

	println!("{}", game.board());
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
