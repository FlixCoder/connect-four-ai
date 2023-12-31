//! Connect four CLI game implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

use game::{Error, Game, GameResult, Team};
use players::{AiValuePlayer, IoPlayer, NdArrayBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_path = "./model";
	let ai = AiValuePlayer::<NdArrayBackend>::init(5).load(model_path).unwrap_or_else(|err| {
		println!("Failed loading model: {err}");
		println!("Starting with new model");
		AiValuePlayer::init(5)
	});

	let mut game = Game::builder().player_x(&IoPlayer).player_o(&ai).build();
	let result = match game.run() {
		Ok(res) => res,
		Err(Error::FieldFullAtColumn(team)) => {
			println!("Player {team:?} made a mistake! Player {:?} won!", team.other());
			return Ok(());
		}
		r => r?,
	};

	println!("{}", game.board().colored_string(Team::X));
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
