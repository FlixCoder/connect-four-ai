//! Run games between players for flamegraphing purposes.

use game::Game;
use players::{MinimaxPlayer, RandomPlayer};

fn main() {
	let player_x = MinimaxPlayer::new_1(5);
	let player_o = RandomPlayer;
	for _ in 0..1_000 {
		let mut game = Game::builder().player_x(&player_x).player_o(&player_o).build();
		game.run_error_loss();
	}
}
