//! Random player.

use game::{Board, Player, Team};
use rand::{seq::IteratorRandom, thread_rng};

/// Random player.
#[derive(Debug)]
pub struct RandomPlayer;

impl Player for RandomPlayer {
	fn make_move(&self, board: &Board, _me: Team) -> usize {
		let mut rng = thread_rng();
		let possible_moves = board.possible_moves();
		*possible_moves.iter().choose(&mut rng).expect("No possible moves")
	}
}
