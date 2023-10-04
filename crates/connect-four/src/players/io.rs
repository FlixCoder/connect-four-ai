//! Terminal IO player.

use std::io::Write;

use game::{Board, Player};

/// Terminal IO player.
#[derive(Debug)]
pub struct IoPlayer;

impl Player for IoPlayer {
	fn make_move(&self, board: &Board) -> usize {
		println!("Current board:\n{board}");
		println!("0 | 1 | 2 | 3 | 4 | 5 | 6 \n");

		let possible_moves = board.possible_moves();
		loop {
			print!("Enter number column to place tile in: ");
			std::io::stdout().flush().expect("flush STDOUT");

			let mut input = String::new();
			std::io::stdin().read_line(&mut input).expect("read STDIO");
			match input.trim().parse::<usize>() {
				Ok(column) if possible_moves.contains(&column) => break column,
				_ => {
					println!("Invalid move, try again!");
				}
			}
		}
	}
}
