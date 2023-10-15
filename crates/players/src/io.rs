//! Terminal IO player.
#![allow(clippy::print_stdout)]

use std::io::{IsTerminal, Write};

use game::{Board, Player, Team};

/// Terminal IO player.
#[derive(Debug)]
pub struct IoPlayer;

impl Player for IoPlayer {
	fn make_move(&self, board: &Board, me: Team) -> usize {
		if std::io::stdout().is_terminal() {
			println!("Current board:\n{}", board.colored_string(me));
		} else {
			println!("Current board:\n{board}");
		}
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
