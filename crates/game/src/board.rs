//! Connect four game board implementation.

use std::{collections::HashSet, fmt::Display};

use crate::Error;

/// Width of the connect four field. Must fit in a u8.
const W: usize = 7;
/// Height of the connect four field. Must fit in a u8.
const H: usize = 6;

/// Connect four game board instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
	/// The field to play on. It is a WxH (columns x rows) field organized in a
	/// 1D array.
	///
	/// Unlike one might assume, the way to address a field is as follows:
	/// `field[x][y] = field[x*H + y]` (not `y*W + x`)
	/// This should allow faster iteration when placing new tiles.
	///
	/// The first tile is put to y = 0, the last to y = H - 1.
	field: [Option<Team>; W * H],
}

/// Team identifiers, X and O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
	/// Team X.
	X,
	/// Team O.
	O,
}

/// Game result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameResult {
	/// There is a draw.
	Draw,
	/// There is a winner.
	Winner(Team),
}

impl Default for Board {
	/// Make new empty board.
	fn default() -> Self {
		Self { field: [None; W * H] }
	}
}

impl Board {
	/// Get the dimensions of the board. Returns (Widht, Height).
	#[must_use]
	pub fn dimensions(&self) -> (usize, usize) {
		(W, H)
	}

	/// Get access to the raw underlying board data.
	#[must_use]
	pub fn field(&self) -> &[Option<Team>] {
		&self.field
	}

	/// Get current state of the board, returning whether there is a result and
	/// if so, who won. Returns unpredictable results if there are multiple
	/// winners at once, as this can only happen when multiple turns are done
	/// without checking the state in between.
	#[must_use]
	// TODO: Benchmark?
	pub fn game_result(&self) -> Option<GameResult> {
		// First check in the y direction as it should be the fastest.
		for x in 0..W {
			for y in 0..H - 3 {
				if let Some(team) = self.field[x * H + y] {
					if self.field[x * H + y + 1] == Some(team)
						&& self.field[x * H + y + 2] == Some(team)
						&& self.field[x * H + y + 3] == Some(team)
					{
						return Some(GameResult::Winner(team));
					}
				}
			}
		}

		// Next check in x direction.
		for y in 0..H {
			for x in 0..W - 3 {
				if let Some(team) = self.field[x * H + y] {
					if self.field[(x + 1) * H + y] == Some(team)
						&& self.field[(x + 2) * H + y] == Some(team)
						&& self.field[(x + 3) * H + y] == Some(team)
					{
						return Some(GameResult::Winner(team));
					}
				}
			}
		}

		// Next check diagonally upwards.
		for x in 0..W - 3 {
			for y in 0..H - 3 {
				if let Some(team) = self.field[x * H + y] {
					if self.field[(x + 1) * H + y + 1] == Some(team)
						&& self.field[(x + 2) * H + y + 2] == Some(team)
						&& self.field[(x + 3) * H + y + 3] == Some(team)
					{
						return Some(GameResult::Winner(team));
					}
				}
			}
		}

		// Finally check diagonally downwards.
		for x in 3..W {
			for y in 0..H - 3 {
				if let Some(team) = self.field[x * H + y] {
					if self.field[(x - 1) * H + y + 1] == Some(team)
						&& self.field[(x - 2) * H + y + 2] == Some(team)
						&& self.field[(x - 3) * H + y + 3] == Some(team)
					{
						return Some(GameResult::Winner(team));
					}
				}
			}
		}

		// Otherwise the game is running or drawn (if it is full).
		if self.field.iter().any(Option::is_none) {
			None
		} else {
			Some(GameResult::Draw)
		}
	}

	/// Return whos turn it is. Just checks the number of set tiles. Empty field
	/// means X, next O, etc..
	#[must_use]
	pub fn whos_turn(&self) -> Team {
		if self.field.iter().filter(|t| t.is_some()).count() % 2 == 0 {
			Team::X
		} else {
			Team::O
		}
	}

	/// Return the set of possible moves, i.e. which columns still have open
	/// fields.
	#[must_use]
	pub fn possible_moves(&self) -> HashSet<usize> {
		let mut set = HashSet::with_capacity(W);
		for x in 0..W {
			if self.field[x * H + H - 1].is_none() {
				set.insert(x);
			}
		}
		set
	}

	/// Put a tile of the specified team to the corresponding column.
	pub fn put_tile(&mut self, column: usize, team: Team) -> Result<(), Error> {
		if column >= W {
			return Err(Error::IndexOutOfBounds);
		}

		for y in 0..H {
			if self.field[column * H + y].is_none() {
				self.field[column * H + y] = Some(team);
				return Ok(());
			}
		}

		Err(Error::FieldFullAtColumn(team))
	}

	/// Heuristic function to evaluate the board's position. Returns 0.0 for an
	/// estimated draw, above that for estimated wins and below for estimated
	/// losses.
	#[must_use]
	#[allow(clippy::cast_possible_wrap)] // The board isn't that wide, there is no wraps.
	pub fn heuristic_1(&self, me: Team) -> f64 {
		match self.game_result() {
			Some(GameResult::Draw) => return 0.0,
			Some(GameResult::Winner(team)) => return if team == me { f64::MAX } else { f64::MIN },
			None => {}
		}

		let mut value = 0.0;
		for x in 0..W {
			for y in 0..H {
				if let Some(team) = self.field[x * H + y] {
					let mut surrounding = 0.0;
					for (displace_x, displace_y) in [
						(1, 0),
						(1, 1),
						(0, 1),
						(-1, 1),
						(-1, 0),
						(-1_i32, -1_i32),
						(0, -1),
						(1, -1),
					] {
						if let Some(field) = self.field.get(
							(x as i32 + displace_x)
								.saturating_mul(H as i32)
								.saturating_add(y as i32 + displace_y) as usize,
						) {
							match field {
								None => surrounding += 0.333,
								Some(t) if *t == team => surrounding += 1.0,
								_ => surrounding -= 1.0,
							}
						}
					}
					if team == me {
						value += surrounding;
					} else {
						value -= surrounding;
					}
				}
			}
		}

		value
	}
}

impl Team {
	/// Get the other team.
	#[must_use]
	pub fn other(&self) -> Self {
		match self {
			Self::X => Self::O,
			Self::O => Self::X,
		}
	}
}

impl Display for Board {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let mut field = String::new();
		field.push_str(&"----".repeat(W));
		field.pop();
		field.pop();
		field.pop();
		field.push('\n');
		for y in (0..H).rev() {
			for x in 0..W {
				field.push(match self.field[x * H + y] {
					Some(Team::X) => 'X',
					Some(Team::O) => 'O',
					None => ' ',
				});
				field.push_str(" | ");
			}
			field.pop();
			field.pop();
			field.pop();
			field.push('\n');
			field.push_str(&"----".repeat(W));
			field.pop();
			field.pop();
			field.pop();
			field.push('\n');
		}
		field.pop();
		f.write_str(&field)
	}
}

#[cfg(test)]
mod tests {
	#![allow(clippy::unwrap_used, clippy::print_stdout)]

	use super::*;

	/// Make sure each tile on the board only takes a single byte.
	#[test]
	fn size_of() {
		let size_of = std::mem::size_of::<Board>();
		assert_eq!(size_of, W * H);
	}

	#[test]
	fn state_check() {
		let board = Board::default();
		assert_eq!(board.game_result(), None);

		let mut board = Board::default();
		board.put_tile(0, Team::X).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(3, Team::X).unwrap();
		assert_eq!(board.game_result(), Some(GameResult::Winner(Team::X)));

		let mut board = Board::default();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::X).unwrap();
		assert_eq!(board.game_result(), Some(GameResult::Winner(Team::X)));

		let mut board = Board::default();
		board.put_tile(0, Team::X).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(3, Team::X).unwrap();
		assert_eq!(board.game_result(), Some(GameResult::Winner(Team::X)));

		let mut board = Board::default();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(0, Team::X).unwrap();
		assert_eq!(board.game_result(), Some(GameResult::Winner(Team::X)));

		let mut board = Board::default();
		board.put_tile(0, Team::X).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(0, Team::X).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(0, Team::X).unwrap();
		board.put_tile(0, Team::O).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(1, Team::O).unwrap();
		board.put_tile(1, Team::X).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(2, Team::X).unwrap();
		board.put_tile(2, Team::O).unwrap();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(3, Team::X).unwrap();
		board.put_tile(3, Team::O).unwrap();
		board.put_tile(4, Team::X).unwrap();
		board.put_tile(4, Team::O).unwrap();
		board.put_tile(4, Team::X).unwrap();
		board.put_tile(4, Team::O).unwrap();
		board.put_tile(4, Team::X).unwrap();
		board.put_tile(4, Team::O).unwrap();
		board.put_tile(5, Team::O).unwrap();
		board.put_tile(5, Team::X).unwrap();
		board.put_tile(5, Team::O).unwrap();
		board.put_tile(5, Team::X).unwrap();
		board.put_tile(5, Team::O).unwrap();
		board.put_tile(5, Team::X).unwrap();
		board.put_tile(6, Team::X).unwrap();
		board.put_tile(6, Team::O).unwrap();
		board.put_tile(6, Team::X).unwrap();
		board.put_tile(6, Team::O).unwrap();
		board.put_tile(6, Team::X).unwrap();
		board.put_tile(6, Team::O).unwrap();
		println!("Board:\n{board}");
		assert_eq!(board.game_result(), Some(GameResult::Draw));
	}
}
