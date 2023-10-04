//! Generic player implementation.

use std::fmt::Debug;

use crate::{board::Board, Team};

/// Everything a player needs to play to game of connect four.
pub trait Player: Debug {
	/// Make a move based on the current board positions. Return the column to
	/// put the new tile in.
	fn make_move(&self, board: &Board, me: Team) -> usize;
}
