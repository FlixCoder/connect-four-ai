//! Errors that can appear.

/// Game error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Index out of bounds.
	#[error("Given index was out of bounds")]
	IndexOutOfBounds,

	/// Field already filled at the given column.
	#[error("Field already full at given column")]
	FieldFullAtColumn,
}
