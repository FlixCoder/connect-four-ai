//! Errors that can appear.

/// Game error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Builder error: missing field.
	#[error("Missing field: {0}")]
	BuilderMissingField(&'static str),

	/// Index out of bounds.
	#[error("Given index was out of bounds")]
	IndexOutOfBounds,

	/// Field already filled at the given column.
	#[error("Field already full at given column")]
	FieldFullAtColumn,
}
