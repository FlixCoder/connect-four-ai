//! Connect four game player implementations.

mod io;
mod minimax;
mod random;

pub use self::{io::IoPlayer, minimax::MinimaxPlayer, random::RandomPlayer};
