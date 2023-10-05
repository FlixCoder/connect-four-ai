//! Connect four game player implementations.
#![allow(clippy::expect_used)]

mod io;
mod minimax;
mod random;

pub use self::{io::IoPlayer, minimax::MinimaxPlayer, random::RandomPlayer};
