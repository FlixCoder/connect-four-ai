//! Connect four game player implementations.
#![allow(clippy::expect_used)]

mod io;
mod minimax;
mod policy_conv_nn;
mod random;

pub use burn::backend::{NdArrayBackend, WgpuBackend};

pub use self::{
	io::IoPlayer, minimax::MinimaxPlayer, policy_conv_nn::ModelPlayer, random::RandomPlayer,
};
