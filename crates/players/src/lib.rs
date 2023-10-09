//! Connect four game player implementations.
#![allow(clippy::expect_used)]

mod io;
mod minimax;
mod policy_conv_nn;
mod random;
mod value_conv_nn;

pub use burn::backend::{NdArrayBackend, WgpuBackend};

pub use self::{
	io::IoPlayer, minimax::MinimaxPlayer, policy_conv_nn::AiPolicyPlayer, random::RandomPlayer,
	value_conv_nn::AiValuePlayer,
};
