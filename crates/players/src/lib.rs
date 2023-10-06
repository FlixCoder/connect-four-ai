//! Connect four game player implementations.
#![allow(clippy::expect_used)]

mod io;
mod minimax;
mod policy_conv_nn;
mod random;

pub use burn::backend::{NdArrayBackend, WgpuBackend};
use burn::{
	module::{Module, ModuleMapper},
	tensor::{backend::Backend, Tensor},
};

pub use self::{
	io::IoPlayer, minimax::MinimaxPlayer, policy_conv_nn::ModelPlayer, random::RandomPlayer,
};

/// Burn module mapper that modifies modules with a flat tensor, by adding the
/// parameters on top of the module parameters.
struct ModifyMapper<B: Backend> {
	/// Flat parameters to be added to the unknown-shaped other tensors.
	parameters: Tensor<B, 1>,
	/// Number of parameters already used.
	used: usize,
}

impl<B: Backend> ModifyMapper<B> {
	/// Verify that the mapper has been fully using the parameters.
	pub fn verify(self) {
		if self.used != self.parameters.num_params() {
			panic!("ModifyMapper not fully used!");
		}
	}
}

impl<B: Backend> ModuleMapper<B> for ModifyMapper<B> {
	fn map<const D: usize>(
		&mut self,
		_id: &burn::module::ParamId,
		tensor: Tensor<B, D>,
	) -> Tensor<B, D> {
		let num = tensor.num_params();
		let range = Tensor::arange(self.used..self.used + num);
		let params = self.parameters.clone().select(0, range).reshape(tensor.shape());
		self.used += num;
		tensor.add(params)
	}
}
