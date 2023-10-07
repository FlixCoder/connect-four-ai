//! Utils for training and handling burn models.

use burn::{
	module::{Module, ModuleMapper},
	tensor::{backend::Backend, Tensor},
};

/// Burn module mapper that modifies modules with a flat tensor, by adding the
/// parameters on top of the module parameters.
#[derive(Debug)]
pub struct ModifyMapper<B: Backend> {
	/// Flat parameters to be added to the unknown-shaped other tensors.
	pub parameters: Tensor<B, 1>,
	/// Number of parameters already used.
	pub used: usize,
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

/// Time a call to a function.
#[macro_export]
macro_rules! time {
	($e: expr, $msg: literal) => {{
		let now = std::time::Instant::now();
		let result = $e;
		let elapsed = now.elapsed();
		println!(concat!($msg, ": {:?}"), elapsed);
		result
	}};
}
