//! Utils for training and handling burn models.

use burn::{
	module::{Module, ModuleMapper, ModuleVisitor},
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

/// Burn module mapper that overrides modules with a flat tensor.
#[derive(Debug)]
pub struct OverrideMapper<B: Backend> {
	/// Flat parameters to overwrite the unknown-shaped other tensors.
	pub parameters: Tensor<B, 1>,
	/// Number of parameters already used.
	pub used: usize,
}

impl<B: Backend> OverrideMapper<B> {
	/// Verify that the mapper has been fully using the parameters.
	pub fn verify(self) {
		if self.used != self.parameters.num_params() {
			panic!("OverrideMapper not fully used!");
		}
	}
}

impl<B: Backend> ModuleMapper<B> for OverrideMapper<B> {
	fn map<const D: usize>(
		&mut self,
		_id: &burn::module::ParamId,
		tensor: Tensor<B, D>,
	) -> Tensor<B, D> {
		let num = tensor.num_params();
		let range = Tensor::arange(self.used..self.used + num);
		let params = self.parameters.clone().select(0, range).reshape(tensor.shape());
		self.used += num;
		params
	}
}

/// Burn module visitor to flatten the whole model into a flat tensor.
#[derive(Debug)]
pub struct FlattenVisitor<B: Backend> {
	/// parameters that are being collected.
	pub parameters: Option<Tensor<B, 1>>,
}

impl<B: Backend> ModuleVisitor<B> for FlattenVisitor<B> {
	fn visit<const D: usize>(&mut self, _id: &burn::module::ParamId, tensor: &Tensor<B, D>) {
		let flat = tensor.clone().reshape([tensor.num_params()]);
		self.parameters = Some(if let Some(params) = self.parameters.take() {
			Tensor::cat(vec![params, flat], 0)
		} else {
			flat
		});
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
