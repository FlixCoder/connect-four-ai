//! Optimizers implementation.

use std::{fs::File, path::Path};

use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};

/// Optimizer functionality trait.
pub trait Optimizer<B: Backend> {
	/// Compute a step, i.e. get the gradient and compute the parameter updates
	/// (delta).
	fn step(&mut self, gradient: Tensor<B, 1>) -> Tensor<B, 1>;
}

/// SGD Optimizer with momentum.
#[derive(Debug, Serialize, Deserialize, typed_builder::TypedBuilder)]
pub struct Sgd<B: Backend> {
	/// The learning rate lr.
	learning_rate: f32,
	/// Beta, the momentum coefficient.
	momentum: f32,
	/// Last momentum gradient.
	#[serde(with = "tensor_serde")]
	#[builder(default = Tensor::zeros([1]))]
	last_v: Tensor<B, 1>,
	/// Number of iterations t.
	#[builder(default)]
	iterations: usize,
}

impl<B: Backend> Optimizer<B> for Sgd<B> {
	fn step(&mut self, gradient: Tensor<B, 1>) -> Tensor<B, 1> {
		if self.last_v.shape() != gradient.shape() {
			self.last_v = Tensor::zeros(gradient.shape());
		}

		// Momentum update.
		self.last_v = self.last_v.clone().mul_scalar(self.momentum)
			+ gradient.mul_scalar(1.0 - self.momentum);
		// Compute delta based on momentum.
		let delta = self.last_v.clone().mul_scalar(-self.learning_rate);

		self.iterations += 1;
		delta
	}
}

impl<B: Backend> Sgd<B> {
	/// Save the optimizer to a file.
	pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
		let file = File::create(path)?;
		serde_json::to_writer(file, self)?;
		Ok(())
	}

	/// Load the optimizer from a file.
	pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
		let file = File::open(path)?;
		let this = serde_json::from_reader(file)?;
		Ok(this)
	}
}

/// Serde module for serializing and deserializing burn tensors.
mod tensor_serde {
	use burn::tensor::{backend::Backend, ElementConversion, Tensor};
	use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serializer};

	/// Serialize a burn tensor as array.
	pub fn serialize<S: Serializer, B: Backend>(
		tensor: &Tensor<B, 1>,
		serializer: S,
	) -> Result<S::Ok, S::Error> {
		let mut seq = serializer.serialize_seq(Some(tensor.dims()[0]))?;
		for val in tensor.clone().into_data().value {
			seq.serialize_element(&val.elem::<f32>())?;
		}
		seq.end()
	}

	/// Deserialize a burn tensor from an array.
	pub fn deserialize<'de, D: Deserializer<'de>, B: Backend>(
		deserializer: D,
	) -> Result<Tensor<B, 1>, D::Error> {
		let data = Vec::<f32>::deserialize(deserializer)?;
		Ok(Tensor::from_floats(data.as_slice()))
	}
}
