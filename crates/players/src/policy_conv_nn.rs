//! Player implementation using a convolutional neural network model to choose a
//! column.

use std::path::Path;

use burn::{
	module::Module,
	nn::{
		conv::{Conv2d, Conv2dConfig},
		pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
		Linear, LinearConfig, ReLU,
	},
	record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
	tensor::{activation::softmax, backend::Backend, ElementConversion, Tensor},
};
use game::{Board, Player, Team};

use crate::{ModifyMapper, RandomPlayer};

/// Convolutional neural network model to choose a connect four column. Model
/// and player at once.
#[derive(Debug, Module)]
pub struct ModelPlayer<B: Backend> {
	/// Conv layer 1.
	conv1: Conv2d<B>,
	/// Conv layer 2.
	conv2: Conv2d<B>,
	/// Pooling layer.
	pool: AdaptiveAvgPool2d,
	/// Linear layer 1.
	linear1: Linear<B>,
	/// Linear layer 2.
	linear2: Linear<B>,
	/// Activation.
	activation: ReLU,
}

impl<B: Backend> ModelPlayer<B> {
	/// Create new fresh random model.
	#[must_use]
	pub fn init() -> Self {
		Self {
			conv1: Conv2dConfig::new([1, 6], [3, 3]).init(),
			conv2: Conv2dConfig::new([6, 10], [4, 4]).init(),
			pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
			linear1: LinearConfig::new(10 * 4 * 4, 80).init(),
			linear2: LinearConfig::new(80, 7).init(),
			activation: ReLU::new(),
		}
		.no_grad()
	}

	/// Modify the model with a random variation tensor.
	#[must_use]
	pub fn modify(self, parameters: Tensor<B, 1>) -> Self {
		let mut mapper = ModifyMapper { parameters, used: 0 };
		let this = self.map(&mut mapper);
		mapper.verify();
		this
	}

	/// Load the module from a file.
	pub fn load(self, path: impl AsRef<Path>) -> Result<Self, burn::record::RecorderError> {
		self.load_file(path.as_ref(), &NamedMpkGzFileRecorder::<FullPrecisionSettings>::new())
			.map(Module::no_grad)
	}

	/// Save the module to a file.
	pub fn save(self, path: impl AsRef<Path>) -> Result<(), burn::record::RecorderError> {
		self.save_file(path.as_ref(), &NamedMpkGzFileRecorder::<FullPrecisionSettings>::new())
	}

	/// Run model prediction.
	fn forward(&self, field: Tensor<B, 3>) -> Tensor<B, 2> {
		let [batch, height, width] = field.dims();
		let data = field.reshape([batch, 1, height, width]);
		let data = self.conv1.forward(data);
		let data = self.conv2.forward(data);
		let data = self.activation.forward(data);
		let data = self.pool.forward(data);
		let data = data.reshape([batch, 10 * 4 * 4]);
		let data = self.linear1.forward(data);
		let data = self.activation.forward(data);
		let data = self.linear2.forward(data);
		softmax(data, 1)
	}

	/// Convert the board to a workable tensor.
	fn board_to_tensor(board: &Board, me: Team) -> Tensor<B, 2> {
		let data: Vec<_> = board
			.field()
			.iter()
			.map(|tile| match tile {
				None => 0.0,
				Some(team) if *team == me => 1.0,
				_ => -1.0,
			})
			.collect();
		Tensor::from_floats(data.as_slice()).reshape([7, 6]).transpose()
	}

	/// Convert board to a field tensor and run the model prediction.
	fn predict(&self, board: &Board, me: Team) -> usize {
		assert_eq!(board.dimensions(), (7, 6));
		let data = Self::board_to_tensor(board, me);

		let classes = self.forward(data.reshape([1, 6, 7])).reshape([7]);
		let select: u8 = classes.argmax(0).into_scalar().elem();
		select as usize
	}
}

impl<B: Backend> Player for ModelPlayer<B> {
	fn make_move(&self, board: &Board, me: Team) -> usize {
		let column = self.predict(board, me);
		let possible_moves = board.possible_moves();
		if possible_moves.contains(&column) {
			column
		} else {
			RandomPlayer.make_move(board, me)
		}
	}
}
