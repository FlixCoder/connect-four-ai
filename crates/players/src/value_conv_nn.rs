//! Player implementation using a convolutional neural network model as value
//! heuristic.

use std::path::Path;

use burn::{
	module::Module,
	nn::{
		conv::{Conv2d, Conv2dConfig},
		Linear, LinearConfig,
	},
	record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
	tensor::{activation::tanh, backend::Backend, ElementConversion, Tensor},
};
use game::{Board, Player, Team};

use crate::MinimaxPlayer;

/// Convolutional neural network model to evaluate board positions. Model
/// and player at once.
#[derive(Debug, Module)]
pub struct AiValuePlayer<B: Backend> {
	/// Minimax deepness level.
	deepness: usize,
	/// Conv layer 1.
	conv1: Conv2d<B>,
	/// Linear layer 1.
	linear1: Linear<B>,
	/// Linear layer 2.
	linear2: Linear<B>,
}

impl<B: Backend> AiValuePlayer<B> {
	/// Create new fresh random model.
	#[must_use]
	pub fn init(deepness: usize) -> Self {
		Self {
			deepness,
			conv1: Conv2dConfig::new([1, 8], [4, 4]).init(),
			linear1: LinearConfig::new(8 * 3 * 4, 50).init(), // 4x4 kernel makes 6x7 to 3x4.
			linear2: LinearConfig::new(50, 1).init(),
		}
		.no_grad()
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
		let data = tanh(data);
		let [batch, channels, height, width] = data.dims();
		let data = data.reshape([batch, channels * height * width]);
		let data = self.linear1.forward(data);
		let data = tanh(data);
		let data = self.linear2.forward(data);
		tanh(data)
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
	fn predict(&self, board: &Board, me: Team) -> f64 {
		assert_eq!(board.dimensions(), (7, 6));
		let data = Self::board_to_tensor(board, me);

		let value = self.forward(data.reshape([1, 6, 7])).reshape([1]);
		value.into_scalar().elem()
	}
}

impl<B: Backend> Player for AiValuePlayer<B> {
	fn make_move(&self, board: &Board, me: Team) -> usize {
		let heuristic = |b: &Board, m: Team| self.predict(b, m);
		let minimax = MinimaxPlayer::new(self.deepness, &heuristic);
		minimax.make_move(board, me)
	}
}
