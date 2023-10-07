//! Implementation of model evaluation, evaluating a whole population.

use std::sync::Mutex;

use burn::{module::Module, tensor::backend::Backend};
use game::{Game, GameResult, Player, Team};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Evaluation function for a set of models. Run games between each of the
/// leagues participants and return their scores.
pub fn league_scores<B, Model>(models: &[Model]) -> Vec<f32>
where
	B: Backend,
	Model: Module<B> + Player,
{
	let mut matchups = Vec::new();
	for i in 0..models.len() {
		for j in 0..models.len() {
			matchups.push((i, j));
		}
	}

	let scores = Mutex::new(vec![0.0; models.len()]);
	matchups.into_par_iter().for_each(|(i, j)| {
		let mut game = Game::builder().player_x(&models[i]).player_o(&models[j]).build();
		let result = game.run().expect("running game");
		if let GameResult::Winner(winner) = result {
			let mut scores = scores.lock().expect("lock poisened");
			if winner == Team::X {
				scores[i] += 1.0;
				scores[j] -= 1.0;
			} else {
				scores[i] -= 1.0;
				scores[j] += 1.0;
			}
		}
	});
	scores.into_inner().expect("lock poisened")
}
