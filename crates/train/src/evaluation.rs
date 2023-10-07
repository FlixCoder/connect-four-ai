//! Implementation of model evaluation, evaluating a whole population.

use std::sync::Mutex;

use burn::{module::Module, tensor::backend::Backend};
use game::{Game, GameResult, Player, Team};
use players::RandomPlayer;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

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

/// Evaluation function for a set of models. Run games against the random
/// player.
pub fn random_player_scores<B, Model>(models: &[Model]) -> Vec<f32>
where
	B: Backend,
	Model: Module<B> + Player,
{
	models
		.par_iter()
		.map(|model| {
			let mut score = 0.0;

			for _ in 0..500 {
				let mut game = Game::builder().player_x(&RandomPlayer).player_o(model).build();
				let result = game.run().expect("running game");
				match result {
					GameResult::Winner(Team::X) => score -= 1.0,
					GameResult::Winner(Team::O) => score += 1.0,
					_ => {}
				}
			}

			for _ in 0..500 {
				let mut game = Game::builder().player_x(model).player_o(&RandomPlayer).build();
				let result = game.run().expect("running game");
				match result {
					GameResult::Winner(Team::X) => score += 1.0,
					GameResult::Winner(Team::O) => score -= 1.0,
					_ => {}
				}
			}

			score / 1000.0
		})
		.collect()
}
