//! Implementation of model evaluation, evaluating a whole population.

use std::sync::Mutex;

use game::{Game, GameResult, Player, Team};
use players::{MinimaxPlayer, RandomPlayer};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Evaluator interface that evaluators and evaluation functions implement to
/// determine performance of the population's models.
pub trait Evaluator<Model>
where
	Model: Player,
{
	/// Evaluate a set of models and return their scores in the same order.
	fn evaluate(&mut self, models: &[Model]) -> Vec<f32>;
}

impl<Model, F> Evaluator<Model> for F
where
	Model: Player,
	F: FnMut(&[Model]) -> Vec<f32>,
{
	fn evaluate(&mut self, models: &[Model]) -> Vec<f32> {
		(self)(models)
	}
}

/// Evaluation function for a set of models. Run games between each of the
/// leagues participants and return their scores.
pub fn league_scores<Model>(models: &[Model]) -> Vec<f32>
where
	Model: Player + Send + Sync,
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
		let result = game.run_error_loss();
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
/// player and the minimax player.
pub fn player_scores<Model>(models: &[Model]) -> Vec<f32>
where
	Model: Player + Send + Sync,
{
	models
		.par_iter()
		.map(|model| test_random::<_, 1000>(model) + test_minimax::<_, 5>(model))
		.collect()
}

/// Evaluator for a set of models. Run games against a random player, minimax
/// player and a set of previous models.
#[derive(Debug)]
pub struct PlayerPlusEvaluator<Model>
where
	Model: Player + Clone + Send + Sync,
{
	/// Set of previous models to also test against.
	previous: Vec<Model>,
}

impl<Model> Evaluator<Model> for PlayerPlusEvaluator<Model>
where
	Model: Player + Clone + Send + Sync,
{
	fn evaluate(&mut self, models: &[Model]) -> Vec<f32> {
		let scores = models
			.par_iter()
			.map(|model| {
				let mut previous_score = 0.0;
				for previous in &self.previous {
					let mut game = Game::builder().player_x(model).player_o(previous).build();
					let result = game.run_error_loss();
					match result {
						GameResult::Winner(Team::X) => previous_score += 1.0,
						GameResult::Winner(Team::O) => previous_score -= 1.0,
						_ => {}
					}

					let mut game = Game::builder().player_x(previous).player_o(model).build();
					let result = game.run_error_loss();
					match result {
						GameResult::Winner(Team::X) => previous_score -= 1.0,
						GameResult::Winner(Team::O) => previous_score += 1.0,
						_ => {}
					}
				}
				if !self.previous.is_empty() {
					previous_score /= (self.previous.len() * 2) as f32;
				}

				test_random::<_, 1000>(model) + test_minimax::<_, 5>(model) + previous_score
			})
			.collect::<Vec<_>>();

		if let Some((max_index, _max)) = scores
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Score was NaN"))
		{
			self.previous.push(models[max_index].clone());
		}

		scores
	}
}

impl<Model> Default for PlayerPlusEvaluator<Model>
where
	Model: Player + Clone + Send + Sync,
{
	fn default() -> Self {
		Self { previous: Vec::new() }
	}
}

impl<Model> PlayerPlusEvaluator<Model>
where
	Model: Player + Clone + Send + Sync,
{
	/// Add a "previous" model to the set so that it is used in evaluation.
	#[must_use]
	pub fn with_model(mut self, model: Model) -> Self {
		self.previous.push(model);
		self
	}

	/// Add a "previous" model to the set so that it is used in evaluation.
	pub fn add_model(&mut self, model: Model) -> &mut Self {
		self.previous.push(model);
		self
	}

	// TODO: Load and save..
}

/// Test the performance of the model against the random player.
pub fn test_random<Model, const N: usize>(model: &Model) -> f32
where
	Model: Player,
{
	let mut score = 0.0;

	for _ in 0..N / 2 {
		let mut game = Game::builder().player_x(&RandomPlayer).player_o(model).build();
		let result = game.run_error_loss();
		match result {
			GameResult::Winner(Team::X) => score -= 1.0,
			GameResult::Winner(Team::O) => score += 1.0,
			_ => {}
		}
	}

	for _ in 0..N / 2 {
		let mut game = Game::builder().player_x(model).player_o(&RandomPlayer).build();
		let result = game.run_error_loss();
		match result {
			GameResult::Winner(Team::X) => score += 1.0,
			GameResult::Winner(Team::O) => score -= 1.0,
			_ => {}
		}
	}

	score / N as f32
}

/// Test performance against the minimax player.
pub fn test_minimax<Model, const DEEPNESS: usize>(model: &Model) -> f32
where
	Model: Player,
{
	let mut score = 0.0;
	let minimax_player = MinimaxPlayer::new_1(DEEPNESS);

	for _ in 0..50 {
		let mut game = Game::builder().player_x(model).player_o(&minimax_player).build();
		let result = game.run_error_loss();
		match result {
			GameResult::Winner(Team::X) => score += 1.0,
			GameResult::Winner(Team::O) => score -= 1.0,
			_ => {}
		}
	}

	for _ in 0..50 {
		let mut game = Game::builder().player_x(&minimax_player).player_o(model).build();
		let result = game.run_error_loss();
		match result {
			GameResult::Winner(Team::X) => score -= 1.0,
			GameResult::Winner(Team::O) => score += 1.0,
			_ => {}
		}
	}

	score / 100.0
}
