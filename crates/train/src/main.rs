//! Execute training of the connect four AI.
#![allow(clippy::print_stdout, clippy::expect_used)]

use burn::tensor::backend::Backend;
use game::{Game, GameResult, Team};
use players::{ModelPlayer, NdArrayBackend, RandomPlayer};
use train::{optimizers::Sgd, time, Trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_path = "model";
	let optimizer_path = "optimizer.json";

	let model = ModelPlayer::<NdArrayBackend>::init().load(model_path).unwrap_or_else(|err| {
		println!("Failed loading model: {err}");
		println!("Starting with new model");
		ModelPlayer::init()
	});

	let optimizer = Sgd::load(optimizer_path).unwrap_or_else(|err| {
		println!("Failed loading optimizer: {err}");
		println!("Starting with new optimizer");
		Sgd::builder().learning_rate(0.01).momentum(0.9).build()
	});

	let mut trainer =
		Trainer::builder().model(model).std(0.02).samples(100).optimizer(optimizer).build();

	for _ in 0..100 {
		time!(trainer.train_step(), "One training step");

		let optimizer = trainer.optimizer();
		optimizer.save(optimizer_path)?;
		let model = trainer.model().clone();
		model.save(model_path)?;

		let score = time!(test_performance(trainer.model()), "Testing performance");
		println!("Performance score: {score:.3}");

		println!();
	}

	Ok(())
}

/// Test the performance of the model.
fn test_performance<B: Backend>(model: &ModelPlayer<B>) -> f32 {
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
}
