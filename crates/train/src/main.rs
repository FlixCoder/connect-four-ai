//! Execute training of the connect four AI.
#![allow(clippy::print_stdout, clippy::expect_used)]

use burn::tensor::backend::Backend;
use game::{Game, GameResult, Team};
use players::{MinimaxPlayer, ModelPlayer, NdArrayBackend, RandomPlayer};
use train::{evaluation::league_scores, optimizers::Sgd, time, Trainer};

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
		Sgd::builder().learning_rate(0.05).momentum(0.9).build()
	});

	let mut trainer = Trainer::builder()
		.model(model)
		.std(0.025)
		.samples(100)
		.evaluator(league_scores)
		.optimizer(optimizer)
		.build();

	for i in 0..10000 {
		time!(trainer.train_step(), "One training step");

		if i % 10 == 0 {
			let optimizer = trainer.optimizer();
			optimizer.save(optimizer_path)?;
			let model = trainer.model().clone();
			model.save(model_path)?;
			println!("Model saved!");
		}

		let score = time!(test_random(trainer.model()), "Testing performance");
		println!("Performance score: {score:.3}");
		let score = test_minimax(trainer.model());
		println!("Minimax performance: {score:.1}");

		println!();
	}

	Ok(())
}

/// Test the performance of the model against the random player.
fn test_random<B: Backend>(model: &ModelPlayer<B>) -> f32 {
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

/// Test performance against the minimax player.
fn test_minimax<B: Backend>(model: &ModelPlayer<B>) -> f32 {
	let mut score = 0.0;
	let minimax_player = MinimaxPlayer::new_1(5);

	let mut game = Game::builder().player_x(model).player_o(&minimax_player).build();
	let result = game.run().expect("running game");
	match result {
		GameResult::Winner(Team::X) => score += 1.0,
		GameResult::Winner(Team::O) => score -= 1.0,
		_ => {}
	}

	let mut game = Game::builder().player_x(&minimax_player).player_o(model).build();
	let result = game.run().expect("running game");
	match result {
		GameResult::Winner(Team::X) => score -= 1.0,
		GameResult::Winner(Team::O) => score += 1.0,
		_ => {}
	}

	score / 2.0
}
