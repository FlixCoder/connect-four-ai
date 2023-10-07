//! Execute training of the connect four AI.
#![allow(clippy::print_stdout, clippy::expect_used)]

use players::{ModelPlayer, NdArrayBackend};
use train::{evaluation::*, optimizers::*, time, Trainer};

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
		.evaluator(player_scores)
		.optimizer(optimizer)
		.build();

	for i in 0..10000 {
		time!(trainer.train_step(), "One training step");

		let score = time!(test_random::<_, 1000>(trainer.model()), "Testing performance");
		println!("Random performance: {score:.3}");
		let score = test_minimax::<_, 5>(trainer.model());
		println!("Minimax performance: {score:.1}");

		if i % 10 == 0 {
			let optimizer = trainer.optimizer();
			optimizer.save(optimizer_path)?;
			let model = trainer.model().clone();
			model.save(model_path)?;
			println!("Model saved!");
		}

		println!();
	}

	Ok(())
}
