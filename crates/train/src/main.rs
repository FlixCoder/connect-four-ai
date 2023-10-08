//! Execute training of the connect four AI.
#![allow(clippy::print_stdout, clippy::expect_used)]

use std::path::Path;

use burn::tensor::backend::Backend;
use players::{ModelPlayer, NdArrayBackend};
use train::{evaluation::*, time, EvolutionTrainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_path = "./models";
	let population = load_all::<NdArrayBackend>(model_path);

	let mut trainer = EvolutionTrainer::builder()
		.population(population)
		.init_fn(Box::new(ModelPlayer::init))
		.evaluator(PlayerPlusEvaluator::default())
		.population_max(200)
		.population_min(15)
		.generate_new(0.0)
		.mutation_probability(0.1)
		.mutation_std(0.01)
		.build();

	for i in 0..10000 {
		time!(trainer.train_step(), "One training step");

		let score = time!(test_random::<_, 1000>(&trainer.population()[0]), "Testing performance");
		println!("Random performance: {score:.3}");
		let score = test_minimax::<_, 5>(&trainer.population()[0]);
		println!("Minimax performance: {score:.1}");

		if i % 10 == 0 {
			save_all(model_path, trainer.population());
			println!("Models saved!");
		}

		println!();
	}

	Ok(())
}

/// Load all models numbered by index from the given folder.
fn load_all<B>(folder: impl AsRef<Path>) -> Vec<ModelPlayer<B>>
where
	B: Backend,
{
	let Ok(entries) = folder.as_ref().read_dir() else {
		return Vec::new();
	};

	let mut models = Vec::new();
	for entry in entries {
		let entry = entry.expect("read directory entry");
		if entry.path().is_file() {
			let file = folder.as_ref().join(entry.path().file_stem().expect("model file name"));
			let model = ModelPlayer::init().load(file).expect("loading model");
			models.push(model);
		}
	}
	models
}

/// Save all models numbered by index to the given folder.
fn save_all<B>(folder: impl AsRef<Path>, models: &[ModelPlayer<B>])
where
	B: Backend,
{
	if !folder.as_ref().exists() {
		std::fs::create_dir_all(folder.as_ref()).expect("creating directory");
	}
	for (i, model) in models.iter().enumerate() {
		let file = folder.as_ref().join(format!("model_{i:02}"));
		model.clone().save(file).expect("saving model");
	}
}
