//! Connect four agent training implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

pub mod evaluation;
pub mod optimizers;
mod utils;

use std::{fmt::Debug, marker::PhantomData};

use burn::{
	module::Module,
	tensor::{backend::Backend, Tensor},
};
use game::Player;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Distribution;

use self::{evaluation::Evaluator, optimizers::Optimizer, utils::ModifyMapper};

/// The model trainer.
#[derive(Debug, typed_builder::TypedBuilder)]
pub struct Trainer<B, Model, Eval, Opt>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model>,
	Opt: Optimizer<B> + Debug,
{
	/// The model to train.
	model: Model,
	/// The backend to use for tensors.
	#[builder(setter(skip), default)]
	backend: PhantomData<B>,
	/// Standard deviation to use for sampling while training.
	std: f32,
	/// The double-sided sample/population size.
	samples: usize,
	/// Evaluation function to compute the scores of a population.
	evaluator: Eval,
	/// The optimizer to use.
	optimizer: Opt,
}

impl<B, Model, Eval, Opt> Trainer<B, Model, Eval, Opt>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model>,
	Opt: Optimizer<B> + Debug,
{
	/// Get the optimizer.
	pub fn optimizer(&self) -> &Opt {
		&self.optimizer
	}

	/// Get the inner model as copy.
	pub fn model(&self) -> &Model {
		&self.model
	}

	/// Get the evaluator.
	pub fn evaluator(&self) -> &Eval {
		&self.evaluator
	}

	/// Get the evaluator mutably.
	pub fn evaluator_mut(&mut self) -> &mut Eval {
		&mut self.evaluator
	}

	/// Get a modified copy of the model.
	fn modified_model(&self, parameters: Tensor<B, 1>) -> Model {
		let mut mapper = ModifyMapper { parameters, used: 0 };
		let this = self.model.clone().map(&mut mapper);
		mapper.verify();
		this
	}

	/// Generate the model parameter updates for the i's iteration of sampling.
	fn generate_model_params(&self, seed: u64, i: usize) -> Tensor<B, 1> {
		let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
		let disposition = rand_distr::Normal::new(0.0, self.std)
			.expect("standard deviation must be finite and defined")
			.sample_iter(&mut rng)
			.take(self.model.num_params())
			.collect::<Vec<_>>();
		Tensor::from_floats(disposition.as_slice())
	}

	/// Generate the population for one iteration. Use the seed to generate
	/// random dispositions.
	fn generate_population(&self, seed: u64) -> Vec<Model> {
		let mut population = Vec::with_capacity(self.samples * 2);
		for i in 0..self.samples {
			let disposition = self.generate_model_params(seed, i);
			population.push(self.modified_model(disposition.clone()));
			population.push(self.modified_model(disposition.mul_scalar(-1)));
		}
		population
	}

	/// Compute the gradient from the scores. Generates the same dispositions as
	/// the population generation using the same seed.
	fn compute_gradient(&self, seed: u64, scores: &[f32]) -> Tensor<B, 1> {
		let mut gradient = Tensor::zeros([self.model.num_params()]);
		for i in 0..self.samples {
			let disposition = self.generate_model_params(seed, i);
			gradient = gradient + disposition.mul_scalar(scores[i * 2] - scores[i * 2 + 1]);
		}
		gradient.mul_scalar(1.0 / (2.0 * self.samples as f32 * self.std))
	}

	/// Train the model for one step.
	pub fn train_step(&mut self) -> &mut Self {
		let seed = rand::random();
		let population = time!(self.generate_population(seed), "Generating population");
		let mut scores = time!(self.evaluator.evaluate(&population), "Computing population scores");
		normalize_scores(&mut scores);
		let gradient = time!(self.compute_gradient(seed, &scores), "Computing gradient");
		// Invert gradient so that we do descent and not ascent.
		let delta = self.optimizer.step(-gradient);
		self.model = self.modified_model(delta);
		self
	}
}

/// Normalize a vec of floats.
fn normalize_scores(scores: &mut [f32]) {
	let mut mean = 0.0;
	for score in scores.iter() {
		mean += *score;
	}
	mean /= scores.len() as f32;

	let mut std = 0.0;
	for score in scores.iter() {
		let diff = *score - mean;
		std += diff * diff;
	}
	std /= scores.len() as f32;
	std = std.sqrt();

	for score in scores {
		*score = (*score - mean) / std;
	}
}
