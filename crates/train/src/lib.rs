//! Connect four agent training implementation.
#![allow(clippy::print_stdout, clippy::expect_used)]

pub mod evaluation;
pub mod optimizers;
mod utils;

use std::{fmt::Debug, marker::PhantomData};

use burn::{
	module::Module,
	tensor::{backend::Backend, ElementConversion, Tensor},
};
use game::Player;
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_distr::Distribution;

use self::{
	evaluation::Evaluator,
	optimizers::Optimizer,
	utils::{FlattenVisitor, ModifyMapper, OverrideMapper},
};

/// The model trainer using evolution strategy optimization.
#[derive(Debug, typed_builder::TypedBuilder)]
pub struct EsTrainer<B, Model, Eval, Opt>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model>,
	Opt: Optimizer<B> + Debug,
{
	/// The backend to use for tensors.
	#[builder(setter(skip), default)]
	backend: PhantomData<B>,
	/// The model to train.
	model: Model,
	/// Standard deviation to use for sampling while training.
	std: f32,
	/// The double-sided sample/population size.
	samples: usize,
	/// Evaluation function to compute the scores of a population.
	evaluator: Eval,
	/// The optimizer to use.
	optimizer: Opt,
}

impl<B, Model, Eval, Opt> EsTrainer<B, Model, Eval, Opt>
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

/// The model trainer using pure evolution with breeding, mutation and
/// selection.
#[derive(typed_builder::TypedBuilder)]
pub struct EvolutionTrainer<B, Model, Eval>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model>,
{
	/// The backend to use for tensors.
	#[builder(setter(skip), default)]
	backend: PhantomData<B>,
	/// The population to use for training.
	population: Vec<Model>,
	/// Function to initialize a new fresh model.
	init_fn: Box<dyn FnMut() -> Model>,
	/// Maximum population size to generate.
	population_max: usize,
	/// Minimum population size to select.
	population_min: usize,
	/// Probability to generate a new model.
	generate_new: f64,
	/// Probability of mutation.
	mutation_probability: f64,
	/// Mutation range standard deviation.
	mutation_std: f64,
	/// Evaluation function to compute the scores of a population.
	evaluator: Eval,
}

impl<B, Model, Eval> EvolutionTrainer<B, Model, Eval>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model>,
{
	/// Get the population.
	pub fn population(&self) -> &[Model] {
		&self.population
	}

	/// Get the evaluator.
	pub fn evaluator(&self) -> &Eval {
		&self.evaluator
	}

	/// Get the evaluator mutably.
	pub fn evaluator_mut(&mut self) -> &mut Eval {
		&mut self.evaluator
	}

	/// Breed a new model from 2 parent models.
	pub fn breed(a: &Model, b: &Model) -> Model {
		let mut visitor_a = FlattenVisitor { parameters: None };
		a.visit(&mut visitor_a);
		let params_a = visitor_a.parameters.expect("Model should not be empty");
		let mut visitor_b = FlattenVisitor { parameters: None };
		b.visit(&mut visitor_b);
		let params_b = visitor_b.parameters.expect("Model should not be empty");

		let mask = Tensor::random(
			[a.num_params()],
			burn::tensor::Distribution::Uniform(0.0.elem(), 1.0.elem()),
		);
		let parameters = mask.clone() * params_a + mask.mul_scalar(-1.0).add_scalar(1.0) * params_b;

		let mut setter = OverrideMapper { parameters, used: 0 };
		let child = a.clone().map(&mut setter);
		setter.verify();
		child
	}

	/// Mutate a model with random permutations.
	pub fn mutate(&self, model: Model) -> Model {
		let parameters = Tensor::random(
			[model.num_params()],
			burn::tensor::Distribution::Normal(0.0, self.mutation_std),
		);
		let mut mapper = ModifyMapper { parameters, used: 0 };
		let model = model.map(&mut mapper);
		mapper.verify();
		model
	}

	/// Generate population via breeding and mutation.
	pub fn generate_population(&mut self) {
		let mut rng = thread_rng();

		while self.population.len() < self.population_min {
			self.population.push((self.init_fn)());
		}

		while self.population.len() < self.population_max {
			if rng.gen::<f64>() < self.generate_new {
				self.population.push((self.init_fn)());
			} else {
				let selected = self.population.choose_multiple(&mut rng, 2).collect::<Vec<_>>();
				let mut model = Self::breed(selected[0], selected[1]);
				if rng.gen::<f64>() < self.mutation_probability {
					model = self.mutate(model);
				}
				self.population.push(model);
			}
		}
	}

	/// Train for one step.
	pub fn train_step(&mut self) -> &mut Self {
		time!(self.generate_population(), "Generating population");
		let scores =
			time!(self.evaluator.evaluate(&self.population), "Computing population scores");

		// Sort population by scores and select the best.
		let mut population_scores = self.population.drain(..).zip(scores).collect::<Vec<_>>();
		population_scores
			.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).expect("Score was NaN"));
		self.population.append(
			&mut population_scores.into_iter().take(self.population_min).map(|(m, _s)| m).collect(),
		);

		self
	}
}

impl<B, Model, Eval> Debug for EvolutionTrainer<B, Model, Eval>
where
	B: Backend + Debug,
	Model: Module<B> + Player + Debug,
	Eval: Evaluator<Model> + Debug,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("EvolutionTrainer")
			.field("backend", &self.backend)
			.field("population", &self.population)
			.field("init_fn", &"<function to initialize model>")
			.field("population_max", &self.population_max)
			.field("population_min", &self.population_min)
			.field("generate_new", &self.generate_new)
			.field("mutation_probability", &self.mutation_probability)
			.field("mutation_std", &self.mutation_std)
			.field("evaluator", &self.evaluator)
			.finish()
	}
}
