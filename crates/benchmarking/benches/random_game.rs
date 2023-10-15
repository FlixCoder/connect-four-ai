//! Benchmark a game with random players, essentially targeting the game result
//! checking.
#![allow(missing_docs, clippy::missing_docs_in_private_items)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use game::Game;
use players::RandomPlayer;

criterion_main!(benches);
criterion_group!(benches, random_game_benchmark);

fn random_game_benchmark(c: &mut Criterion) {
	c.bench_function("random_game", move |b| {
		b.iter(|| {
			let mut game =
				black_box(Game::builder().player_x(&RandomPlayer).player_o(&RandomPlayer).build());
			game.run_error_loss()
		});
	});
}
