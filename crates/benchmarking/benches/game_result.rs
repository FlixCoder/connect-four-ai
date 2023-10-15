//! Benchmark the game_result checking function.
#![allow(missing_docs, clippy::missing_docs_in_private_items)]

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use game::Game;
use players::RandomPlayer;

criterion_main!(benches);
criterion_group!(benches, game_result_benchmark);

fn game_result_benchmark(c: &mut Criterion) {
	c.bench_function("game_result random", move |b| {
		b.iter_custom(|iters| {
			let mut game = Game::builder().player_x(&RandomPlayer).player_o(&RandomPlayer).build();
			game.run_error_loss();

			let now = Instant::now();
			for _ in 0..iters {
				let board = black_box(game.board());
				black_box(board.game_result());
			}
			now.elapsed()
		});
	});
}
