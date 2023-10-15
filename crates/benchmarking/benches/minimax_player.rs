//! Benchmark a game with minimax players, benchmarking the minimax player
//! performance.
#![allow(missing_docs, clippy::missing_docs_in_private_items)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use game::Game;
use players::MinimaxPlayer;

criterion_main!(benches);
criterion_group!(benches, minimax_player_benchmark);

fn minimax_player_benchmark(c: &mut Criterion) {
	c.bench_function("minimax_player", move |b| {
		b.iter(|| {
			let player = MinimaxPlayer::new_1(5);
			let mut game = black_box(Game::builder().player_x(&player).player_o(&player).build());
			game.run_error_loss()
		});
	});
}
