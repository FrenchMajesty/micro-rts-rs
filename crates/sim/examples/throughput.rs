//! Measure the Rust step loop with fixed, deterministic orders.
use micro_rts_sim::{Action, Direction, Game, Mobile, Order, Status, Visibility};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut game = Game::new(Visibility::Full);
    let opening = [
        vec![
            Order {
                unit: 0,
                action: Action::Train(Mobile::Worker, Direction::North),
            },
            Order {
                unit: 1,
                action: Action::Move(Direction::East),
            },
        ],
        vec![
            Order {
                unit: 2,
                action: Action::Train(Mobile::Worker, Direction::South),
            },
            Order {
                unit: 3,
                action: Action::Move(Direction::West),
            },
        ],
    ];
    let idle = [vec![], vec![]];
    let ticks = 1_000_000;
    let start = Instant::now();
    for _ in 0..ticks {
        if game.status() != Status::Running {
            game.reset();
        }
        let orders = if game.tick() == 0 { &opening } else { &idle };
        black_box(game.step(black_box(orders)).unwrap());
    }
    println!(
        "Rust simulation: {:.0} joint ticks/s ({ticks} ticks, opening and idle orders)",
        ticks as f64 / start.elapsed().as_secs_f64()
    );
}
