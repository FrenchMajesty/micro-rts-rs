use super::*;
use Action::*;
use Direction::*;
use Player::*;

fn order(unit: UnitId, action: Action) -> Order {
    Order { unit, action }
}

fn step(game: &mut Game, a: Vec<Order>, b: Vec<Order>) -> Status {
    game.step(&[a, b]).unwrap()
}

fn pos(x: u8, y: u8) -> Position {
    Position { x, y }
}

#[test]
fn starting_map_is_symmetric_and_action_mask_has_native_choices() {
    let game = Game::new(Visibility::Full);
    let a = game.observe(Zero);
    let b = game.observe(One);
    assert_eq!(a.board, std::array::from_fn(|i| b.board[CELLS - 1 - i]));
    assert_eq!(a.credits, [Some(3), Some(3)]);
    let mask = game.legal_actions(Zero);
    assert!(mask[pos(1, 1).index()][13]);
    assert!(!mask[pos(1, 1).index()][17]);
    assert!(!mask[pos(2, 1).index()][4]);
    assert!(mask[pos(2, 1).index()][2]);
    assert_eq!(game.source_mask(Zero).into_iter().filter(|&v| v).count(), 2);
    assert!(Action::from_index(ACTIONS).is_none());
}

#[test]
fn waits_are_safe_and_duplicate_orders_are_atomic() {
    let mut game = Game::new(Visibility::Full);
    let before = game.state_hash();
    let result = game.step(&[
        vec![order(0, Train(Mobile::Worker, North))],
        vec![order(3, Move(North)), order(3, Wait)],
    ]);
    assert_eq!(
        result,
        Err(StepError::DuplicateOrder {
            player: One,
            unit: 3
        })
    );
    assert_eq!(game.state_hash(), before);
    step(
        &mut game,
        vec![order(1, Move(West)), order(3, Move(West))],
        vec![],
    );
    assert_eq!(game.units[1].position, pos(2, 1));
    assert!(game.units.iter().all(|u| u.pending.is_none()));
}

#[test]
fn competing_moves_resolve_without_player_priority() {
    let mut game = Game::new(Visibility::Full);
    game.units[1].position = pos(3, 3);
    game.units[3].position = pos(4, 4);
    step(
        &mut game,
        vec![order(1, Move(East))],
        vec![order(3, Move(North))],
    );
    assert_eq!(game.units[1].pending.unwrap().completes_at, 2);
    step(&mut game, vec![], vec![]);
    assert_eq!(game.units[1].position, pos(3, 3));
    assert_eq!(game.units[3].position, pos(4, 4));
}

#[test]
fn production_reserves_credits_and_refunds_a_contested_spawn() {
    let mut game = Game::new(Visibility::Full);
    game.units[1].position = pos(0, 0);
    game.units[3].position = pos(3, 1);
    game.units[3].kind = UnitKind::Scout;
    step(
        &mut game,
        vec![order(0, Train(Mobile::Worker, East))],
        vec![],
    );
    assert_eq!(game.credits[0], 0);
    for _ in 0..4 {
        step(&mut game, vec![], vec![]);
    }
    step(&mut game, vec![], vec![order(3, Move(West))]);
    assert_eq!(game.credits[0], 3);
    assert_eq!(game.units.len(), 4);
    assert_eq!(game.units[3].position, pos(3, 1));

    step(
        &mut game,
        vec![order(0, Train(Mobile::Worker, North))],
        vec![],
    );
    for _ in 0..5 {
        step(&mut game, vec![], vec![]);
    }
    let worker = game.units.last().unwrap();
    assert_eq!(
        (worker.id, worker.kind, worker.position),
        (4, UnitKind::Worker, pos(1, 0))
    );
    assert_eq!(game.credits[0], 0);
}

#[test]
fn a_dead_base_cancels_unfinished_training_and_refunds_credits() {
    let mut game = Game::new(Visibility::Full);
    game.units[0].hp = 1;
    game.units[3].position = pos(0, 1);
    step(
        &mut game,
        vec![order(0, Train(Mobile::Worker, North))],
        vec![order(3, Attack(East))],
    );
    step(&mut game, vec![], vec![]);
    assert!(game.units.iter().all(|unit| unit.id != 0 && unit.id != 4));
    assert_eq!(game.credits[0], 3);
    assert_eq!(game.status(), Status::Running);
}

#[test]
fn gathering_and_deposit_use_one_snapshot() {
    let mut game = Game::new(Visibility::Full);
    game.units[1].position = pos(1, 2);
    game.units[3].position = pos(0, 3);
    game.deposits[0].ore = 1;
    step(
        &mut game,
        vec![order(1, Gather(South))],
        vec![order(3, Gather(East))],
    );
    step(&mut game, vec![], vec![]);
    assert_eq!(game.deposits[0].ore, 1);
    assert_eq!(game.units[1].cargo, 0);
    assert_eq!(game.units[3].cargo, 0);

    step(&mut game, vec![order(1, Gather(South))], vec![]);
    step(&mut game, vec![], vec![]);
    assert_eq!(game.deposits[0].ore, 0);
    assert_eq!(game.units[1].cargo, 1);
    step(&mut game, vec![order(1, Deposit(North))], vec![]);
    step(&mut game, vec![], vec![]);
    assert_eq!(game.credits[0], 4);
    assert_eq!(game.units[1].cargo, 0);
}

#[test]
fn simultaneous_kills_draw_but_single_elimination_wins() {
    let mut game = Game::new(Visibility::Full);
    game.units.retain(|unit| unit.kind == UnitKind::Worker);
    game.units[0].position = pos(3, 3);
    game.units[1].position = pos(4, 3);
    game.units[0].hp = 1;
    game.units[1].hp = 1;
    let mut one_attack = game.clone();
    step(
        &mut game,
        vec![order(1, Attack(East))],
        vec![order(3, Attack(West))],
    );
    assert_eq!(step(&mut game, vec![], vec![]), Status::Draw);
    assert_eq!(game.step(&[vec![], vec![]]), Err(StepError::Finished));

    step(&mut one_attack, vec![order(1, Attack(East))], vec![]);
    let status = step(&mut one_attack, vec![], vec![]);
    assert_eq!(status, Status::Won(Zero));
    assert_eq!(status.rewards(), [1, -1]);
    assert!(status.terminated());
}

#[test]
fn relay_pays_only_after_an_uncontested_eighth_tick() {
    let mut game = Game::new(Visibility::Full);
    game.units[1].position = pos(3, 3);
    for _ in 0..7 {
        step(&mut game, vec![], vec![]);
    }
    assert_eq!(game.credits, [3, 3]);
    step(&mut game, vec![], vec![]);
    assert_eq!(game.credits, [4, 3]);
    game.units[3].position = pos(4, 4);
    for _ in 0..8 {
        step(&mut game, vec![], vec![]);
    }
    assert_eq!(game.credits, [4, 3]);
}

#[test]
fn fog_observations_and_masks_ignore_hidden_units_and_credits() {
    let game = Game::new(Visibility::Fog);
    let mut other = game.clone();
    other.credits[1] = 10;
    other.units[3].position = pos(4, 6);
    other.units[3].pending = Some(Pending {
        action: Move(North),
        completes_at: 2,
    });
    assert_eq!(game.observe(Zero), other.observe(Zero));
    assert_eq!(game.legal_actions(Zero), other.legal_actions(Zero));
    assert_eq!(game.source_mask(Zero), other.source_mask(Zero));
    assert_eq!(game.observe(Zero).credits, [Some(3), None]);
    assert_eq!(game.observe(Zero).board[pos(4, 5).index()][0], 1);
    assert_eq!(game.observe(Zero).board[pos(6, 6).index()][2..], [0; 7]);
    other.units[3].position = pos(2, 3);
    assert_ne!(game.observe(Zero), other.observe(Zero));
}

#[test]
fn reset_and_replay_are_deterministic_and_timeout_truncates() {
    let mut game = Game::new(Visibility::Full);
    let mut reordered = game.clone();
    let mut replay = Replay::new(Visibility::Full);
    let first = [
        vec![order(0, Train(Mobile::Worker, North)), order(1, Move(East))],
        vec![order(2, Train(Mobile::Worker, South)), order(3, Move(West))],
    ];
    let mut reversed = first.clone();
    for orders in &mut reversed {
        orders.reverse();
    }
    replay.record(&mut game, first).unwrap();
    reordered.step(&reversed).unwrap();
    assert_eq!(reordered.state_hash(), game.state_hash());
    for _ in 1..MAX_TICKS {
        let orders = [vec![], vec![]];
        reordered.step(&orders).unwrap();
        replay.record(&mut game, orders).unwrap();
        assert_eq!(reordered.state_hash(), game.state_hash());
    }
    assert!(game.status().truncated());
    assert!(!game.status().terminated());
    let encoded = serde_json::to_string(&replay).unwrap();
    let mut decoded: Replay = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.play().unwrap(), game);
    decoded.frames[4].state_hash.clear();
    assert!(matches!(decoded.play(), Err(ReplayError::Diverged(5))));
    game.reset();
    assert_eq!(game, Game::new(Visibility::Full));
}
