//! Deterministic, simultaneous-order RTS simulation for the Relay scenario.

mod replay;
pub use replay::{Replay, ReplayError, ReplayFrame};

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const WIDTH: usize = 8;
pub const CELLS: usize = WIDTH * WIDTH;
pub const ACTIONS: usize = 25;
pub const BOARD_FEATURES: usize = 9;
pub const MAX_TICKS: u16 = 512;
pub const SCENARIO: &str = "relay-8x8-v1";

pub type UnitId = u16;
pub type LegalMask = [[bool; ACTIONS]; CELLS];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Player {
    Zero,
    One,
}

impl Player {
    pub const ALL: [Self; 2] = [Self::Zero, Self::One];

    pub const fn index(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }

    pub const fn opponent(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Full,
    Fog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub x: u8,
    pub y: u8,
}

impl Position {
    pub const fn index(self) -> usize {
        self.y as usize * WIDTH + self.x as usize
    }

    fn from_index(index: usize) -> Self {
        Self {
            x: (index % WIDTH) as u8,
            y: (index / WIDTH) as u8,
        }
    }

    fn neighbor(self, direction: Direction) -> Option<Self> {
        let (dx, dy) = match direction {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        };
        let x = i16::from(self.x) + dx;
        let y = i16::from(self.y) + dy;
        ((0..WIDTH as i16).contains(&x) && (0..WIDTH as i16).contains(&y)).then_some(Self {
            x: x as u8,
            y: y as u8,
        })
    }

    fn on_relay(self) -> bool {
        (3..=4).contains(&self.x) && (3..=4).contains(&self.y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    const ALL: [Self; 4] = [Self::North, Self::East, Self::South, Self::West];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mobile {
    Worker,
    Scout,
}

impl Mobile {
    fn cost(self) -> u16 {
        match self {
            Self::Worker => 3,
            Self::Scout => 4,
        }
    }

    fn kind(self) -> UnitKind {
        match self {
            Self::Worker => UnitKind::Worker,
            Self::Scout => UnitKind::Scout,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitKind {
    Base,
    Worker,
    Scout,
}

impl UnitKind {
    fn max_hp(self) -> u8 {
        match self {
            Self::Base => 12,
            Self::Worker => 3,
            Self::Scout => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Wait,
    Move(Direction),
    Gather(Direction),
    Deposit(Direction),
    Train(Mobile, Direction),
    Attack(Direction),
}

impl Action {
    pub fn from_index(index: usize) -> Option<Self> {
        if index == 0 {
            return Some(Self::Wait);
        }
        let direction = Direction::ALL[(index - 1) % 4];
        match (index - 1) / 4 {
            0 => Some(Self::Move(direction)),
            1 => Some(Self::Gather(direction)),
            2 => Some(Self::Deposit(direction)),
            3 => Some(Self::Train(Mobile::Worker, direction)),
            4 => Some(Self::Train(Mobile::Scout, direction)),
            5 => Some(Self::Attack(direction)),
            _ => None,
        }
    }

    fn direction(self) -> Option<Direction> {
        match self {
            Self::Wait => None,
            Self::Move(d)
            | Self::Gather(d)
            | Self::Deposit(d)
            | Self::Train(_, d)
            | Self::Attack(d) => Some(d),
        }
    }

    fn duration(self, kind: UnitKind) -> u16 {
        match self {
            Self::Wait => 1,
            Self::Move(_) if kind == UnitKind::Scout => 1,
            Self::Train(Mobile::Worker, _) => 6,
            Self::Train(Mobile::Scout, _) => 8,
            Self::Move(_) | Self::Gather(_) | Self::Deposit(_) | Self::Attack(_) => 2,
        }
    }

    fn feature(self) -> i32 {
        match self {
            Self::Wait => 0,
            Self::Move(_) => 1,
            Self::Gather(_) => 2,
            Self::Deposit(_) => 3,
            Self::Train(Mobile::Worker, _) => 4,
            Self::Train(Mobile::Scout, _) => 5,
            Self::Attack(_) => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    pub unit: UnitId,
    pub action: Action,
}

pub type JointOrders = [Vec<Order>; 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Pending {
    pub action: Action,
    pub completes_at: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Unit {
    pub id: UnitId,
    pub owner: Player,
    pub kind: UnitKind,
    pub position: Position,
    pub hp: u8,
    pub cargo: u8,
    pub pending: Option<Pending>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct Deposit {
    position: Position,
    ore: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Status {
    Running,
    Won(Player),
    Draw,
    Truncated,
}

impl Status {
    pub fn rewards(self) -> [i8; 2] {
        match self {
            Self::Won(Player::Zero) => [1, -1],
            Self::Won(Player::One) => [-1, 1],
            _ => [0, 0],
        }
    }

    pub fn terminated(self) -> bool {
        matches!(self, Self::Won(_) | Self::Draw)
    }

    pub fn truncated(self) -> bool {
        self == Self::Truncated
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepError {
    Finished,
    DuplicateOrder { player: Player, unit: UnitId },
    InvalidChoice { cell: usize, choice: u8 },
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finished => write!(f, "match is over; reset before stepping"),
            Self::DuplicateOrder { player, unit } => {
                write!(
                    f,
                    "player {} submitted multiple orders for unit {unit}",
                    player.index()
                )
            }
            Self::InvalidChoice { cell, choice } => {
                write!(
                    f,
                    "action choice {choice} at cell {cell} is outside 0..{ACTIONS}"
                )
            }
        }
    }
}

impl Error for StepError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// Wall, relay, visible, relative owner, kind, HP, ore/cargo, action, ticks left.
    pub board: [[i32; BOARD_FEATURES]; CELLS],
    /// Player's own credits, then the opponent's credits when visible.
    pub credits: [Option<u16>; 2],
    pub tick: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Game {
    visibility: Visibility,
    tick: u16,
    walls: u64,
    deposits: [Deposit; 2],
    credits: [u16; 2],
    units: Vec<Unit>,
    next_id: UnitId,
    status: Status,
}

#[derive(Clone, Copy)]
enum Completion {
    Move(Position),
    Gather(usize),
    Deposit,
    Spawn(Mobile, Position),
    Attack(usize),
    FailedSpawn(Mobile),
}

impl Game {
    pub fn new(visibility: Visibility) -> Self {
        let mut game = Self {
            visibility,
            tick: 0,
            walls: 0,
            deposits: [
                Deposit {
                    position: Position { x: 1, y: 3 },
                    ore: 12,
                },
                Deposit {
                    position: Position { x: 6, y: 4 },
                    ore: 12,
                },
            ],
            credits: [3, 3],
            units: Vec::with_capacity(CELLS),
            next_id: 0,
            status: Status::Running,
        };
        for (x, y) in [(3, 2), (2, 4), (4, 5), (5, 3)] {
            game.walls |= 1 << (Position { x, y }).index();
        }
        for (player, x, y, worker_x) in [(Player::Zero, 1, 1, 2), (Player::One, 6, 6, 5)] {
            game.add_unit(player, UnitKind::Base, Position { x, y });
            game.add_unit(player, UnitKind::Worker, Position { x: worker_x, y });
        }
        game
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.visibility);
    }

    pub fn visibility(&self) -> Visibility {
        self.visibility
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn tick(&self) -> u16 {
        self.tick
    }

    /// Full state for trusted simulators and replay tools.
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    fn add_unit(&mut self, owner: Player, kind: UnitKind, position: Position) {
        self.units.push(Unit {
            id: self.next_id,
            owner,
            kind,
            position,
            hp: kind.max_hp(),
            cargo: 0,
            pending: None,
        });
        self.next_id += 1;
    }

    fn wall(&self, position: Position) -> bool {
        self.walls & (1 << position.index()) != 0
    }

    fn occupancy(&self) -> [Option<usize>; CELLS] {
        let mut grid = [None; CELLS];
        for (index, unit) in self.units.iter().enumerate() {
            grid[unit.position.index()] = Some(index);
        }
        grid
    }

    fn ore_at(&self, position: Position) -> Option<usize> {
        self.deposits
            .iter()
            .position(|deposit| deposit.position == position && deposit.ore > 0)
    }

    fn empty(&self, grid: &[Option<usize>; CELLS], position: Position) -> bool {
        !self.wall(position) && grid[position.index()].is_none() && self.ore_at(position).is_none()
    }

    fn visible(&self, player: Player, position: Position) -> bool {
        self.visibility == Visibility::Full
            || self.units.iter().any(|unit| {
                unit.owner == player
                    && unit.position.x.abs_diff(position.x) + unit.position.y.abs_diff(position.y)
                        <= 3
            })
    }

    pub fn observe(&self, player: Player) -> Observation {
        let mut board = std::array::from_fn(|index| {
            let position = Position::from_index(index);
            [
                self.wall(position).into(),
                position.on_relay().into(),
                self.visible(player, position).into(),
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        });
        for deposit in &self.deposits {
            let cell = &mut board[deposit.position.index()];
            if cell[2] == 1 && deposit.ore > 0 {
                cell[4] = 1;
                cell[5] = 1;
                cell[6] = deposit.ore.into();
            }
        }
        for unit in &self.units {
            let cell = &mut board[unit.position.index()];
            if cell[2] == 0 {
                continue;
            }
            cell[3] = if unit.owner == player { 1 } else { 2 };
            cell[4] = match unit.kind {
                UnitKind::Base => 2,
                UnitKind::Worker => 3,
                UnitKind::Scout => 4,
            };
            cell[5] = unit.hp.into();
            cell[6] = unit.cargo.into();
            if let Some(pending) = unit.pending {
                cell[7] = pending.action.feature();
                cell[8] = (pending.completes_at - self.tick).into();
            }
        }
        Observation {
            board,
            credits: [
                Some(self.credits[player.index()]),
                (self.visibility == Visibility::Full)
                    .then(|| self.credits[player.opponent().index()]),
            ],
            tick: self.tick,
        }
    }

    /// Row-major per-cell action mask. Choice 0 is valid even on non-source cells.
    pub fn legal_actions(&self, player: Player) -> LegalMask {
        let mut mask = [[false; ACTIONS]; CELLS];
        for row in &mut mask {
            row[0] = true;
        }
        if self.status != Status::Running {
            return mask;
        }
        let grid = self.occupancy();
        for unit in self
            .units
            .iter()
            .filter(|u| u.owner == player && u.pending.is_none())
        {
            for (index, allowed) in mask[unit.position.index()].iter_mut().enumerate().skip(1) {
                *allowed = self.available(&grid, unit, Action::from_index(index).unwrap());
            }
        }
        mask
    }

    pub fn source_mask(&self, player: Player) -> [bool; CELLS] {
        let mut mask = [false; CELLS];
        if self.status == Status::Running {
            for unit in self
                .units
                .iter()
                .filter(|u| u.owner == player && u.pending.is_none())
            {
                mask[unit.position.index()] = true;
            }
        }
        mask
    }

    fn available(&self, grid: &[Option<usize>; CELLS], unit: &Unit, action: Action) -> bool {
        if action == Action::Wait {
            return true;
        }
        let Some(target) = action.direction().and_then(|d| unit.position.neighbor(d)) else {
            return false;
        };
        let occupant = grid[target.index()].map(|index| &self.units[index]);
        match action {
            Action::Move(_) => unit.kind != UnitKind::Base && self.empty(grid, target),
            Action::Gather(_) => {
                unit.kind == UnitKind::Worker && unit.cargo == 0 && self.ore_at(target).is_some()
            }
            Action::Deposit(_) => {
                unit.kind == UnitKind::Worker
                    && unit.cargo > 0
                    && occupant.is_some_and(|u| u.kind == UnitKind::Base && u.owner == unit.owner)
            }
            Action::Train(kind, _) => {
                unit.kind == UnitKind::Base
                    && self.credits[unit.owner.index()] >= kind.cost()
                    && self.empty(grid, target)
            }
            Action::Attack(_) => {
                unit.kind != UnitKind::Base && occupant.is_some_and(|u| u.owner != unit.owner)
            }
            Action::Wait => true,
        }
    }

    /// Convert two row-major action grids into stable-ID orders.
    pub fn dense_orders(&self, choices: &[[u8; CELLS]; 2]) -> Result<JointOrders, StepError> {
        for grid in choices {
            if let Some((cell, &choice)) = grid
                .iter()
                .enumerate()
                .find(|(_, value)| **value as usize >= ACTIONS)
            {
                return Err(StepError::InvalidChoice { cell, choice });
            }
        }
        let mut orders = [Vec::new(), Vec::new()];
        for unit in &self.units {
            let player = unit.owner.index();
            let choice = choices[player][unit.position.index()];
            if choice != 0 && unit.pending.is_none() {
                orders[player].push(Order {
                    unit: unit.id,
                    action: Action::from_index(choice as usize).unwrap(),
                });
            }
        }
        Ok(orders)
    }

    pub fn step(&mut self, orders: &JointOrders) -> Result<Status, StepError> {
        if self.status != Status::Running {
            return Err(StepError::Finished);
        }
        let mut requests = [Vec::new(), Vec::new()];
        for player in Player::ALL {
            let list = &mut requests[player.index()];
            list.extend_from_slice(&orders[player.index()]);
            list.sort_unstable_by_key(|order| order.unit);
            if let Some(duplicate) = list.windows(2).find(|pair| pair[0].unit == pair[1].unit) {
                return Err(StepError::DuplicateOrder {
                    player,
                    unit: duplicate[0].unit,
                });
            }
        }

        let grid = self.occupancy();
        let mut accepted = [None; CELLS];
        for (index, unit) in self
            .units
            .iter()
            .enumerate()
            .filter(|(_, u)| u.pending.is_none())
        {
            let list = &requests[unit.owner.index()];
            let action = list
                .binary_search_by_key(&unit.id, |order| order.unit)
                .ok()
                .map(|i| list[i].action)
                .filter(|&action| self.available(&grid, unit, action))
                .unwrap_or(Action::Wait);
            accepted[index] = Some(Pending {
                action,
                completes_at: self.tick + action.duration(unit.kind),
            });
        }
        for (unit, pending) in self.units.iter_mut().zip(accepted) {
            if let Some(pending) = pending {
                if let Action::Train(kind, _) = pending.action {
                    self.credits[unit.owner.index()] -= kind.cost();
                }
                unit.pending = Some(pending);
            }
        }

        self.tick += 1;
        self.complete_orders(&grid);
        if self.tick % 8 == 0 {
            self.pay_relay();
        }
        let mut alive = [false; 2];
        for unit in &self.units {
            alive[unit.owner.index()] = true;
        }
        self.status = match alive {
            [true, false] => Status::Won(Player::Zero),
            [false, true] => Status::Won(Player::One),
            [false, false] => Status::Draw,
            [true, true] if self.tick == MAX_TICKS => Status::Truncated,
            [true, true] => Status::Running,
        };
        Ok(self.status)
    }

    fn completion(
        &self,
        grid: &[Option<usize>; CELLS],
        unit: &Unit,
        action: Action,
    ) -> Option<Completion> {
        let target = unit.position.neighbor(action.direction()?)?;
        let occupant = grid[target.index()];
        match action {
            Action::Move(_) if self.empty(grid, target) => Some(Completion::Move(target)),
            Action::Gather(_) => self.ore_at(target).map(Completion::Gather),
            Action::Deposit(_)
                if occupant.is_some_and(|i| {
                    let other = &self.units[i];
                    other.kind == UnitKind::Base && other.owner == unit.owner
                }) =>
            {
                Some(Completion::Deposit)
            }
            Action::Train(kind, _) if self.empty(grid, target) => {
                Some(Completion::Spawn(kind, target))
            }
            Action::Train(kind, _) => Some(Completion::FailedSpawn(kind)),
            Action::Attack(_) => occupant
                .filter(|&i| self.units[i].owner != unit.owner)
                .map(Completion::Attack),
            _ => None,
        }
    }

    fn complete_orders(&mut self, grid: &[Option<usize>; CELLS]) {
        // All completion intents are derived before any changes to positions or health.
        let mut completions = [None; CELLS];
        let mut claims = [0u8; CELLS];
        let mut gather = [0u8; 2];
        for (index, unit) in self.units.iter().enumerate() {
            if let Some(pending) = unit.pending.filter(|p| p.completes_at == self.tick) {
                let completion = self.completion(grid, unit, pending.action);
                match completion {
                    Some(Completion::Move(target) | Completion::Spawn(_, target)) => {
                        claims[target.index()] += 1
                    }
                    Some(Completion::Gather(deposit)) => gather[deposit] += 1,
                    _ => {}
                }
                completions[index] = completion;
            }
        }
        let supply = self.deposits.map(|deposit| deposit.ore);
        let mut damage = [0u8; CELLS];
        let mut spawns = [None; CELLS];
        for (index, unit) in self.units.iter_mut().enumerate() {
            if unit.pending.is_some_and(|p| p.completes_at == self.tick) {
                unit.pending = None;
            }
            let bank = &mut self.credits[unit.owner.index()];
            match completions[index] {
                Some(Completion::Move(target)) if claims[target.index()] == 1 => {
                    unit.position = target
                }
                Some(Completion::Gather(deposit)) if gather[deposit] <= supply[deposit] => {
                    self.deposits[deposit].ore -= 1;
                    unit.cargo = 1;
                }
                Some(Completion::Deposit) => {
                    *bank += u16::from(unit.cargo);
                    unit.cargo = 0;
                }
                Some(Completion::Spawn(kind, target)) if claims[target.index()] == 1 => {
                    spawns[target.index()] = Some((unit.owner, kind, target));
                }
                Some(Completion::Spawn(kind, _) | Completion::FailedSpawn(kind)) => {
                    *bank += kind.cost()
                }
                Some(Completion::Attack(target)) => {
                    damage[target] += if unit.kind == UnitKind::Scout { 2 } else { 1 };
                }
                _ => {}
            }
        }
        for (unit, hit) in self.units.iter_mut().zip(damage) {
            unit.hp = unit.hp.saturating_sub(hit);
        }
        // A base killed before its training completes releases the prepaid credits.
        for unit in self.units.iter().filter(|unit| unit.hp == 0) {
            if let Some(Pending {
                action: Action::Train(kind, _),
                ..
            }) = unit.pending
            {
                self.credits[unit.owner.index()] += kind.cost();
            }
        }
        self.units.retain(|unit| unit.hp > 0);
        for (owner, kind, target) in spawns.into_iter().flatten() {
            self.add_unit(owner, kind.kind(), target);
        }
    }

    fn pay_relay(&mut self) {
        let mut controls = [false; 2];
        for unit in &self.units {
            if unit.kind != UnitKind::Base && unit.position.on_relay() {
                controls[unit.owner.index()] = true;
            }
        }
        if controls[0] != controls[1] {
            self.credits[usize::from(controls[1])] += 1;
        }
    }
}

#[cfg(test)]
mod tests;
