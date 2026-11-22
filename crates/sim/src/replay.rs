use crate::{Game, JointOrders, SCENARIO, Status, StepError, Visibility};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub orders: JointOrders,
    pub state_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replay {
    pub version: u32,
    pub scenario: String,
    pub visibility: Visibility,
    pub frames: Vec<ReplayFrame>,
}

#[derive(Debug)]
pub enum ReplayError {
    Unsupported,
    OutOfSync,
    Step(StepError),
    Diverged(u16),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => write!(f, "unsupported replay version or scenario"),
            Self::OutOfSync => write!(f, "match does not agree with recorded replay"),
            Self::Step(error) => error.fmt(f),
            Self::Diverged(tick) => write!(f, "replay state hash differs at tick {tick}"),
        }
    }
}

impl Error for ReplayError {}

impl Game {
    /// Hash trusted state, including hidden units and the next stable unit ID.
    pub fn state_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("game state serializes to JSON");
        format!("{:x}", Sha256::digest(bytes))
    }
}

impl Replay {
    pub fn new(visibility: Visibility) -> Self {
        Self {
            version: VERSION,
            scenario: SCENARIO.to_owned(),
            visibility,
            frames: Vec::new(),
        }
    }

    fn validate_header(&self) -> Result<(), ReplayError> {
        if self.version != VERSION || self.scenario != SCENARIO {
            return Err(ReplayError::Unsupported);
        }
        Ok(())
    }

    pub fn record(&mut self, game: &mut Game, orders: JointOrders) -> Result<Status, ReplayError> {
        self.validate_header()?;
        if game.visibility() != self.visibility || usize::from(game.tick()) != self.frames.len() {
            return Err(ReplayError::OutOfSync);
        }
        let before_hash = self.frames.last().map_or_else(
            || Game::new(self.visibility).state_hash(),
            |frame| frame.state_hash.clone(),
        );
        if before_hash != game.state_hash() {
            return Err(ReplayError::OutOfSync);
        }
        let status = game.step(&orders).map_err(ReplayError::Step)?;
        self.frames.push(ReplayFrame {
            orders,
            state_hash: game.state_hash(),
        });
        Ok(status)
    }

    pub fn play(&self) -> Result<Game, ReplayError> {
        self.validate_header()?;
        let mut game = Game::new(self.visibility);
        for frame in &self.frames {
            game.step(&frame.orders).map_err(ReplayError::Step)?;
            if game.state_hash() != frame.state_hash {
                return Err(ReplayError::Diverged(game.tick()));
            }
        }
        Ok(game)
    }
}
