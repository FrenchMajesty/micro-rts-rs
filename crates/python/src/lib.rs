use micro_rts_sim::{
    ACTIONS, BOARD_FEATURES, CELLS, Game, JointOrders, MAX_TICKS, Player, Replay, Status,
    Visibility, WIDTH,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

type RawObservation = (Vec<i32>, [i32; 2], u16, Vec<u8>, Vec<u8>);
type RawViews = [RawObservation; 2];
type RawStep = (RawViews, [i8; 2], bool, bool);
type Choices = [[u8; CELLS]; 2];

fn visibility(fog: bool) -> Visibility {
    if fog {
        Visibility::Fog
    } else {
        Visibility::Full
    }
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

struct Match {
    game: Game,
    replay: Option<Replay>,
}

impl Match {
    fn new(fog: bool, record: bool) -> Self {
        let mode = visibility(fog);
        Self {
            game: Game::new(mode),
            replay: record.then(|| Replay::new(mode)),
        }
    }

    fn reset(&mut self) -> RawViews {
        self.game.reset();
        if self.replay.is_some() {
            self.replay = Some(Replay::new(self.game.visibility()));
        }
        self.views()
    }

    fn views(&self) -> RawViews {
        Player::ALL.map(|player| {
            let view = self.game.observe(player);
            (
                view.board.into_iter().flatten().collect(),
                view.credits.map(|v| v.map_or(-1, i32::from)),
                view.tick,
                self.game
                    .legal_actions(player)
                    .into_iter()
                    .flatten()
                    .map(u8::from)
                    .collect(),
                self.game
                    .source_mask(player)
                    .into_iter()
                    .map(u8::from)
                    .collect(),
            )
        })
    }

    fn orders(&self, choices: &Choices) -> PyResult<JointOrders> {
        if self.game.status() != Status::Running {
            return Err(value_error("match is over; reset before stepping"));
        }
        self.game.dense_orders(choices).map_err(value_error)
    }

    fn apply(&mut self, orders: JointOrders) -> PyResult<RawStep> {
        let status = if let Some(replay) = &mut self.replay {
            replay.record(&mut self.game, orders).map_err(value_error)?
        } else {
            self.game.step(&orders).map_err(value_error)?
        };
        Ok((
            self.views(),
            status.rewards(),
            status.terminated(),
            status.truncated(),
        ))
    }
}

#[pyclass(module = "micro_rts_rs._native")]
struct CoreMatch(Match);

#[pymethods]
impl CoreMatch {
    #[new]
    #[pyo3(signature = (fog=false, record_replay=false))]
    fn new(fog: bool, record_replay: bool) -> Self {
        Self(Match::new(fog, record_replay))
    }

    fn reset(&mut self) -> RawViews {
        self.0.reset()
    }

    fn observe(&self) -> RawViews {
        self.0.views()
    }

    fn step(&mut self, py: Python<'_>, choices: Choices) -> PyResult<RawStep> {
        py.detach(|| {
            let orders = self.0.orders(&choices)?;
            self.0.apply(orders)
        })
    }

    fn replay_json(&self) -> PyResult<String> {
        let replay = self
            .0
            .replay
            .as_ref()
            .ok_or_else(|| value_error("replay recording is disabled"))?;
        serde_json::to_string(replay).map_err(value_error)
    }
}

#[pyclass(module = "micro_rts_rs._native")]
struct CoreBatch(Vec<Match>);

#[pymethods]
impl CoreBatch {
    #[new]
    #[pyo3(signature = (size, fog=false))]
    fn new(size: usize, fog: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(value_error("batch size must be positive"));
        }
        Ok(Self((0..size).map(|_| Match::new(fog, false)).collect()))
    }

    #[pyo3(signature = (index=None))]
    fn reset(&mut self, index: Option<usize>) -> PyResult<Vec<RawViews>> {
        match index {
            Some(index) => self
                .0
                .get_mut(index)
                .map(|game| vec![game.reset()])
                .ok_or_else(|| PyIndexError::new_err("match index outside batch")),
            None => Ok(self.0.iter_mut().map(Match::reset).collect()),
        }
    }

    fn step(&mut self, py: Python<'_>, choices: Vec<Choices>) -> PyResult<Vec<RawStep>> {
        py.detach(|| {
            if choices.len() != self.0.len() {
                return Err(value_error("expected one pair of action grids per match"));
            }
            let orders: Vec<_> = self
                .0
                .iter()
                .zip(&choices)
                .map(|(game, choices)| game.orders(choices))
                .collect::<PyResult<_>>()?;
            self.0
                .iter_mut()
                .zip(orders)
                .map(|(game, orders)| game.apply(orders))
                .collect()
        })
    }
}

#[pyfunction]
fn verify_replay(document: &str) -> PyResult<(u16, String)> {
    let replay: Replay = serde_json::from_str(document).map_err(value_error)?;
    let game = replay.play().map_err(value_error)?;
    Ok((game.tick(), game.state_hash()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CoreMatch>()?;
    module.add_class::<CoreBatch>()?;
    module.add_function(wrap_pyfunction!(verify_replay, module)?)?;
    module.add("WIDTH", WIDTH)?;
    module.add("CELLS", CELLS)?;
    module.add("ACTIONS", ACTIONS)?;
    module.add("BOARD_FEATURES", BOARD_FEATURES)?;
    module.add("MAX_TICKS", MAX_TICKS)?;
    Ok(())
}
