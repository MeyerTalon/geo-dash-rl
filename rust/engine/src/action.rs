use std::fmt;

const RELEASE_INDEX: u8 = 0;
const HOLD_INDEX: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Release,
    Hold,
}

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidActionIndex(pub u8);

impl fmt::Display for InvalidActionIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "action index {} is not {RELEASE_INDEX} (release) or {HOLD_INDEX} (hold)",
            self.0
        )
    }
}

impl std::error::Error for InvalidActionIndex {}

impl TryFrom<u8> for Action {
    type Error = InvalidActionIndex;

    fn try_from(index: u8) -> Result<Self, Self::Error> {
        match index {
            RELEASE_INDEX => Ok(Self::Release),
            HOLD_INDEX => Ok(Self::Hold),
            other => Err(InvalidActionIndex(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_map_to_actions() {
        assert_eq!(Action::try_from(0), Ok(Action::Release));
        assert_eq!(Action::try_from(1), Ok(Action::Hold));
    }

    #[test]
    fn unknown_index_is_rejected() {
        assert_eq!(Action::try_from(2), Err(InvalidActionIndex(2)));
    }
}
