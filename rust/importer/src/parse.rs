const OBJECT_SEPARATOR: char = ';';
const FIELD_SEPARATOR: char = ',';
const FIELDS_PER_PAIR: usize = 2;
const HEADER_SEGMENTS: usize = 1;

const KEY_ID: &str = "1";
const KEY_X_PX: &str = "2";
const KEY_Y_PX: &str = "3";
const KEY_FLIP_Y: &str = "5";
const KEY_ROTATION_DEG: &str = "6";
const FLAG_SET: &str = "1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawObject {
    pub id: u32,
    pub x_px: f32,
    pub y_px: f32,
    pub flip_y: bool,
    pub rotation_deg: f32,
}

pub fn parse_objects(decoded: &str) -> Result<Vec<RawObject>, String> {
    decoded
        .split(OBJECT_SEPARATOR)
        .enumerate()
        .skip(HEADER_SEGMENTS)
        .filter(|(_, segment)| !segment.is_empty())
        .map(|(index, segment)| {
            parse_object(segment).map_err(|error| format!("object segment {index}: {error}"))
        })
        .collect()
}

fn parse_object(segment: &str) -> Result<RawObject, String> {
    let fields: Vec<&str> = segment.split(FIELD_SEPARATOR).collect();
    if !fields.len().is_multiple_of(FIELDS_PER_PAIR) {
        return Err(format!("odd number of fields in {segment:?}"));
    }
    let value_of = |key: &str| {
        fields
            .chunks_exact(FIELDS_PER_PAIR)
            .find_map(|pair| match pair {
                [pair_key, value] if *pair_key == key => Some(*value),
                _ => None,
            })
    };
    let required = |key: &str| value_of(key).ok_or_else(|| format!("missing key {key}"));
    let number = |key: &str, text: &str| {
        text.parse::<f32>()
            .map_err(|error| format!("key {key} is not a number ({text:?}): {error}"))
    };
    let id_text = required(KEY_ID)?;
    Ok(RawObject {
        id: id_text
            .parse()
            .map_err(|error| format!("object id is not an integer ({id_text:?}): {error}"))?,
        x_px: number(KEY_X_PX, required(KEY_X_PX)?)?,
        y_px: number(KEY_Y_PX, required(KEY_Y_PX)?)?,
        flip_y: value_of(KEY_FLIP_Y) == Some(FLAG_SET),
        rotation_deg: value_of(KEY_ROTATION_DEG)
            .map_or(Ok(0.0), |text| number(KEY_ROTATION_DEG, text))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_skipped_and_each_object_is_read() {
        let objects =
            parse_objects("kS29,1,kA2,0;1,8,2,525,3,15;1,1,2,45.5,3,15,5,1,6,-90;").unwrap();
        assert_eq!(
            objects,
            vec![
                RawObject {
                    id: 8,
                    x_px: 525.0,
                    y_px: 15.0,
                    flip_y: false,
                    rotation_deg: 0.0
                },
                RawObject {
                    id: 1,
                    x_px: 45.5,
                    y_px: 15.0,
                    flip_y: true,
                    rotation_deg: -90.0
                },
            ]
        );
    }

    #[test]
    fn unrelated_keys_are_ignored() {
        let objects = parse_objects("header;1,8,2,1,3,2,155,1,36,1;").unwrap();
        assert_eq!(objects.len(), 1);
    }

    #[test]
    fn missing_position_is_rejected_with_the_segment_index() {
        let error = parse_objects("header;1,8,2,1;").unwrap_err();
        assert!(error.contains("segment 1"));
        assert!(error.contains("missing key 3"));
    }

    #[test]
    fn malformed_fields_are_rejected() {
        assert!(parse_objects("header;1,8,2;").is_err());
        assert!(parse_objects("header;1,spike,2,1,3,2;").is_err());
        assert!(parse_objects("header;1,8,2,far,3,2;").is_err());
    }
}
