use std::io::Read;

use base64::Engine;
use base64::alphabet::URL_SAFE;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use flate2::read::GzDecoder;

const URL_SAFE_ANY_PADDING: GeneralPurpose = GeneralPurpose::new(
    &URL_SAFE,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

pub fn decode_level_string(encoded: &str) -> Result<String, String> {
    let compressed = URL_SAFE_ANY_PADDING
        .decode(encoded.trim())
        .map_err(|error| format!("level string is not base64: {error}"))?;
    let mut decoded = String::new();
    GzDecoder::new(compressed.as_slice())
        .read_to_string(&mut decoded)
        .map_err(|error| format!("level string is not gzipped text: {error}"))?;
    Ok(decoded)
}

#[cfg(test)]
pub fn encode_level_string(decoded: &str) -> String {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(decoded.as_bytes()).unwrap();
    URL_SAFE_ANY_PADDING.encode(encoder.finish().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVEL_TEXT: &str = "kS29,1;1,8,2,525,3,15;";

    #[test]
    fn decoding_inverts_encoding() {
        assert_eq!(
            decode_level_string(&encode_level_string(LEVEL_TEXT)).unwrap(),
            LEVEL_TEXT
        );
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        let encoded = format!("{}\n", encode_level_string(LEVEL_TEXT));
        assert_eq!(decode_level_string(&encoded).unwrap(), LEVEL_TEXT);
    }

    #[test]
    fn text_that_is_not_base64_is_rejected() {
        assert!(decode_level_string("not base64!").is_err());
    }

    #[test]
    fn base64_that_is_not_gzip_is_rejected() {
        let encoded = URL_SAFE_ANY_PADDING.encode(LEVEL_TEXT);
        assert!(decode_level_string(&encoded).is_err());
    }
}
