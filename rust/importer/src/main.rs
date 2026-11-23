mod convert;
mod decode;
mod parse;
mod table;

use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;
use std::{env, fs};

use convert::import_level;
use decode::decode_level_string;
use parse::parse_objects;

const USAGE: &str = "usage: geo-dash-importer <level string file> <output level json>";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args, &mut io::stdout()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String], out: &mut impl Write) -> Result<(), String> {
    let [source, output] = args else {
        return Err(USAGE.to_owned());
    };
    let encoded =
        fs::read_to_string(source).map_err(|error| format!("cannot read {source}: {error}"))?;
    let objects = parse_objects(&decode_level_string(&encoded)?)?;
    let (level, report) = import_level(&objects)?;
    let json = serde_json::to_string(&level).map_err(|error| error.to_string())?;
    if let Some(directory) = Path::new(output).parent() {
        fs::create_dir_all(directory)
            .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    }
    fs::write(output, json).map_err(|error| format!("cannot write {output}: {error}"))?;
    writeln!(out, "{source} -> {output}: {report}").map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use decode::encode_level_string;
    use geo_dash_engine::level::Level;

    const LEVEL_TEXT: &str = "kS29,1;1,8,2,525,3,15;1,1,2,300,3,15;";
    const SPIKES: usize = 1;
    const SOLIDS: usize = 1;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = env::temp_dir().join(format!("geo_dash_importer_{name}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn imported_level_file_loads_back_as_the_same_level() {
        let dir = scratch_dir("round_trip");
        let source = dir.join("level.txt");
        let output = dir.join("out").join("level.json");
        fs::write(&source, encode_level_string(LEVEL_TEXT)).unwrap();
        let mut report = Vec::new();
        run(
            &[
                source.to_string_lossy().into_owned(),
                output.to_string_lossy().into_owned(),
            ],
            &mut report,
        )
        .unwrap();
        let level: Level = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
        assert_eq!(level.spikes().len(), SPIKES);
        assert_eq!(level.solids().len(), SOLIDS);
        assert!(String::from_utf8(report).unwrap().contains("1 spikes"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wrong_argument_count_prints_the_usage() {
        assert_eq!(run(&[], &mut Vec::new()).unwrap_err(), USAGE);
    }

    #[test]
    fn unreadable_source_names_the_path() {
        let error = run(
            &["missing.txt".to_owned(), "out.json".to_owned()],
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(error.contains("missing.txt"));
    }
}
