// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::{
    io::{self, Read},
    path::PathBuf,
};

use thiserror::Error;

/// Errors that can occur when resolving extraction input.
#[derive(Debug, Error)]
pub enum InputError {
    #[error(
        "conflicting inputs: provide text as an argument, use --file, or pipe via stdin — not a combination"
    )]
    Conflict,
    #[error(
        "no input provided and stdin is empty: pass text as an argument, pipe text via stdin, or use --file"
    )]
    Empty,
    #[error("stdin can only be read once; remove the duplicate '-' from your --file list")]
    DuplicateStdin,
    #[error("failed to read file '{path}': {source}")]
    FileIo {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("failed to read stdin: {0}")]
    StdinIo(#[source] io::Error),
}

/// Resolves the extraction input from mutually exclusive sources.
///
/// Priority and exclusivity rules:
/// - `positional` (some) **and** non-empty `files` → [`InputError::Conflict`]
/// - `positional` is `Some` → return the string as-is
/// - `files` is non-empty → read each path; `"-"` reads from `stdin_reader`
///   (only once — a second `"-"` yields [`InputError::DuplicateStdin`]); join
///   multiple file contents with `"\n\n"`
/// - both empty → read all of `stdin_reader`
///
/// Returns [`InputError::Empty`] if the resolved string is blank.
///
/// `stdin_reader` is a generic `R: Read` so callers (and tests) can inject any
/// source instead of hard-coding `std::io::stdin()`.
pub fn read_extraction_input<R: Read>(
    positional: Option<String>,
    files: &[PathBuf],
    stdin_reader: R,
) -> Result<String, InputError> {
    if positional.is_some() && !files.is_empty() {
        return Err(InputError::Conflict);
    }

    let text = if let Some(t) = positional {
        t
    } else if !files.is_empty() {
        read_files(files, stdin_reader)?
    } else {
        read_stdin(stdin_reader)?
    };

    if text.trim().is_empty() {
        return Err(InputError::Empty);
    }

    Ok(text)
}

fn read_files<R: Read>(files: &[PathBuf], stdin_reader: R) -> Result<String, InputError> {
    let mut parts = Vec::with_capacity(files.len());
    let mut stdin_used = false;
    let mut stdin_reader = Some(stdin_reader);

    for path in files {
        if path == std::path::Path::new("-") {
            if stdin_used {
                return Err(InputError::DuplicateStdin);
            }
            stdin_used = true;
            let reader = stdin_reader.take().expect("stdin reader already consumed");
            parts.push(read_stdin(reader)?);
        } else {
            let content = std::fs::read_to_string(path).map_err(|source| InputError::FileIo {
                path: path.display().to_string(),
                source,
            })?;
            parts.push(content);
        }
    }

    Ok(parts.join("\n\n"))
}

fn read_stdin<R: Read>(mut reader: R) -> Result<String, InputError> {
    let mut buf = String::new();
    reader
        .read_to_string(&mut buf)
        .map_err(InputError::StdinIo)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;

    use super::{InputError, read_extraction_input};

    fn no_stdin() -> &'static [u8] {
        b""
    }

    #[test]
    fn test_positional_text_is_returned_as_is() {
        let result =
            read_extraction_input(Some("hello world".to_string()), &[], no_stdin()).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_positional_and_files_is_conflict() {
        let err = read_extraction_input(
            Some("text".to_string()),
            &[PathBuf::from("a.txt")],
            no_stdin(),
        )
        .unwrap_err();
        assert!(matches!(err, InputError::Conflict));
    }

    #[test]
    fn test_single_file_is_read() {
        let mut f = NamedTempFile::new().unwrap();
        use std::io::Write as _;
        write!(f, "file content").unwrap();
        let result = read_extraction_input(None, &[f.path().to_path_buf()], no_stdin()).unwrap();
        assert_eq!(result, "file content");
    }

    #[test]
    fn test_multiple_files_are_joined_with_double_newline() {
        let mut f1 = NamedTempFile::new().unwrap();
        let mut f2 = NamedTempFile::new().unwrap();
        use std::io::Write as _;
        write!(f1, "first").unwrap();
        write!(f2, "second").unwrap();
        let result = read_extraction_input(
            None,
            &[f1.path().to_path_buf(), f2.path().to_path_buf()],
            no_stdin(),
        )
        .unwrap();
        assert_eq!(result, "first\n\nsecond");
    }

    #[test]
    fn test_missing_file_returns_file_io_error() {
        let err = read_extraction_input(
            None,
            &[PathBuf::from("/nonexistent/path/to/file.txt")],
            no_stdin(),
        )
        .unwrap_err();
        assert!(matches!(err, InputError::FileIo { .. }));
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn test_dash_reads_from_stdin_reader() {
        let stdin_data = b"stdin content";
        let result =
            read_extraction_input(None, &[PathBuf::from("-")], stdin_data.as_ref()).unwrap();
        assert_eq!(result, "stdin content");
    }

    #[test]
    fn test_duplicate_dash_is_error() {
        let err = read_extraction_input(
            None,
            &[PathBuf::from("-"), PathBuf::from("-")],
            b"data".as_ref(),
        )
        .unwrap_err();
        assert!(matches!(err, InputError::DuplicateStdin));
    }

    #[test]
    fn test_no_args_reads_from_stdin() {
        let stdin_data = b"piped in";
        let result = read_extraction_input(None, &[], stdin_data.as_ref()).unwrap();
        assert_eq!(result, "piped in");
    }

    #[test]
    fn test_empty_positional_is_error() {
        let err = read_extraction_input(Some("   ".to_string()), &[], no_stdin()).unwrap_err();
        assert!(matches!(err, InputError::Empty));
    }

    #[test]
    fn test_empty_file_is_error() {
        let f = NamedTempFile::new().unwrap();
        let err = read_extraction_input(None, &[f.path().to_path_buf()], no_stdin()).unwrap_err();
        assert!(matches!(err, InputError::Empty));
    }

    #[test]
    fn test_empty_stdin_is_error() {
        let err = read_extraction_input(None, &[], b"".as_ref()).unwrap_err();
        assert!(matches!(err, InputError::Empty));
    }
}
