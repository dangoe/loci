// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::num::NonZeroUsize;

/// Splits text into a sequence of chunks suitable for downstream processing.
pub trait Chunker: Send + Sync {
    fn chunk(&self, input: &str) -> Vec<String>;
}

/// A [`Chunker`] that splits at paragraph and sentence boundaries using
/// **character counts** for size limits.
///
/// When a boundary-aligned segment still exceeds `chunk_size`, it falls back
/// to word-level splitting, finishing the current word before cutting so no
/// word is ever broken mid-character.
pub struct SentenceAwareChunker {
    /// Target maximum number of characters per chunk. The splitter finishes
    /// the current word before cutting, so actual chunks may be slightly
    /// larger.
    chunk_size: NonZeroUsize,
    /// Characters of overlap between consecutive chunks. Clamped to
    /// `chunk_size - 1`. The overlap region always starts at a word boundary
    /// so it never begins mid-word.
    overlap_size: usize,
}

impl SentenceAwareChunker {
    /// Creates a new [`SentenceAwareChunker`] with the given `chunk_size` and `overlap_size`.
    pub fn new(chunk_size: NonZeroUsize, overlap_size: usize) -> Self {
        Self {
            chunk_size,
            overlap_size,
        }
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size.get()
    }

    /// Splits `text` into a flat list of segments at paragraph then sentence
    /// boundaries. Blank lines (one or more) are treated as paragraph separators.
    fn split_into_segments(input: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut paragraph_lines: Vec<&str> = Vec::new();

        for line in input.lines() {
            if line.trim().is_empty() {
                if !paragraph_lines.is_empty() {
                    segments.extend(Self::split_into_sentences(&paragraph_lines.join(" ")));
                    paragraph_lines.clear();
                }
            } else {
                paragraph_lines.push(line);
            }
        }

        if !paragraph_lines.is_empty() {
            segments.extend(Self::split_into_sentences(&paragraph_lines.join(" ")));
        }

        segments
    }

    fn split_into_sentences(input: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut chars = input.chars().peekable();

        while let Some(ch) = chars.next() {
            current.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                let at_boundary = chars.peek().is_none_or(|c| c.is_whitespace());
                if at_boundary {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current.clear();
                }
            }
        }

        let remainder = current.trim().to_string();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }

        sentences
    }

    /// Appends sub-segments of `s` to `out`, each at most `max_chars` characters
    /// long. Cuts at word boundaries — the last word of each sub-segment is always
    /// completed even if it slightly exceeds `max_chars`.
    fn split_at_word_boundaries(input: &str, max_chars: usize, out: &mut Vec<String>) {
        let mut start = 0;
        while start < input.len() {
            // Convert char count to byte offset within the remaining slice.
            let char_byte_offset = input[start..]
                .char_indices()
                .nth(max_chars)
                .map(|(i, _)| i)
                .unwrap_or(input[start..].len());
            let end = Self::word_boundary_end(input, start + char_byte_offset);
            let sub = input[start..end].trim().to_string();
            if !sub.is_empty() {
                out.push(sub);
            }
            start = end;
            let skipped: usize = input[start..]
                .chars()
                .take_while(|c| c.is_whitespace())
                .map(|c| c.len_utf8())
                .sum();
            start += skipped;
        }
    }

    fn word_boundary_end(input: &str, pos: usize) -> usize {
        let pos = ceil_char_boundary(input, pos.min(input.len()));
        if pos == input.len() || input[pos..].starts_with(char::is_whitespace) {
            return pos;
        }
        input[pos..]
            .find(char::is_whitespace)
            .map(|i| pos + i)
            .unwrap_or(input.len())
    }
}

impl Chunker for SentenceAwareChunker {
    fn chunk(&self, input: &str) -> Vec<String> {
        let chunk_size = self.chunk_size();

        let overlap_size = self.overlap_size.min(chunk_size.saturating_sub(1));
        let segments = Self::split_into_segments(input);

        // Break any segment that exceeds chunk_size characters at word boundaries.
        let mut fine_segments: Vec<String> = Vec::new();
        for seg in &segments {
            if seg.chars().count() <= chunk_size {
                fine_segments.push(seg.clone());
            } else {
                Self::split_at_word_boundaries(seg, chunk_size, &mut fine_segments);
            }
        }

        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for seg in &fine_segments {
            let would_len = if current.is_empty() {
                seg.chars().count()
            } else {
                current.chars().count() + 1 + seg.chars().count() // +1 for the joining space
            };

            if !current.is_empty() && would_len > chunk_size {
                let prev = std::mem::take(&mut current);
                current = overlap_tail(&prev, overlap_size);
                chunks.push(prev);
            }

            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(seg);
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        if chunks.is_empty() {
            vec![input.to_string()]
        } else {
            chunks
        }
    }
}

/// Returns the suffix of `s` that begins at the first word boundary at or
/// after `s.len() - overlap_size`, ensuring the overlap region never starts
/// mid-word.
///
/// If `s` is shorter than `overlap_size`, the entire string is returned.
fn overlap_tail(input: &str, overlap_size: usize) -> String {
    if overlap_size == 0 {
        return String::new();
    }
    let char_count = input.chars().count();
    if char_count <= overlap_size {
        return input.to_string();
    }
    // Find the byte offset of the character at position (char_count - overlap_size).
    let raw_start = input
        .char_indices()
        .nth(char_count - overlap_size)
        .map(|(i, _)| i)
        .unwrap_or(input.len());
    let start = if !input[raw_start..].starts_with(char::is_whitespace) {
        input[raw_start..]
            .find(char::is_whitespace)
            .map(|i| raw_start + i)
            .unwrap_or(input.len())
    } else {
        raw_start
    };
    input[start..].trim_start().to_string()
}

/// Advances `pos` to the nearest valid UTF-8 character boundary in `s`.
fn ceil_char_boundary(input: &str, pos: usize) -> usize {
    let mut p = pos;
    while p < input.len() && !input.is_char_boundary(p) {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use pretty_assertions::assert_eq;

    use super::{Chunker, SentenceAwareChunker};

    fn split_into_chunks(text: &str, chunk_size: usize, overlap_size: usize) -> Vec<String> {
        SentenceAwareChunker::new(NonZeroUsize::new(chunk_size).unwrap(), overlap_size).chunk(text)
    }

    #[test]
    fn test_short_text_produces_single_chunk() {
        // 34 chars — well within a 200-char limit.
        let text = "Hello world. This is a short text.";
        let chunks = split_into_chunks(text, 200, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_splits_at_paragraph_boundary() {
        // Two paragraphs, each ~250 chars; chunk_size 300 → 2 chunks.
        let para = "word ".repeat(50); // 250 chars
        let text = format!("{}\n\n{}", para.trim(), para.trim());
        let chunks = split_into_chunks(&text, 300, 0);
        assert!(chunks.len() >= 2, "expected split, got: {chunks:?}");
    }

    #[test]
    fn test_splits_at_sentence_boundary() {
        // "The quick brown fox jumps over the lazy dog in the meadow. " = 59 chars
        // 3 sentences = ~177 chars; chunk_size 100 → ≥ 2 chunks.
        let sentence = "The quick brown fox jumps over the lazy dog in the meadow. ";
        let text = sentence.repeat(3);
        let chunks = split_into_chunks(&text, 100, 0);
        assert!(
            chunks.len() >= 2,
            "expected multiple chunks, got: {chunks:?}"
        );
    }

    #[test]
    fn test_oversized_segment_falls_back_to_word_split() {
        // One long run-on with no punctuation — no sentence/paragraph boundaries.
        // 200 words of "wordN" (avg ~7 chars each) = ~1400 chars.
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = split_into_chunks(&text, 100, 0);
        assert!(
            chunks.len() >= 4,
            "expected at least 4 chunks, got: {chunks:?}"
        );
        // Each chunk may slightly exceed 100 chars by at most one word length,
        // but should not be grossly over (< 120 chars for these short words).
        for chunk in &chunks {
            assert!(
                chunk.len() < 120,
                "chunk is too long (expected ~100 chars): {chunk}"
            );
        }
    }

    #[test]
    fn test_overlap_bleeds_context_into_next_chunk() {
        // Build a predictable string: "aaaa bbbb cccc dddd ..." where each word is 4 chars.
        // 30 such words = 150 chars (with spaces). chunk_size=50, overlap_size=15.
        let words: Vec<String> = (b'a'..=b'z')
            .take(30)
            .map(|c| std::str::from_utf8(&[c; 4]).unwrap().to_string())
            .collect();
        let input = words.join(" ");
        let chunks = split_into_chunks(&input, 50, 15);
        assert!(chunks.len() >= 2, "expected multiple chunks");

        // The tail of chunk[0] should appear at the start of chunk[1].
        let tail: Vec<&str> = chunks[0]
            .split_whitespace()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let tail_str = tail.join(" ");
        assert!(
            chunks[1].starts_with(&tail_str),
            "expected chunk[1] to start with the tail of chunk[0] (\"{tail_str}\"), got: \"{}\"",
            chunks[1]
        );
    }

    #[test]
    fn test_zero_overlap_no_word_repeated_across_chunks() {
        // All words are unique; with zero overlap no word should appear in two chunks.
        let words: Vec<String> = (0..100).map(|i| format!("unique{i}")).collect();
        let input = words.join(" ");
        let chunks = split_into_chunks(&input, 100, 0);
        assert!(chunks.len() >= 2, "expected multiple chunks");

        let mut seen = std::collections::HashSet::new();
        for chunk in &chunks {
            for word in chunk.split_whitespace() {
                assert!(
                    seen.insert(word.to_string()),
                    "word '{word}' appeared in more than one chunk"
                );
            }
        }
    }

    #[test]
    fn test_empty_input_returns_single_chunk() {
        let chunks = split_into_chunks("", 100, 10);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_overlap_clamped_to_chunk_size_minus_one() {
        // overlap_size >= chunk_size should not panic or loop infinitely.
        let words: Vec<String> = (0..50).map(|i| format!("w{i}")).collect();
        let input = words.join(" ");
        let chunks = split_into_chunks(&input, 30, 50); // overlap > chunk_size
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_multiple_blank_lines_treated_as_paragraph_separator() {
        let para = "word ".repeat(40); // 200 chars
        let input = format!("{}\n\n\n\n{}", para.trim(), para.trim());
        let chunks = split_into_chunks(&input, 250, 0);
        assert!(
            chunks.len() >= 2,
            "expected split on multiple blank lines: {chunks:?}"
        );
    }

    #[test]
    fn test_sentence_aware_chunker_short_text() {
        let input = "Hello world. This is a short text.";
        let chunker = SentenceAwareChunker::new(NonZeroUsize::new(200).unwrap(), 20);
        let chunks = chunker.chunk(input);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], input);
    }

    #[test]
    fn test_sentence_aware_chunker_respects_chunk_size() {
        // 200 words of "wordN", chunk_size=100 chars → many chunks.
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let input = words.join(" ");
        let chunker = SentenceAwareChunker::new(NonZeroUsize::new(100).unwrap(), 0);
        let chunks = chunker.chunk(&input);
        assert!(
            chunks.len() >= 4,
            "expected at least 4 chunks, got: {chunks:?}"
        );
    }
}
