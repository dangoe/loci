// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{collections::HashSet, future::Future, marker::PhantomData, sync::Arc};

use crate::{error::MemoryExtractionError, memory::MemoryInput};

use super::MemoryExtractionStrategy;

/// Parameters for [`ChunkingMemoryExtractionStrategy`].
pub struct ChunkingParams<P> {
    /// Maximum number of words per chunk.
    pub chunk_size: usize,
    /// Number of words from the end of each chunk to repeat at the start of the
    /// next, preserving context across chunk boundaries.
    /// Clamped to `chunk_size - 1` if larger.
    pub overlap: usize,
    /// Params forwarded verbatim to the inner extraction strategy for every chunk.
    pub inner: P,
}

/// A [`MemoryExtractionStrategy`] wrapper that splits large inputs into
/// overlapping chunks before delegating to an inner strategy.
///
/// Chunks are split at paragraph then sentence boundaries; segments that exceed
/// `chunk_size` words fall back to word-level splitting. After collecting
/// results from all chunks, exact-duplicate entries (same content string) are
/// removed before returning.
pub struct ChunkingMemoryExtractionStrategy<E, P> {
    inner: Arc<E>,
    phantom: PhantomData<P>,
}

impl<E, P> ChunkingMemoryExtractionStrategy<E, P> {
    /// Creates a new strategy wrapping `inner`.
    pub fn new(inner: E) -> Self {
        Self {
            inner: Arc::new(inner),
            phantom: PhantomData,
        }
    }

    /// Creates a new strategy from a pre-existing `Arc` handle — useful when
    /// the inner strategy is already shared with another component.
    pub fn from_arc(inner: Arc<E>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<E, P> MemoryExtractionStrategy<ChunkingParams<P>> for ChunkingMemoryExtractionStrategy<E, P>
where
    E: MemoryExtractionStrategy<P> + Send + Sync,
    P: Clone + Send + Sync + 'static,
{
    fn extract(
        &self,
        input: &str,
        params: ChunkingParams<P>,
    ) -> impl Future<Output = Result<Vec<MemoryInput>, MemoryExtractionError>> + Send {
        let chunks = split_into_chunks(input, params.chunk_size, params.overlap);
        let inner = Arc::clone(&self.inner);

        async move {
            let mut all_entries: Vec<MemoryInput> = Vec::new();

            for chunk in chunks {
                let mut entries = inner.extract(&chunk, params.inner.clone()).await?;
                all_entries.append(&mut entries);
            }

            // Exact-match dedup: retain the first occurrence of each content string.
            let mut seen: HashSet<String> = HashSet::new();
            all_entries.retain(|e| seen.insert(e.content.clone()));

            Ok(all_entries)
        }
    }
}

/// Splits `text` into a flat list of segments at paragraph then sentence
/// boundaries. Blank lines (one or more) are treated as paragraph separators.
fn split_into_segments(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut paragraph_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            if !paragraph_lines.is_empty() {
                segments.extend(split_into_sentences(&paragraph_lines.join(" ")));
                paragraph_lines.clear();
            }
        } else {
            paragraph_lines.push(line);
        }
    }

    if !paragraph_lines.is_empty() {
        segments.extend(split_into_sentences(&paragraph_lines.join(" ")));
    }

    segments
}

/// Splits `text` into sentences, breaking after `.`, `!`, or `?` when followed
/// by whitespace or end-of-string.
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let at_boundary = chars.get(i + 1).is_none_or(|c| c.is_whitespace());
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

/// Splits `text` into overlapping word-counted chunks.
///
/// Boundaries are respected in priority order: paragraphs → sentences →
/// individual words (fallback for segments that exceed `chunk_size`). Each
/// chunk except the first begins with the last `overlap` words of the preceding
/// chunk. `overlap` is silently clamped to `chunk_size - 1`.
///
/// Returns a single-element `Vec` containing the entire input when it fits
/// within one chunk or when `chunk_size` is zero.
pub(super) fn split_into_chunks(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if chunk_size == 0 {
        return vec![text.to_string()];
    }

    let overlap = overlap.min(chunk_size.saturating_sub(1));
    let segments = split_into_segments(text);

    // Break any segment that exceeds chunk_size into word-level sub-segments.
    let mut fine_segments: Vec<Vec<String>> = Vec::new();
    for seg in &segments {
        let words: Vec<String> = seg.split_whitespace().map(str::to_owned).collect();
        if words.len() <= chunk_size {
            fine_segments.push(words);
        } else {
            let mut start = 0;
            while start < words.len() {
                let end = (start + chunk_size).min(words.len());
                fine_segments.push(words[start..end].to_vec());
                start += chunk_size;
            }
        }
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for seg_words in fine_segments {
        if !current.is_empty() && current.len() + seg_words.len() > chunk_size {
            chunks.push(current.join(" "));
            let overlap_start = current.len().saturating_sub(overlap);
            current = current[overlap_start..].to_vec();
        }
        current.extend(seg_words);
    }

    if !current.is_empty() {
        chunks.push(current.join(" "));
    }

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::split_into_chunks;

    fn word_count(s: &str) -> usize {
        s.split_whitespace().count()
    }

    #[test]
    fn test_short_text_produces_single_chunk() {
        let text = "Hello world. This is a short text.";
        let chunks = split_into_chunks(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_splits_at_paragraph_boundary() {
        let para = "word ".repeat(50);
        let text = format!("{}\n\n{}", para.trim(), para.trim());
        // 100 words total, chunk_size 60 → should produce multiple chunks
        let chunks = split_into_chunks(&text, 60, 0);
        assert!(chunks.len() >= 2, "expected split, got: {chunks:?}");
    }

    #[test]
    fn test_splits_at_sentence_boundary() {
        // Each sentence is 13 words; 3 sentences = 39 words; chunk_size 25 → 3 chunks.
        let sentence = "The quick brown fox jumps over the lazy dog in the meadow. ";
        let text = sentence.repeat(3);
        let chunks = split_into_chunks(&text, 25, 0);
        assert!(chunks.len() >= 2, "expected multiple chunks, got: {chunks:?}");
    }

    #[test]
    fn test_oversized_segment_falls_back_to_word_split() {
        // One long run-on with no punctuation — no sentence/paragraph boundaries.
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = split_into_chunks(&text, 50, 0);
        assert!(chunks.len() >= 4, "expected at least 4 chunks, got: {chunks:?}");
        for chunk in &chunks {
            assert!(
                word_count(chunk) <= 50,
                "chunk exceeds chunk_size: {chunk}"
            );
        }
    }

    #[test]
    fn test_overlap_bleeds_context_into_next_chunk() {
        let words: Vec<String> = (0..120).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let overlap = 10;
        let chunks = split_into_chunks(&text, 50, overlap);
        assert!(chunks.len() >= 2, "expected multiple chunks");

        let end_of_first: Vec<&str> = chunks[0]
            .split_whitespace()
            .rev()
            .take(overlap)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let start_of_second: Vec<&str> = chunks[1].split_whitespace().take(overlap).collect();
        assert_eq!(
            end_of_first, start_of_second,
            "last {overlap} words of chunk[0] should be the first {overlap} words of chunk[1]"
        );
    }

    #[test]
    fn test_zero_overlap_no_repeated_words() {
        // Without overlap, each word should appear in exactly one chunk.
        let words: Vec<String> = (0..150).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let chunks = split_into_chunks(&text, 50, 0);
        let total: usize = chunks.iter().map(|c| word_count(c)).sum();
        assert_eq!(total, 150);
    }

    #[test]
    fn test_empty_input_returns_single_chunk() {
        let chunks = split_into_chunks("", 100, 10);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_overlap_clamped_to_chunk_size_minus_one() {
        // overlap >= chunk_size should not panic or loop infinitely.
        let words: Vec<String> = (0..100).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let chunks = split_into_chunks(&text, 20, 30); // overlap > chunk_size
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_multiple_blank_lines_treated_as_paragraph_separator() {
        let para = "word ".repeat(40);
        let text = format!("{}\n\n\n\n{}", para.trim(), para.trim());
        let chunks = split_into_chunks(&text, 50, 0);
        assert!(chunks.len() >= 2, "expected split on multiple blank lines: {chunks:?}");
    }
}
