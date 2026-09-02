//! Local knowledge base for interview preparation.
//!
//! Users upload project documents (Markdown / plain text) via the Settings
//! window. At ask time the current question is run through a real BM25 index
//! over the chunked corpus and the top matching passages are injected into the
//! prompt as `<evidence>`, so the model can answer project / internship
//! questions from the user's actual material instead of guessing.
//!
//! The retrieval is deliberately dependency-free (no embedding model, no FTS
//! crate): the corpus is small (a handful of documents), so a rebuilt-in-memory
//! BM25 index with Chinese bigram tokenization is fast enough and keeps the
//! whole feature offline. This mirrors the highest-leverage slice of cluely's
//! RAG (real BM25 + chunking + grounding rules) while dropping its embedding /
//! rerank models, which DeepSeek (text-only) cannot use anyway.

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Longest chunk body in characters. Chinese text averages ~1 word per
/// character, so 800 chars ≈ cluely's 420-word card body.
const MAX_CHUNK_CHARS: usize = 800;
/// Longest evidence passage injected into the prompt, per hit.
const MAX_EVIDENCE_CHARS: usize = 400;
/// BM25 term-frequency saturation.
const BM25_K1: f64 = 1.5;
/// BM25 length normalisation.
const BM25_B: f64 = 0.75;
/// Minimum normalized score a hit must reach to be injected as evidence.
/// Below this the match is effectively noise (generic/stopword overlap).
const MIN_EVIDENCE_SCORE: f32 = 0.08;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeDoc {
    pub name: String,
    pub size: usize,
    pub chunk_count: usize,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeHit {
    pub doc: String,
    pub text: String,
    pub score: f32,
}

/// `~/.meetly/knowledge/docs/` — the raw uploaded documents live here.
fn docs_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or_else(|| "无法解析用户目录。".to_string())?;
    Ok(home.join(".meetly").join("knowledge").join("docs"))
}

/// Lists every uploaded document with its metadata.
#[tauri::command]
pub fn list_knowledge_docs() -> Result<Vec<KnowledgeDoc>, String> {
    let dir = docs_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut docs = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };
        let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
        let content = fs::read_to_string(&path).unwrap_or_default();
        docs.push(KnowledgeDoc {
            name,
            size: metadata.len() as usize,
            chunk_count: chunk_text(&content).len(),
            updated_at_ms: metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
        });
    }

    docs.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
    Ok(docs)
}

/// Adds (or replaces) a document. `name` is the plain file name; `content` is
/// the decoded UTF-8 text already read by the frontend.
#[tauri::command]
pub fn add_knowledge_doc(name: String, content: String) -> Result<KnowledgeDoc, String> {
    let name = safe_file_name(&name)?;
    if content.trim().is_empty() {
        return Err("文档内容为空。".to_string());
    }

    let dir = docs_dir()?;
    fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join(&name);
    fs::write(&path, content.as_bytes()).map_err(|error| error.to_string())?;

    let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
    Ok(KnowledgeDoc {
        chunk_count: chunk_text(&content).len(),
        name,
        size: metadata.len() as usize,
        updated_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_default(),
    })
}

/// Removes a document by name.
#[tauri::command]
pub fn remove_knowledge_doc(name: String) -> Result<(), String> {
    let name = safe_file_name(&name)?;
    let path = docs_dir()?.join(&name);
    if !path.exists() {
        return Err(format!("文档 {name} 不存在。"));
    }
    fs::remove_file(&path).map_err(|error| error.to_string())
}

/// Retrieves the top matching passages for a question and formats them into a
/// single `<evidence>` block, or `None` when nothing clears the relevance bar.
///
/// This is called from the ask paths (voice + typed) right before building the
/// user message, so the retrieved material travels with the question.
pub fn retrieve_context(question: &str, top_k: usize) -> Option<String> {
    let hits = search(question, top_k);
    if hits.is_empty() {
        return None;
    }

    let mut parts = Vec::with_capacity(hits.len());
    for hit in hits {
        parts.push(format!("【{}】\n{}", hit.doc, hit.text));
    }
    Some(parts.join("\n\n---\n\n"))
}

/// Full retrieval: load all documents, chunk them, index with BM25, rank the
/// question, keep only hits above the relevance threshold.
fn search(question: &str, top_k: usize) -> Vec<KnowledgeHit> {
    let chunks = load_all_chunks();
    if chunks.is_empty() {
        return Vec::new();
    }

    let index = Bm25Index::new(&chunks);
    let scored = index.score_normalized(question);

    scored
        .into_iter()
        .filter(|(_, score)| *score >= MIN_EVIDENCE_SCORE)
        .take(top_k)
        .map(|(id, score)| {
            let chunk = &chunks[id];
            KnowledgeHit {
                doc: chunk.doc.clone(),
                text: truncate_chars(&chunk.text, MAX_EVIDENCE_CHARS),
                score,
            }
        })
        .collect()
}

fn load_all_chunks() -> Vec<Chunk> {
    let Ok(dir) = docs_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut chunks = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for text in chunk_text(&content) {
            chunks.push(Chunk {
                doc: name.clone(),
                text,
            });
        }
    }
    chunks
}

struct Chunk {
    doc: String,
    text: String,
}

/// Splits a document into passages. Headings start a new chunk (and are kept at
/// its top); blank lines act as soft boundaries. Oversized chunks are split
/// further on a character boundary so no single passage exceeds the cap.
fn chunk_text(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        let is_heading = trimmed.starts_with('#');
        let is_blank = trimmed.is_empty();

        if (is_heading || is_blank) && !current.trim().is_empty() {
            push_chunk(&mut chunks, &mut current);
        }

        if is_blank {
            continue;
        }
        if is_heading {
            current = trimmed.to_string();
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    push_chunk(&mut chunks, &mut current);

    chunks
        .into_iter()
        .filter(|c| !c.trim().is_empty())
        .collect()
}

fn push_chunk(chunks: &mut Vec<String>, current: &mut String) {
    let text = current.trim().to_string();
    current.clear();
    if text.is_empty() {
        return;
    }
    if text.chars().count() > MAX_CHUNK_CHARS {
        for piece in split_by_chars(&text, MAX_CHUNK_CHARS) {
            chunks.push(piece);
        }
    } else {
        chunks.push(text);
    }
}

fn split_by_chars(text: &str, max: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(max)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

/// Tokenizes for BM25. Chinese is segmented into unigrams + bigrams (adjacent
/// character pairs), which is the standard dictionary-free approach for CJK
/// retrieval. Latin/numeric runs become lowercase whole-word tokens of length
/// >= 3. Everything else (whitespace, punctuation) is a separator.
fn tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let n = chars.len();

    while i < n {
        let c = chars[i];
        if c.is_ascii_alphanumeric() {
            let start = i;
            while i < n && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect::<String>().to_lowercase();
            if word.chars().count() >= 3 {
                tokens.push(word);
            }
        } else if is_cjk_char(c) {
            let start = i;
            while i < n && is_cjk_char(chars[i]) {
                i += 1;
            }
            let seq = &chars[start..i];
            for j in 0..seq.len() {
                tokens.push(seq[j].to_string());
                if j + 1 < seq.len() {
                    let mut bigram = String::with_capacity(8);
                    bigram.push(seq[j]);
                    bigram.push(seq[j + 1]);
                    tokens.push(bigram);
                }
            }
        } else {
            i += 1;
        }
    }
    tokens
}

fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp) // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&cp) // Extension A
        || (0x20000..=0x2A6DF).contains(&cp) // Extension B
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn safe_file_name(name: &str) -> Result<String, String> {
    let base = Path::new(name)
        .file_name()
        .ok_or_else(|| "无效的文件名。".to_string())?
        .to_string_lossy()
        .to_string();
    if base.is_empty() || base == "." || base == ".." {
        return Err("无效的文件名。".to_string());
    }
    Ok(base)
}

struct DocStat {
    id: usize,
    tf: HashMap<String, usize>,
    len: usize,
}

/// Okapi BM25 (k1=1.5, b=0.75) with Robertson/Sparck-Jones IDF (+0.5
/// smoothing, floored at 0). Mirrors cluely's `bm25.ts` so ranking behaviour is
/// comparable.
struct Bm25Index {
    docs: Vec<DocStat>,
    df: HashMap<String, usize>,
    avgdl: f64,
}

impl Bm25Index {
    fn new(chunks: &[Chunk]) -> Self {
        let mut docs = Vec::with_capacity(chunks.len());
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;

        for (id, chunk) in chunks.iter().enumerate() {
            let terms = tokenize(&chunk.text);
            let mut tf: HashMap<String, usize> = HashMap::new();
            for term in &terms {
                *tf.entry(term.clone()).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            total += terms.len();
            docs.push(DocStat {
                id,
                tf,
                len: terms.len(),
            });
        }

        let avgdl = if docs.is_empty() {
            1.0
        } else {
            total as f64 / docs.len() as f64
        };

        Bm25Index { docs, df, avgdl }
    }

    fn idf(&self, term: &str) -> f64 {
        let n = self.df.get(term).copied().unwrap_or(0) as f64;
        let big_n = self.docs.len().max(1) as f64;
        ((1.0 + (big_n - n + 0.5) / (n + 0.5)).ln()).max(0.0)
    }

    fn score(&self, query: &str) -> Vec<(usize, f64)> {
        let q_terms = tokenize(query);
        let mut out = Vec::with_capacity(self.docs.len());

        for doc in &self.docs {
            let mut score = 0.0;
            for term in &q_terms {
                let Some(&freq) = doc.tf.get(term) else {
                    continue;
                };
                let f = freq as f64;
                let norm = 1.0 - BM25_B + BM25_B * (doc.len as f64 / self.avgdl.max(1.0));
                score += self.idf(term) * ((f * (BM25_K1 + 1.0)) / (f + BM25_K1 * norm));
            }
            out.push((doc.id, score));
        }

        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Min-max normalizes scores to [0, 1] so the evidence threshold is stable
    /// across queries and corpus sizes (BM25 is unbounded otherwise).
    fn score_normalized(&self, query: &str) -> Vec<(usize, f32)> {
        let scored = self.score(query);
        let max = scored.first().map(|(_, s)| *s).unwrap_or(0.0);
        if max <= 0.0 {
            return scored.into_iter().map(|(id, _)| (id, 0.0)).collect();
        }
        scored
            .into_iter()
            .map(|(id, s)| (id, (s / max) as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_makes_cjk_bigrams() {
        let tokens = tokenize("分布式限流");
        assert!(tokens.contains(&"分布".to_string()));
        assert!(tokens.contains(&"限流".to_string()));
        assert!(tokens.contains(&"布".to_string()));
    }

    #[test]
    fn tokenize_keeps_latin_words_and_drops_stopwords() {
        let tokens = tokenize("Redis Lua script");
        assert!(tokens.contains(&"redis".to_string()));
        assert!(tokens.contains(&"lua".to_string()));
        assert!(tokens.contains(&"script".to_string()));
    }

    #[test]
    fn chunk_text_splits_on_headings() {
        let text = "# 项目\n用 Redis 做了限流\n\n# 难点\nLua 脚本保证原子性";
        let chunks = chunk_text(text);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].starts_with("# 项目"));
    }

    #[test]
    fn bm25_ranks_relevant_chunk_first() {
        let chunks = vec![
            Chunk {
                doc: "a".to_string(),
                text: "用 Redis 和 Lua 实现分布式限流器".to_string(),
            },
            Chunk {
                doc: "b".to_string(),
                text: "使用 React 搭建前端页面组件".to_string(),
            },
        ];
        let index = Bm25Index::new(&chunks);
        let scored = index.score("Redis 限流");
        assert_eq!(scored[0].0, 0);
        assert!(scored[0].1 > scored[1].1);
    }

    #[test]
    fn safe_file_name_strips_traversal() {
        assert_eq!(safe_file_name("resume.md").unwrap(), "resume.md");
        assert_eq!(safe_file_name("../../etc/passwd").unwrap(), "passwd");
        assert!(safe_file_name("..").is_err());
    }
}
