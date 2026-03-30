//! Tokenizer — BPE tokenizer with train/encode/decode/save.
//!
//! Also includes VocabTokenizer for loading pre-built vocabularies.

use std::collections::HashMap;

/// BPE (Byte-Pair Encoding) tokenizer.
pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    token_to_id: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Train BPE on text data with given vocab size.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        // Start with byte-level vocab
        let mut vocab: Vec<String> = (0..256u32)
            .map(|b| format!("{}", b as u8 as char))
            .collect();
        let mut token_to_id: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        // Tokenize as chars
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let mut tokens = chars;
        let mut merges = Vec::new();

        while vocab.len() < vocab_size {
            // Count adjacent pairs
            let mut pairs: HashMap<(String, String), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *pairs.entry((w[0].clone(), w[1].clone())).or_default() += 1;
            }
            if pairs.is_empty() { break; }

            // Find most frequent pair
            let best = pairs.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
            let merged = format!("{}{}", best.0, best.1);

            // Add to vocab
            let new_id = vocab.len() as u32;
            vocab.push(merged.clone());
            token_to_id.insert(merged.clone(), new_id);
            merges.push(best.clone());

            // Apply merge
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == best.0 && tokens[i + 1] == best.1 {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        Self { vocab, merges, token_to_id }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merges in order
        for (a, b) in &self.merges {
            let merged = format!("{}{}", a, b);
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && &tokens[i] == a && &tokens[i + 1] == b {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens.iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                if (id as usize) < self.vocab.len() {
                    self.vocab[id as usize].clone()
                } else {
                    "?".to_string()
                }
            })
            .collect()
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// Save tokenizer to file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut output = String::new();
        output.push_str(&format!("vocab_size={}\n", self.vocab.len()));
        for (i, token) in self.vocab.iter().enumerate() {
            output.push_str(&format!("{}={}\n", i, token));
        }
        output.push_str("---merges---\n");
        for (a, b) in &self.merges {
            output.push_str(&format!("{} {}\n", a, b));
        }
        std::fs::write(path, output).map_err(|e| format!("save tokenizer: {}", e))
    }
}

/// Simple vocabulary tokenizer (char-level or word-level from a file).
pub struct VocabTokenizer {
    pub vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
}

impl VocabTokenizer {
    /// Load from a text file (one token per line).
    pub fn from_file(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}", e))?;
        let vocab: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        let token_to_id: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Ok(Self { vocab, token_to_id })
    }

    /// Create char-level tokenizer from a text corpus.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let vocab: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let token_to_id: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self { vocab, token_to_id }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| *self.token_to_id.get(&c.to_string()).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                if (id as usize) < self.vocab.len() { self.vocab[id as usize].clone() }
                else { "?".to_string() }
            })
            .collect()
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }
}
