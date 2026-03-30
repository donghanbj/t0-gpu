//! DataLoader — Load token data for training with mini-batching and epoch shuffle.


/// DataLoader for language model training.
///
/// Supports:
/// - `.bin` files (raw u32 token IDs, little-endian)
/// - `.txt` files (char-level tokenization)
/// - Mini-batch iteration with configurable batch_size and seq_len
/// - Epoch shuffle
pub struct DataLoader {
    tokens: Vec<u32>,
    batch_size: usize,
    seq_len: usize,
    cursor: usize,
    epoch: usize,
}

impl DataLoader {
    /// Load from a binary token file (u32 little-endian).
    pub fn from_bin(path: &str, batch_size: usize, seq_len: usize) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        if data.len() % 4 != 0 {
            return Err(format!("bin file size {} not multiple of 4", data.len()));
        }
        let tokens: Vec<u32> = data.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("[DataLoader] Loaded {} tokens from {}", tokens.len(), path);
        Ok(Self { tokens, batch_size, seq_len, cursor: 0, epoch: 0 })
    }

    /// Load from a text file (char-level tokenization).
    pub fn from_text(path: &str, batch_size: usize, seq_len: usize) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        let tokens: Vec<u32> = text.chars().map(|c| c as u32).collect();
        eprintln!("[DataLoader] Loaded {} chars from {}", tokens.len(), path);
        Ok(Self { tokens, batch_size, seq_len, cursor: 0, epoch: 0 })
    }

    /// Create from a raw token vector.
    pub fn from_tokens(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> Self {
        Self { tokens, batch_size, seq_len, cursor: 0, epoch: 0 }
    }

    /// Get next batch: (inputs, targets) both of shape [batch_size, seq_len]
    ///
    /// Returns None when epoch is exhausted (call reset_epoch() to restart).
    pub fn next_batch(&mut self) -> Option<(Vec<u32>, Vec<u32>)> {
        let stride = self.seq_len + 1; // +1 for target offset
        let needed = self.batch_size * stride;

        if self.cursor + needed > self.tokens.len() {
            return None; // epoch done
        }

        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for b in 0..self.batch_size {
            let start = self.cursor + b * stride;
            for i in 0..self.seq_len {
                inputs.push(self.tokens[start + i]);
                targets.push(self.tokens[start + i + 1]);
            }
        }

        self.cursor += self.batch_size * stride;
        Some((inputs, targets))
    }

    /// Reset for new epoch with optional shuffle.
    pub fn reset_epoch(&mut self, shuffle: bool) {
        self.cursor = 0;
        self.epoch += 1;
        if shuffle {
            // Simple Fisher-Yates with LCG
            let n = self.tokens.len();
            let mut rng = (self.epoch as u64).wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            for i in (1..n).rev() {
                rng = rng.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                let j = (rng >> 17) as usize % (i + 1);
                self.tokens.swap(i, j);
            }
        }
    }

    /// Total number of tokens.
    pub fn len(&self) -> usize { self.tokens.len() }

    /// Number of batches per epoch.
    pub fn batches_per_epoch(&self) -> usize {
        let stride = self.seq_len + 1;
        self.tokens.len() / (self.batch_size * stride)
    }

    /// Current epoch number.
    pub fn epoch(&self) -> usize { self.epoch }
}
