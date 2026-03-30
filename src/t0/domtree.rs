//! Dominator Tree — 通用支配树基础设施
//!
//! 提供 CFG 抽象 trait + Cooper-Harvey-Kennedy 迭代算法 + Dominance Frontier 计算。
//! 可被 `TileFunc` 和 `MachFunc` 两层 SSA IR 共用。
//!
//! # 算法
//!
//! 使用 Cooper, Harvey, Kennedy (2001) "A Simple, Fast Dominance Algorithm"。
//! 迭代 fixpoint 求解 `idom[b]`，复杂度 O(n²) 最坏，结构化 CFG ~O(n)。
//!
//! # 使用
//!
//! ```ignore
//! use t0_gpu::t0::domtree::{DomTree, CfgProvider};
//!
//! let dt = DomTree::build(&my_cfg);
//! assert!(dt.dominates(0, 3));           // block 0 支配 block 3?
//! let df = dt.dominance_frontier(&my_cfg);  // Dominance Frontier
//! ```

use std::collections::HashSet;

// ============================================================================
// CFG Provider Trait
// ============================================================================

/// 通用 CFG 接口 — MachFunc 和 TileFunc 都可以实现。
///
/// Block 用 u32 编号，0 = entry。
pub trait CfgProvider {
    /// 基本块总数
    fn num_blocks(&self) -> usize;
    /// 入口块编号（通常为 0）
    fn entry(&self) -> u32;
    /// 块 `block` 的前驱列表
    fn preds(&self, block: u32) -> &[u32];
    /// 块 `block` 的后继列表
    fn succs(&self, block: u32) -> &[u32];
}

// ============================================================================
// Dominator Tree
// ============================================================================

/// Dominator Tree — 存储 idom 映射和查询缓存。
#[derive(Clone, Debug)]
pub struct DomTree {
    /// idom[b] = b 的直接支配者。idom[entry] = entry (自环)。
    idom: Vec<u32>,
    /// 反向后序编号（用于加速迭代收敛）
    rpo_order: Vec<u32>,
    /// rpo_number[block] = block 在反向后序中的序号
    rpo_number: Vec<u32>,
    /// 块总数
    num_blocks: usize,
}

/// 未定义 idom 的哨兵值
const UNDEFINED: u32 = u32::MAX;

impl DomTree {
    /// 从 CFG 构建 Dominator Tree（Cooper-Harvey-Kennedy 算法）。
    ///
    /// # 算法步骤
    ///
    /// 1. 计算反向后序遍历（RPO）
    /// 2. 初始化 idom: entry → entry, 其余 → UNDEFINED
    /// 3. 迭代至不动点：对每个 block（按 RPO），idom[b] = intersect(preds_of_b...)
    pub fn build<C: CfgProvider>(cfg: &C) -> Self {
        let n = cfg.num_blocks();
        let entry = cfg.entry() as usize;

        // 空 CFG 或单块
        if n == 0 {
            return DomTree {
                idom: vec![],
                rpo_order: vec![],
                rpo_number: vec![],
                num_blocks: 0,
            };
        }
        if n == 1 {
            return DomTree {
                idom: vec![0],
                rpo_order: vec![0],
                rpo_number: vec![0],
                num_blocks: 1,
            };
        }

        // ── Step 1: 反向后序遍历 ──
        let rpo_order = compute_rpo(cfg);
        let mut rpo_number = vec![UNDEFINED; n];
        for (rpo_idx, &block) in rpo_order.iter().enumerate() {
            rpo_number[block as usize] = rpo_idx as u32;
        }

        // ── Step 2: 初始化 idom ──
        let mut idom = vec![UNDEFINED; n];
        idom[entry] = entry as u32;

        // ── Step 3: 迭代至不动点 ──
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo_order {
                if b as usize == entry {
                    continue;
                }
                let preds = cfg.preds(b);
                // 找到第一个已处理的前驱
                let mut new_idom = UNDEFINED;
                for &p in preds {
                    if idom[p as usize] != UNDEFINED {
                        new_idom = p;
                        break;
                    }
                }
                if new_idom == UNDEFINED {
                    continue; // 不可达块
                }
                // 与其余已处理前驱取交集
                for &p in preds {
                    if p == new_idom {
                        continue;
                    }
                    if idom[p as usize] != UNDEFINED {
                        new_idom = intersect(&idom, &rpo_number, p, new_idom);
                    }
                }
                if idom[b as usize] != new_idom {
                    idom[b as usize] = new_idom;
                    changed = true;
                }
            }
        }

        DomTree {
            idom,
            rpo_order,
            rpo_number,
            num_blocks: n,
        }
    }

    // ════════════════════════════════════════════════════════
    // 查询 API
    // ════════════════════════════════════════════════════════

    /// 获取 block 的直接支配者（idom），entry 的 idom 是自身。
    pub fn idom(&self, block: u32) -> u32 {
        self.idom[block as usize]
    }

    /// block `a` 是否支配 block `b`？
    ///
    /// 定义：a 支配 b 当且仅当从 entry 到 b 的所有路径都经过 a。
    /// 等价于：a 是 b 到 entry 路径上的祖先（或 a == b）。
    pub fn dominates(&self, a: u32, b: u32) -> bool {
        if a as usize >= self.num_blocks || b as usize >= self.num_blocks {
            return false;
        }
        let mut current = b;
        loop {
            if current == a {
                return true;
            }
            let parent = self.idom[current as usize];
            if parent == current {
                // 到达 entry，只有 a == entry 才可能
                return current == a;
            }
            current = parent;
        }
    }

    /// block `a` 是否严格支配 block `b`？（a 支配 b 且 a ≠ b）
    pub fn strictly_dominates(&self, a: u32, b: u32) -> bool {
        a != b && self.dominates(a, b)
    }

    /// 求 a 和 b 的最近公共支配者（Nearest Common Dominator）。
    pub fn common_dominator(&self, a: u32, b: u32) -> u32 {
        intersect(&self.idom, &self.rpo_number, a, b)
    }

    /// 获取 block 的直接被支配子节点列表。
    pub fn dom_children(&self, block: u32) -> Vec<u32> {
        let mut children = Vec::new();
        for i in 0..self.num_blocks {
            if i as u32 != block && self.idom[i] == block {
                children.push(i as u32);
            }
        }
        children
    }

    /// 按支配树前序遍历，返回块编号序列。
    ///
    /// 性质：如果 a 在序列中排在 b 前面，且 a 支配 b，
    /// 那么 a 一定出现在 b 之前。适用于跨块 CSE/LICM。
    pub fn preorder(&self) -> Vec<u32> {
        let entry = self.rpo_order.first().copied().unwrap_or(0);
        let mut result = Vec::with_capacity(self.num_blocks);
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            result.push(b);
            let mut children = self.dom_children(b);
            // 按 RPO 逆序入栈，使 RPO 序更小的先出栈
            children.sort_by(|a, b| self.rpo_number[*b as usize].cmp(&self.rpo_number[*a as usize]));
            stack.extend(children);
        }
        result
    }

    /// 获取 block 在支配树中的深度（entry = 0）。
    pub fn depth(&self, block: u32) -> u32 {
        let mut d = 0;
        let mut current = block;
        while self.idom[current as usize] != current {
            d += 1;
            current = self.idom[current as usize];
        }
        d
    }

    // ════════════════════════════════════════════════════════
    // Dominance Frontier
    // ════════════════════════════════════════════════════════

    /// 计算 Dominance Frontier。
    ///
    /// DF(b) = { y | ∃ pred p of y where b dominates p but b does NOT strictly dominate y }
    ///
    /// 这是 Phi 节点插入位置的核心依据。
    pub fn dominance_frontier<C: CfgProvider>(&self, cfg: &C) -> Vec<HashSet<u32>> {
        let n = self.num_blocks;
        let mut df: Vec<HashSet<u32>> = vec![HashSet::new(); n];

        for b in 0..n as u32 {
            let preds = cfg.preds(b);
            if preds.len() < 2 {
                continue; // 只有 join node 才有 DF entry
            }
            for &p in preds {
                let mut runner = p;
                while runner != self.idom[b as usize] {
                    df[runner as usize].insert(b);
                    if runner == self.idom[runner as usize] {
                        break; // 到达 entry
                    }
                    runner = self.idom[runner as usize];
                }
            }
        }

        df
    }

    /// 获取反向后序遍历序列（调试用）
    pub fn rpo(&self) -> &[u32] {
        &self.rpo_order
    }

    /// 获取块数
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/// 计算反向后序遍历（Reverse Post-Order）
fn compute_rpo<C: CfgProvider>(cfg: &C) -> Vec<u32> {
    let n = cfg.num_blocks();
    let mut visited = vec![false; n];
    let mut post_order: Vec<u32> = Vec::with_capacity(n);

    fn dfs<C: CfgProvider>(
        cfg: &C, block: u32, visited: &mut [bool], post_order: &mut Vec<u32>,
    ) {
        visited[block as usize] = true;
        for &succ in cfg.succs(block) {
            if !visited[succ as usize] {
                dfs(cfg, succ, visited, post_order);
            }
        }
        post_order.push(block);
    }

    dfs(cfg, cfg.entry(), &mut visited, &mut post_order);

    // Reverse to get RPO
    post_order.reverse();
    post_order
}

/// Cooper-Harvey-Kennedy 交集操作。
///
/// 给定两个已处理的 block，找到它们在 idom 树上的最近公共祖先。
fn intersect(idom: &[u32], rpo_number: &[u32], mut a: u32, mut b: u32) -> u32 {
    while a != b {
        while rpo_number[a as usize] > rpo_number[b as usize] {
            a = idom[a as usize];
        }
        while rpo_number[b as usize] > rpo_number[a as usize] {
            b = idom[b as usize];
        }
    }
    a
}

// ============================================================================
// 简单 CFG 实现（用于测试和独立使用）
// ============================================================================

/// 简单 CFG：手动构建的测试用 CFG
#[derive(Clone, Debug)]
pub struct SimpleCfg {
    preds: Vec<Vec<u32>>,
    succs: Vec<Vec<u32>>,
}

impl SimpleCfg {
    /// 创建 N 块的空 CFG
    pub fn new(n: usize) -> Self {
        SimpleCfg {
            preds: vec![vec![]; n],
            succs: vec![vec![]; n],
        }
    }

    /// 添加有向边 from → to
    pub fn add_edge(&mut self, from: u32, to: u32) {
        self.succs[from as usize].push(to);
        self.preds[to as usize].push(from);
    }
}

impl CfgProvider for SimpleCfg {
    fn num_blocks(&self) -> usize { self.succs.len() }
    fn entry(&self) -> u32 { 0 }
    fn preds(&self, block: u32) -> &[u32] { &self.preds[block as usize] }
    fn succs(&self, block: u32) -> &[u32] { &self.succs[block as usize] }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助：创建线性 CFG: BB0 → BB1 → BB2
    fn linear_cfg() -> SimpleCfg {
        let mut cfg = SimpleCfg::new(3);
        cfg.add_edge(0, 1);
        cfg.add_edge(1, 2);
        cfg
    }

    /// 辅助：创建菱形 CFG:
    /// ```text
    /// BB0 → BB1 → BB3
    ///  └──→ BB2 → BB3
    /// ```
    fn diamond_cfg() -> SimpleCfg {
        let mut cfg = SimpleCfg::new(4);
        cfg.add_edge(0, 1);
        cfg.add_edge(0, 2);
        cfg.add_edge(1, 3);
        cfg.add_edge(2, 3);
        cfg
    }

    /// 辅助：创建单层循环 CFG:
    /// ```text
    /// BB0 → BB1 (header) → BB2 (body) → BB1 (back edge)
    ///                   └→ BB3 (exit)
    /// ```
    fn loop_cfg() -> SimpleCfg {
        let mut cfg = SimpleCfg::new(4);
        cfg.add_edge(0, 1);  // entry → header
        cfg.add_edge(1, 2);  // header → body
        cfg.add_edge(1, 3);  // header → exit
        cfg.add_edge(2, 1);  // body → header (back edge)
        cfg
    }

    /// 辅助：嵌套循环 CFG:
    /// ```text
    /// BB0 → BB1 (outer header) → BB2 (inner header) → BB3 (inner body) → BB2
    ///                         │                     └→ BB4 (inner exit)
    ///                         └→ BB5 (outer exit)
    ///       BB4 → BB1 (outer back edge)
    /// ```
    fn nested_loop_cfg() -> SimpleCfg {
        let mut cfg = SimpleCfg::new(6);
        cfg.add_edge(0, 1);  // entry → outer header
        cfg.add_edge(1, 2);  // outer header → inner header
        cfg.add_edge(1, 5);  // outer header → outer exit
        cfg.add_edge(2, 3);  // inner header → inner body
        cfg.add_edge(2, 4);  // inner header → inner exit
        cfg.add_edge(3, 2);  // inner body → inner header (back)
        cfg.add_edge(4, 1);  // inner exit → outer header (back)
        cfg
    }

    // ═══════════════════════════════════════════════════
    //  idom 测试
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_domtree_linear() {
        let cfg = linear_cfg();
        let dt = DomTree::build(&cfg);

        assert_eq!(dt.idom(0), 0, "entry idom = self");
        assert_eq!(dt.idom(1), 0, "BB1 idom = BB0");
        assert_eq!(dt.idom(2), 1, "BB2 idom = BB1");
    }

    #[test]
    fn test_domtree_diamond() {
        let cfg = diamond_cfg();
        let dt = DomTree::build(&cfg);

        assert_eq!(dt.idom(0), 0, "entry idom = self");
        assert_eq!(dt.idom(1), 0, "BB1 idom = BB0");
        assert_eq!(dt.idom(2), 0, "BB2 idom = BB0");
        assert_eq!(dt.idom(3), 0, "BB3 idom = BB0 (join point)");
    }

    #[test]
    fn test_domtree_loop() {
        let cfg = loop_cfg();
        let dt = DomTree::build(&cfg);

        assert_eq!(dt.idom(0), 0, "entry idom = self");
        assert_eq!(dt.idom(1), 0, "header idom = entry");
        assert_eq!(dt.idom(2), 1, "body idom = header");
        assert_eq!(dt.idom(3), 1, "exit idom = header");
    }

    #[test]
    fn test_domtree_nested_loop() {
        let cfg = nested_loop_cfg();
        let dt = DomTree::build(&cfg);

        assert_eq!(dt.idom(0), 0, "entry idom = self");
        assert_eq!(dt.idom(1), 0, "outer header idom = entry");
        assert_eq!(dt.idom(2), 1, "inner header idom = outer header");
        assert_eq!(dt.idom(3), 2, "inner body idom = inner header");
        assert_eq!(dt.idom(4), 2, "inner exit idom = inner header");
        assert_eq!(dt.idom(5), 1, "outer exit idom = outer header");
    }

    // ═══════════════════════════════════════════════════
    //  Dominance Frontier 测试
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_dominance_frontier_diamond() {
        let cfg = diamond_cfg();
        let dt = DomTree::build(&cfg);
        let df = dt.dominance_frontier(&cfg);

        // BB0 支配所有块 → DF(BB0) = {}
        assert!(df[0].is_empty(), "DF(BB0) should be empty");
        // BB1 支配 BB1 但不严格支配 BB3 (BB3 还有 BB2 作为前驱)
        assert!(df[1].contains(&3), "DF(BB1) should contain BB3");
        assert!(df[2].contains(&3), "DF(BB2) should contain BB3");
        // BB3 没有后继 → DF(BB3) = {}
        assert!(df[3].is_empty(), "DF(BB3) should be empty");
    }

    #[test]
    fn test_dominance_frontier_loop() {
        let cfg = loop_cfg();
        let dt = DomTree::build(&cfg);
        let df = dt.dominance_frontier(&cfg);

        // BB2 → BB1 (back edge), BB1 有两个前驱 (BB0, BB2)
        // DF(BB2) should contain BB1 (BB2 dom BB2 as pred, BB2 doesn't strictly dom BB1)
        assert!(df[2].contains(&1), "DF(BB2) should contain BB1 (loop header)");
        // DF(BB1) should contain BB1 (BB1 dom BB2/pred of BB1, BB1 doesn't strictly dom BB1)
        assert!(df[1].contains(&1), "DF(BB1) should contain BB1 (self-loop)");
        // DF(BB0) is empty — BB0 strictly dominates BB1, so no DF entry
        assert!(df[0].is_empty(), "DF(BB0) should be empty (BB0 strictly dom BB1)");
    }

    // ═══════════════════════════════════════════════════
    //  查询 API 测试
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_dominates_query() {
        let cfg = diamond_cfg();
        let dt = DomTree::build(&cfg);

        // BB0 支配所有块
        assert!(dt.dominates(0, 0), "BB0 dom BB0");
        assert!(dt.dominates(0, 1), "BB0 dom BB1");
        assert!(dt.dominates(0, 2), "BB0 dom BB2");
        assert!(dt.dominates(0, 3), "BB0 dom BB3");

        // BB1 不支配 BB2（独立分支）
        assert!(!dt.dominates(1, 2), "BB1 NOT dom BB2");
        assert!(!dt.dominates(2, 1), "BB2 NOT dom BB1");

        // BB1 不支配 BB3（BB3 也可从 BB2 到达）
        assert!(!dt.dominates(1, 3), "BB1 NOT dom BB3 (diamond)");
        assert!(!dt.dominates(2, 3), "BB2 NOT dom BB3 (diamond)");

        // 严格支配
        assert!(!dt.strictly_dominates(0, 0), "BB0 NOT strictly dom BB0");
        assert!(dt.strictly_dominates(0, 1), "BB0 strictly dom BB1");
    }

    #[test]
    fn test_common_dominator() {
        let cfg = diamond_cfg();
        let dt = DomTree::build(&cfg);

        assert_eq!(dt.common_dominator(1, 2), 0, "NCD(BB1, BB2) = BB0");
        assert_eq!(dt.common_dominator(1, 3), 0, "NCD(BB1, BB3) = BB0");
        assert_eq!(dt.common_dominator(0, 3), 0, "NCD(BB0, BB3) = BB0");
    }

    #[test]
    fn test_dom_children() {
        let cfg = diamond_cfg();
        let dt = DomTree::build(&cfg);

        let mut c0 = dt.dom_children(0);
        c0.sort();
        assert_eq!(c0, vec![1, 2, 3], "BB0 children = (BB1, BB2, BB3)");
        assert!(dt.dom_children(1).is_empty(), "BB1 has no dom children in diamond");
        assert!(dt.dom_children(2).is_empty(), "BB2 has no dom children in diamond");
    }

    #[test]
    fn test_preorder() {
        let cfg = linear_cfg();
        let dt = DomTree::build(&cfg);
        let order = dt.preorder();
        assert_eq!(order, vec![0, 1, 2], "linear preorder = [0, 1, 2]");
    }

    #[test]
    fn test_depth() {
        let cfg = linear_cfg();
        let dt = DomTree::build(&cfg);
        assert_eq!(dt.depth(0), 0, "entry depth = 0");
        assert_eq!(dt.depth(1), 1, "BB1 depth = 1");
        assert_eq!(dt.depth(2), 2, "BB2 depth = 2");
    }

    // ═══════════════════════════════════════════════════
    //  边界情况
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_single_block() {
        let cfg = SimpleCfg::new(1);
        let dt = DomTree::build(&cfg);
        assert_eq!(dt.idom(0), 0);
        assert!(dt.dominates(0, 0));
        assert_eq!(dt.depth(0), 0);
    }

    #[test]
    fn test_empty_cfg() {
        let cfg = SimpleCfg::new(0);
        let dt = DomTree::build(&cfg);
        assert_eq!(dt.num_blocks(), 0);
    }

    #[test]
    fn test_dom_children_nested() {
        let cfg = nested_loop_cfg();
        let dt = DomTree::build(&cfg);

        let mut c0 = dt.dom_children(0);
        c0.sort();
        assert_eq!(c0, vec![1], "BB0 children = (BB1)");

        let mut c1 = dt.dom_children(1);
        c1.sort();
        assert_eq!(c1, vec![2, 5], "BB1 children = (BB2, BB5)");

        let mut c2 = dt.dom_children(2);
        c2.sort();
        assert_eq!(c2, vec![3, 4], "BB2 children = (BB3, BB4)");
    }
}
