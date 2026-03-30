# GPU 连续运行硬挂防御 — 实验记录

## 日期
2026-03-25

## 目标
解决连续执行两次 `test_gpu_gemm_tn`（通过 `| head` 管道）百分之百触发 GPU 硬挂的问题。

## 根因分析

通过 dmesg 日志和代码审计确认三层因果链：

1. **ISA 内核 Bug**：`for_range` + LDS 累加路径生成错误指令 → GPU 卡死
2. **SIGPIPE 杀进程**：`| head -N` 管道断开 → SIGPIPE → 进程死亡 → Drop 未执行 → GPU 队列未回收
3. **MODE1 Reset 竞态**：KFD 驱动触发 GPU reset（~3s），第二次进程在恢复窗口内启动 → 硬挂

关键 dmesg 证据：
```
amdgpu: MES might be in unrecoverable state, issue a GPU reset
amdgpu: MODE1 reset
amdgpu: VRAM is lost due to GPU reset!
amdgpu: GPU reset(20) succeeded!
```

## 实施的修复

在 `kfd/mod.rs` 中实现三层防御：

### 1. SIGPIPE 忽略（阻断因果链 ②）
```rust
fn ignore_sigpipe() {
    extern "C" { fn signal(sig: i32, handler: usize) -> usize; }
    unsafe { signal(13, 1); } // SIGPIPE=13, SIG_IGN=1
}
```
在 `open_device_impl()` 入口调用，确保管道断开时进程不被杀死，Drop 清理有机会执行。

### 2. KFD open 重试（覆盖 reset 窗口 ③）
```rust
fn open_kfd_with_retry() -> Result<RawFd, String>
```
最多重试 5 次，每次间隔 1s，覆盖 MODE1 reset 的 ~3s 恢复时间。

### 3. GPU 健康探针（早期检测 ③）
```rust
fn gpu_health_probe(device: &Arc<Self>) -> Result<(), String>
```
启动时分配 GTT buffer → 写入模式 → 读回验证。最多重试 3 次（间隔 2s），捕获 VRAM 丢失后的不稳定状态。

## 结果
- 编译通过（`cargo build --release --features rocm --lib`）
- 无新 error，仅 22 个无关 warning

## 结论

| 防御层 | 作用 | 状态 |
|--------|------|------|
| SIGPIPE 忽略 | 防止管道杀进程，让 Drop 执行 | ✅ 已实现 |
| KFD open 重试 | 覆盖 GPU reset 恢复窗口 | ✅ 已实现 |
| GPU 健康探针 | 早期检测 VRAM 不稳定 | ✅ 已实现 |

> **铁律**：任何通过 `| head` / `| tail` 管道运行 GPU 程序的场景，都必须忽略 SIGPIPE，否则进程可能在 GPU 队列活跃时被杀死。

## 后续
- 用户需验证连续运行是否不再硬挂
- ISA 内核 bug（for_range LICM 死代码）仍需持续修复——这是根本原因
- 防御措施只是减轻后果，不是替代内核正确性
