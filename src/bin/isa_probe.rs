//! ISA Probe CLI — GFX11 指令编码查询 + 代码生成工具
//!
//! 用法:
//!   isa_probe query "v_sin_f32" [--target gfx1100]
//!   isa_probe discover           [--target gfx1100]
//!   isa_probe unused             [--target gfx1100]
//!   isa_probe codegen v_sin_f32 v_cos_f32  [--output dir] [--target gfx1100]
//!   isa_probe codegen --all-unused         [--output dir] [--target gfx1100]
//!   isa_probe codegen --category VOP1      [--output dir] [--target gfx1100]

use t0_gpu::t0::isa_probe;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let command = &args[1];
    let target = get_flag(&args, "--target").unwrap_or_else(|| "gfx1100".to_string());

    match command.as_str() {
        "query" => cmd_query(&args, &target),
        "discover" => cmd_discover(&target),
        "unused" => cmd_unused(&target),
        "codegen" => cmd_codegen(&args, &target),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
        }
    }
}

fn print_usage() {
    eprintln!("ISA Probe — GFX11 指令编码查询 + 代码生成工具");
    eprintln!();
    eprintln!("用法:");
    eprintln!("  isa_probe query <mnemonic>           [--target gfxXXXX]  查询单条指令编码");
    eprintln!("  isa_probe discover                   [--target gfxXXXX]  发现全部可用指令");
    eprintln!("  isa_probe unused                     [--target gfxXXXX]  输出可用但未使用的指令");
    eprintln!("  isa_probe codegen <mn1> <mn2> ...    [--output dir] [--target gfxXXXX]");
    eprintln!("  isa_probe codegen --all-unused       [--output dir]      生成全部未使用指令");
    eprintln!("  isa_probe codegen --category <CAT>   [--output dir]      按类别批量生成");
    eprintln!();
    eprintln!("类别: VOP1, VOP2, VOP3, VOPC, SOP1, SOP2, SOPC, SOPP, SMEM, DS, FLAT, WMMA");
    eprintln!();
    eprintln!("示例:");
    eprintln!("  isa_probe query v_sin_f32");
    eprintln!("  isa_probe codegen v_sin_f32 v_cos_f32 --output src/t0/generated/");
    eprintln!("  isa_probe codegen --all-unused --output src/t0/generated/");
    eprintln!("  isa_probe codegen --category VOP1 --output src/t0/generated/");
}

fn get_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() - 1 {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ── query: 查询单条指令 ──

fn cmd_query(args: &[String], target: &str) {
    if args.len() < 3 {
        eprintln!("用法: isa_probe query <mnemonic>");
        return;
    }
    let mnemonic = &args[2];

    let db = isa_probe::gfx11_instruction_db();
    let template = db.iter().find(|t| t.mnemonic == *mnemonic);

    let test_asm = if let Some(t) = template {
        t.test_asm.clone()
    } else {
        let guesses = vec![
            format!("{} v0, v1", mnemonic),
            format!("{} v0, v1, v2", mnemonic),
            format!("{} s0, s1, s2", mnemonic),
            format!("{} s0, s1", mnemonic),
            format!("{}", mnemonic),
        ];
        let mut found = None;
        for g in &guesses {
            if isa_probe::probe_instruction(g, target).is_ok() {
                found = Some(g.clone());
                break;
            }
        }
        match found {
            Some(asm) => asm,
            None => {
                eprintln!("错误: 指令 '{}' 在 {} 上不可用", mnemonic, target);
                return;
            }
        }
    };

    match isa_probe::probe_instruction(&test_asm, target) {
        Ok(info) => {
            println!("✅ {} — {} verified", mnemonic, target);
            println!("   canonical: {}", info.canonical);
            println!("   format:    {}", info.format);
            println!("   encoding:  {:?} ({} bytes)", info.encoding, info.n_bytes);
            println!("   operands:  {}", info.operand_sig);
            let enc_hex: Vec<String> = info.encoding.iter().map(|b| format!("0x{:02X}", b)).collect();
            println!("   hex:       [{}]", enc_hex.join(", "));
            let implemented = isa_probe::implemented_mnemonics();
            if implemented.contains(mnemonic) {
                println!("   status:    ✅ 已在 T0 中实现");
            } else {
                println!("   status:    ❌ 未在 T0 中实现");
            }
        }
        Err(e) => eprintln!("错误: {}", e),
    }
}

// ── discover: 发现全部可用指令 ──

fn cmd_discover(target: &str) {
    println!("正在探测 {} 支持的全部指令...\n", target);
    let db = isa_probe::gfx11_instruction_db();
    let implemented = isa_probe::implemented_mnemonics();

    let mut available = 0;
    let mut errors = 0;
    let mut by_format: std::collections::HashMap<isa_probe::IsaFormat, Vec<String>> =
        std::collections::HashMap::new();

    for template in &db {
        match isa_probe::probe_instruction(&template.test_asm, target) {
            Ok(info) => {
                available += 1;
                let status = if implemented.contains(&template.mnemonic) { "✅" } else { "  " };
                println!("{} {}", status, isa_probe::format_instruction(&info));
                by_format.entry(info.format).or_default().push(template.mnemonic.clone());
            }
            Err(_) => { errors += 1; }
        }
    }

    println!("\n─── 统计 ───");
    println!("已测试: {} 条", db.len());
    println!("可用:   {} 条", available);
    println!("无效:   {} 条", errors);
    println!("已实现: {} 条", implemented.len());

    println!("\n按格式分布:");
    let mut formats: Vec<_> = by_format.iter().collect();
    formats.sort_by_key(|(f, _)| format!("{:?}", f));
    for (format, instrs) in &formats {
        println!("  {:>8}: {} 条", format.to_string(), instrs.len());
    }
}

// ── unused: 输出未使用指令清单 ──

fn cmd_unused(target: &str) {
    println!("正在探测 {} 的可用但未使用指令...\n", target);
    let db = isa_probe::gfx11_instruction_db();
    let (available, unused) = isa_probe::discover_and_diff(target);
    let report = isa_probe::format_unused_report(&unused, &db);
    println!("{}", report);
    println!("─── 摘要 ───");
    println!("可用总数: {} 条", available.len());
    println!("已实现:   {} 条", available.len() - unused.len());
    println!("未使用:   {} 条", unused.len());
}

// ── codegen: 生成 Rust 代码（支持单条/批量/全部未使用/按类别） ──

fn cmd_codegen(args: &[String], target: &str) {
    let output_dir = get_flag(args, "--output");

    // Determine which mnemonics to generate
    let mnemonics: Vec<String> = if has_flag(args, "--all-unused") {
        // All unused instructions
        eprintln!("正在发现 {} 的全部未使用指令...", target);
        let (_, unused) = isa_probe::discover_and_diff(target);
        unused.iter().map(|i| i.mnemonic.clone()).collect()
    } else if let Some(category) = get_flag(args, "--category") {
        // Filter by category
        eprintln!("正在生成 {} 类别的未使用指令...", category);
        let db = isa_probe::gfx11_instruction_db();
        let implemented = isa_probe::implemented_mnemonics();
        let category_upper = category.to_uppercase();
        db.iter()
            .filter(|t| t.category.to_uppercase() == category_upper)
            .filter(|t| !implemented.contains(&t.mnemonic))
            .map(|t| t.mnemonic.clone())
            .collect()
    } else {
        // Explicit mnemonic list: positional args after "codegen", skipping --flag and their values
        let mut result = Vec::new();
        let mut skip_next = false;
        for a in &args[2..] {
            if skip_next {
                skip_next = false;
                continue;
            }
            if a.starts_with("--") {
                skip_next = true; // skip the flag's value
                continue;
            }
            result.push(a.clone());
        }
        result
    };

    if mnemonics.is_empty() {
        eprintln!("用法: isa_probe codegen <mn1> [mn2 ...] [--output dir]");
        eprintln!("      isa_probe codegen --all-unused [--output dir]");
        eprintln!("      isa_probe codegen --category VOP1 [--output dir]");
        return;
    }

    // Single instruction: use old behavior (print to stdout)
    if mnemonics.len() == 1 && output_dir.is_none() {
        let db = isa_probe::gfx11_instruction_db();
        let template = db.iter().find(|t| t.mnemonic == mnemonics[0]);
        let test_asm = if let Some(t) = template {
            t.test_asm.clone()
        } else {
            format!("{} v0, v1", mnemonics[0])
        };
        match isa_probe::probe_instruction(&test_asm, target) {
            Ok(info) => println!("{}", isa_probe::generate_rust_code(&info)),
            Err(e) => eprintln!("错误: {}", e),
        }
        return;
    }

    // Batch codegen
    eprintln!("正在批量生成 {} 条指令的代码...", mnemonics.len());
    let batch = isa_probe::generate_batch(&mnemonics, target);

    if let Some(dir) = &output_dir {
        // Write to files
        match isa_probe::write_codegen_files(&batch, dir) {
            Ok(()) => {
                println!("✅ 代码已生成到: {}/", dir);
                println!("   ir_ops.rs      — {} 条 Op enum variant", batch.count);
                println!("   asm_emitter.rs — {} 条 emit_op match arm", batch.count);
                println!("   summary.md     — 编码摘要表");
                if !batch.errors.is_empty() {
                    println!("   ⚠️  {} 条指令失败:", batch.errors.len());
                    for e in &batch.errors {
                        println!("      - {}", e);
                    }
                }
            }
            Err(e) => eprintln!("错误: {}", e),
        }
    } else {
        // Print to stdout
        println!("{}", batch.ir_ops);
        println!("{}", batch.emitter_code);
        eprintln!("─── {} 条指令生成完成 ───", batch.count);
        if !batch.errors.is_empty() {
            eprintln!("⚠️  {} 条失败:", batch.errors.len());
            for e in &batch.errors {
                eprintln!("  - {}", e);
            }
        }
    }
}
