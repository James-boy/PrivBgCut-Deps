// src-tauri/src/bin/ort_dylib_test.rs
//
// 独立测试二进制：用于验证某个 libonnxruntime.dylib 是否能加载、
// 是否能对指定 ONNX 模型跑推理，并检查输出合理性（NaN/Inf/退化）。
//
// 与 src/bin/ort_smoke_test.rs 的区别：
//   - ort_smoke_test 只测 u2netp，用于 universal CI 里的 ARM 门禁
//   - 本程序接受任意模型列表，用于验证自定义编译的 dylib（如
//     PR #27393 的 DeformConv CPU 支持）
//
// 用法：
//   ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
//     cargo run --release --features dylib-test --bin ort_dylib_test \
//     -- models/u2netp.onnx models/birefnet-lite-dynamic.onnx

use std::path::Path;

fn main() {
    let dylib_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "（未设置）".into());
    println!("[test] ORT_DYLIB_PATH = {}", dylib_path);

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("用法: ort_dylib_test <model1.onnx> [model2.onnx ...]");
        std::process::exit(1);
    }

    let mut failures: Vec<String> = Vec::new();

    for model_path in &args {
        println!("\n[test] ════════════════════════════════════════════");
        println!("[test] 测试模型: {}", model_path);
        println!("[test] ════════════════════════════════════════════");
        match test_model(model_path) {
            Ok(()) => println!("[test] ✓ PASS: {}", model_path),
            Err(e) => {
                eprintln!("[test] ✗ FAIL: {} — {}", model_path, e);
                failures.push(model_path.clone());
            }
        }
    }

    println!("\n[test] ════════════════════════════════════════════");
    if failures.is_empty() {
        println!("[test] 全部 {} 个模型通过 ✓", args.len());
    } else {
        eprintln!("[test] 失败 {} / {}: {:?}", failures.len(), args.len(), failures);
        std::process::exit(1);
    }
}

fn test_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(model_path).exists() {
        return Err(format!("模型文件不存在: {}", model_path).into());
    }

    // 按文件名猜测输入尺寸（不做完整的 ModelSpec 反查，测试够用即可）
    let side: u32 = if model_path.contains("u2netp") {
        320
    } else if model_path.contains("ppmatting") {
        512
    } else {
        512 // birefnet 默认；DynamicSquare 会自适应
    };
    println!("[test] 使用方形输入: {}x{}", side, side);

    // 构造 deterministic 的伪随机输入（避免引入 rand 依赖）
    let mut pixels = vec![0u8; (side * side * 3) as usize];
    for y in 0..side {
        for x in 0..side {
            let idx = ((y * side + x) * 3) as usize;
            pixels[idx]     = ((x * 7 + y * 13) & 0xff) as u8;
            pixels[idx + 1] = ((x * 31 + y * 17) & 0xff) as u8;
            pixels[idx + 2] = ((x * 3 + y * 29) & 0xff) as u8;
        }
    }

    // ImageNet 归一化（u2netp / birefnet 通用；ppmatting 用 (0.5,0.5) 但我们
    // 的目标是验证 dylib 可加载，不是复现业务 pipeline，归一化差异不影响）
    let mean = [0.485f32, 0.456, 0.406];
    let std_dev = [0.229f32, 0.224, 0.225];
    let hw = (side * side) as usize;
    let mut input_data = vec![0f32; 3 * hw];
    for i in 0..hw {
        for c in 0..3 {
            let v = pixels[i * 3 + c] as f32 / 255.0;
            input_data[c * hw + i] = (v - mean[c]) / std_dev[c];
        }
    }

    // ── 加载 Session（这一步就会触发 ONNX Runtime 的算子解析）──
    println!("[test] 加载 Session…");
    let t0 = std::time::Instant::now();
    let mut session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?
        .commit_from_file(model_path)?;
    println!("[test] Session 加载耗时: {:?}", t0.elapsed());

    // 打印 IO 信息
    for i in session.inputs() {
        println!("[test] input:  {}  {:?}", i.name(), i.input_type());
    }
    for o in session.outputs() {
        println!("[test] output: {}  {:?}", o.name(), o.output_type());
    }

    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or("模型没有声明输入")?;
    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or("模型没有声明输出")?;

    // ── 推理 ────────────────────────────────────────────────
    let shape = [1usize, 3, side as usize, side as usize];
    let input_tensor = ort::value::Tensor::from_array((shape, input_data))?;

    println!("[test] 运行推理…");
    let t1 = std::time::Instant::now();
    let outputs = session.run(ort::inputs![input_name => input_tensor])?;
    println!("[test] 推理耗时: {:?}", t1.elapsed());

    // ── 检查输出 ────────────────────────────────────────────
    let output = outputs
        .get(output_name.as_str())
        .ok_or("输出 tensor 不存在")?;
    let (out_shape, data) = output.try_extract_tensor::<f32>()?;
    println!("[test] 输出 shape: {:?}, len: {}", out_shape, data.len());

    if data.is_empty() {
        return Err("输出为空".into());
    }

    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for v in data.iter() {
        if v.is_nan() { nan_count += 1; }
        if v.is_infinite() { inf_count += 1; }
    }
    if nan_count > 0 {
        return Err(format!("输出含 {} 个 NaN", nan_count).into());
    }
    if inf_count > 0 {
        return Err(format!("输出含 {} 个 Inf", inf_count).into());
    }

    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_val: f32 = data.iter().sum::<f32>() / data.len() as f32;
    println!(
        "[test] 输出统计: min={:.4}, max={:.4}, mean={:.4}",
        min, max, mean_val
    );

    if (max - min).abs() < 1e-9 {
        return Err(format!("输出全为同一值 {}（疑似推理无效）", min).into());
    }

    Ok(())
}
