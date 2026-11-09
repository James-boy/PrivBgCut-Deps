// PrivBgCut-Deps/ort-test/src/main.rs
//
// 独立测试程序：验证某个 libonnxruntime.dylib 是否能加载，
// 是否能对指定 ONNX 模型跑推理，输出是否合理。
//
// 输入优先级：
//   1. 若设置了 TEST_IMAGE_PATH 且文件存在，则用真实图片（resize + ImageNet 归一化）
//   2. 否则使用确定性伪随机输入（保证可复现，且不引入 rand 依赖）
//
// 用法：
//   ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
//   TEST_IMAGE_PATH=/path/to/input.jpg \
//     ./ort_dylib_test model1.onnx model2.onnx ...

use std::path::Path;

use image::GenericImageView;
use ort::session::Session;
use ort::value::Tensor;

fn main() {
    // 显式从 ORT_DYLIB_PATH 加载 dylib，避免依赖 ort 的隐式加载行为
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        println!("[test] ORT_DYLIB_PATH = {}", path);
        if Path::new(&path).exists() {
            match ort::init_from(&path) {
                Ok(builder) => {
                    builder.commit();
                    println!("[test] dylib 显式加载成功");
                }
                Err(e) => {
                    eprintln!("[test] ✗ dylib 显式加载失败: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            eprintln!("[test] ✗ ORT_DYLIB_PATH 指向的文件不存在: {}", path);
            std::process::exit(1);
        }
    } else {
        println!("[test] 未设置 ORT_DYLIB_PATH，使用默认搜索");
    }

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
        eprintln!(
            "[test] 失败 {} / {}: {:?}",
            failures.len(),
            args.len(),
            failures
        );
        std::process::exit(1);
    }
}

fn test_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(model_path).exists() {
        return Err(format!("模型文件不存在: {}", model_path).into());
    }

    // 按文件名推测输入尺寸（测试够用即可）
    let side: u32 = if model_path.contains("u2netp") {
        320
    } else {
        512
    };
    println!("[test] 方形输入尺寸: {}x{}", side, side);

    // ── 构造输入像素（真实图片优先，失败回退随机）────────────
    let mut pixels: Vec<u8> = Vec::new();
    let mut used_real_image = false;

    if let Ok(img_path) = std::env::var("TEST_IMAGE_PATH") {
        if Path::new(&img_path).exists() {
            println!("[test] 尝试加载真实测试图片: {}", img_path);
            match image::open(&img_path) {
                Ok(img) => {
                    let resized =
                        img.resize_exact(side, side, image::imageops::FilterType::Triangle);
                    let rgb = resized.to_rgb8();
                    pixels = rgb.into_raw();
                    used_real_image = true;
                    println!("[test] 真实图片加载成功，尺寸: {:?}", resized.dimensions());
                }
                Err(e) => {
                    println!("[test] 图片加载失败: {}，回退到伪随机输入", e);
                }
            }
        } else {
            println!(
                "[test] TEST_IMAGE_PATH 指向的文件不存在: {}，回退到伪随机输入",
                img_path
            );
        }
    } else {
        println!("[test] 未设置 TEST_IMAGE_PATH，使用伪随机输入");
    }

    if !used_real_image {
        pixels = vec![0u8; (side * side * 3) as usize];
        for y in 0..side {
            for x in 0..side {
                let idx = ((y * side + x) * 3) as usize;
                pixels[idx] = ((x * 7 + y * 13) & 0xff) as u8;
                pixels[idx + 1] = ((x * 31 + y * 17) & 0xff) as u8;
                pixels[idx + 2] = ((x * 3 + y * 29) & 0xff) as u8;
            }
        }
        println!("[test] 已生成 {}x{} 确定性伪随机输入", side, side);
    }

    // ImageNet 归一化（u2netp / birefnet 通用；归一化差异不影响加载验证）
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

    // ── 加载 Session（触发 ONNX Runtime 算子解析，DeformConv 缺失会在这里报错）──
    println!("[test] 加载 Session…");
    let t0 = std::time::Instant::now();
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?
        .commit_from_file(model_path)?;
    println!("[test] Session 加载耗时: {:?}", t0.elapsed());

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
    println!("[test] input name:  {}", input_name);
    println!("[test] output name: {}", output_name);

    // ── 推理 ────────────────────────────────────────────────
    let shape = [1usize, 3, side as usize, side as usize];
    let input_tensor = Tensor::from_array((shape, input_data))?;

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
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
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
