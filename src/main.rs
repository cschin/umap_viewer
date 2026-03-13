use umap_viewer::UmapApp;

// ---------------------------------------------------------------------------
// Web entry point
// ---------------------------------------------------------------------------
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();

    let canvas = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document")
        .get_element_by_id("the_canvas_id")
        .expect("canvas #the_canvas_id not found")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is not a canvas");

    let web_options = eframe::WebOptions {
        wgpu_options: egui_wgpu::WgpuConfiguration {
            supported_backends: wgpu::Backends::GL,
            ..Default::default()
        },
        ..Default::default()
    };

    wasm_bindgen_futures::spawn_local(async move {
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(UmapApp::new(cc)))),
            )
            .await
            .expect("failed to start eframe");
    });
}

// ---------------------------------------------------------------------------
// Native entry point
// ---------------------------------------------------------------------------
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init();

    let config = umap_viewer::config::Config::from_args()
        .expect("failed to load config (use --config <path>, default: config.yaml)");

    // Verify data files exist before launching the GUI.
    for path in [&config.coords_parquet, &config.labels_parquet] {
        if !std::path::Path::new(path).exists() {
            eprintln!("error: data file not found: {path}");
            eprintln!("hint: check the paths in config.yaml or use --config <path>");
            std::process::exit(1);
        }
    }

    // --export-bin: write points.bin for the WASM build, then exit.
    if config.export_bin {
        umap_viewer::data::PointCloud::export_to_bin(
            &config.coords_parquet,
            &config.labels_parquet,
            &config.output_bin,
        ).expect("export failed");
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("UMAP Viewer")
            .with_inner_size([1280.0, 800.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "UMAP Viewer",
        native_options,
        Box::new(move |cc| Ok(Box::new(UmapApp::with_config(cc, &config)))),
    )
}
