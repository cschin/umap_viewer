use std::sync::Arc;

use egui::{Context, Pos2, Vec2};
use egui_extras::{Column, TableBuilder};
use egui_wgpu::wgpu;

use crate::data::PointCloud;
use crate::point_renderer::{build_transform, PointRenderer, Uniforms};

// ---------------------------------------------------------------------------
// wgpu paint callback
// ---------------------------------------------------------------------------

struct PointsCallback {
    transform: [[f32; 4]; 4],
    point_size: f32,
    viewport_aspect: f32,
    alpha: f32,
}

pub struct PointsCallbackResources {
    pub renderer: PointRenderer,
}

impl egui_wgpu::CallbackTrait for PointsCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get::<PointsCallbackResources>() {
            let u = Uniforms {
                transform: self.transform,
                point_size: self.point_size,
                viewport_aspect: self.viewport_aspect,
                alpha: self.alpha,
                _pad: [0.0; 1],
            };
            queue.write_buffer(res.renderer.uniform_buf(), 0, bytemuck::bytes_of(&u));
        }
        vec![]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(res) = resources.get::<PointsCallbackResources>() {
            // SAFETY: PointsCallbackResources is stored in CallbackResources which
            // is owned by the wgpu renderer and outlives every paint callback.
            let res: &'static PointsCallbackResources = unsafe { std::mem::transmute(res) };
            res.renderer.draw(rpass);
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction mode
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum Mode {
    Navigate,
    SelectPolygon,
}

// ---------------------------------------------------------------------------
// Table sort state
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum SortCol {
    Row,
    Category,
    Label,
    X,
    Y,
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

pub struct UmapApp {
    cloud: PointCloud,
    // view state
    pan: Vec2,
    zoom: f32,
    point_size: f32,
    alpha: f32,
    // navigate mode
    drag_start: Option<(Pos2, Vec2)>,
    // hover tooltip
    hover_data_pos: Option<(f32, f32)>,
    hovered_point: Option<usize>,
    // selection mode
    mode: Mode,
    poly_verts: Vec<[f32; 2]>,                // data-space polygon vertices
    poly_closed: bool,                        // true after selection is confirmed
    selected_indices: Vec<usize>,             // indices into cloud.points
    focused_category: Option<String>,         // category highlighted via histogram click
    focus_size_scale: f32,                    // size multiplier for focused points
    category_histogram: Vec<(String, usize)>, // (category, count) sorted descending
    pinned_point: Option<usize>,              // point index pinned by clicking a table row
    histogram_visible: bool,
    table_visible: bool,
    left_panel_visible: bool,
    // table sort state
    table_sort_col: SortCol,
    table_sort_asc: bool,
    sorted_rows: Vec<usize>, // permutation of 0..selected_indices.len()
    // wgpu queue for point buffer uploads
    wgpu_queue: Arc<wgpu::Queue>,
    // label file selector
    label_files: Vec<(String, String)>,     // (display name, path)
    all_label_categories: Vec<Vec<String>>, // pre-loaded categories per file
    color_maps: Vec<crate::data::ColorMap>, // per-file colour maps (empty = hue default)
    selected_label_idx: usize,
}

// ---------------------------------------------------------------------------
// Initialisation helpers (free functions used only by the constructors)
// ---------------------------------------------------------------------------

/// Register SFNSMono as a fallback font for Unicode symbols (▲▼→ etc.).
fn register_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "SFNSMono".to_owned(),
        egui::FontData::from_static(include_bytes!("../fonts/SFNSMono.ttf")),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("SFNSMono".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("SFNSMono".to_owned());
    ctx.set_fonts(fonts);
}

/// Create the wgpu point renderer and register it in the egui-wgpu callback
/// resources. Returns the queue handle needed for later buffer uploads.
fn init_wgpu_renderer(
    cc: &eframe::CreationContext<'_>,
    points: &[crate::data::Point],
) -> Arc<wgpu::Queue> {
    let wgpu_rs = cc.wgpu_render_state.as_ref().expect("wgpu render state");
    let renderer = PointRenderer::new(&wgpu_rs.device, wgpu_rs.target_format, points);
    wgpu_rs
        .renderer
        .write()
        .callback_resources
        .insert(PointsCallbackResources { renderer });
    wgpu_rs.queue.clone()
}

/// Load the primary PointCloud from parquet files described by `config`.
#[cfg(not(target_arch = "wasm32"))]
fn load_point_cloud_native(
    config: &crate::config::Config,
    label_pairs: &[(String, String)],
) -> PointCloud {
    let primary_path = label_pairs.first().map(|(_, p)| p.as_str()).unwrap_or("");
    PointCloud::load_from_parquet(&config.coords_parquet, primary_path)
        .expect("failed to load parquet data")
}

/// Pre-load every label file into `Vec<String>` category vectors so that
/// switching label sets in the UI is instant.
#[cfg(not(target_arch = "wasm32"))]
fn load_all_label_categories(
    cloud: &mut PointCloud,
    label_pairs: &[(String, String)],
) -> Vec<Vec<String>> {
    label_pairs
        .iter()
        .map(|(_, path)| {
            cloud
                .load_categories_from_parquet(path)
                .unwrap_or_else(|e| {
                    eprintln!("Warning: could not load {path}: {e}");
                    vec![String::new(); cloud.points.len()]
                })
        })
        .collect()
}

/// Build one `ColorMap` per label set: load from CSV if configured, otherwise
/// derive evenly-spaced hues from the unique category names.
#[cfg(not(target_arch = "wasm32"))]
fn build_color_maps(
    label_pairs: &[(String, String)],
    all_label_categories: &[Vec<String>],
    config: &crate::config::Config,
) -> Vec<crate::data::ColorMap> {
    label_pairs
        .iter()
        .zip(all_label_categories.iter())
        .map(|((name, _), cats)| {
            if let Some(csv_path) = config.color_file_for(name) {
                PointCloud::load_color_csv(csv_path).unwrap_or_else(|e| {
                    eprintln!("Warning: could not load color CSV {csv_path}: {e}");
                    crate::data::ColorMap::new()
                })
            } else {
                let mut order: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                for cat in cats {
                    let n = order.len();
                    order.entry(cat.as_str()).or_insert(n);
                }
                let n_cats = order.len().max(1);
                order
                    .iter()
                    .map(|(&name, &idx)| {
                        (
                            name.to_string(),
                            crate::data::hue_to_rgb(idx as f32 / n_cats as f32),
                        )
                    })
                    .collect()
            }
        })
        .collect()
}

/// Store pre-loaded label sets and colour maps on the cloud, then apply the
/// first label set so point colours are ready for the initial frame.
#[cfg(not(target_arch = "wasm32"))]
fn apply_label_sets_to_cloud(
    cloud: &mut PointCloud,
    label_pairs: &[(String, String)],
    all_label_categories: &[Vec<String>],
    color_maps: &[crate::data::ColorMap],
) {
    cloud.label_set_names = label_pairs.iter().map(|(name, _)| name.clone()).collect();
    cloud.all_categories = all_label_categories.to_vec();
    cloud.category_color_maps = color_maps.to_vec();
    if let (Some(cats), Some(cmap)) = (all_label_categories.first(), color_maps.first()) {
        cloud.apply_categories(cats.clone(), cmap);
    }
}

impl UmapApp {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Shared initialisation once the PointCloud is ready.
    fn build(cc: &eframe::CreationContext<'_>, cloud: PointCloud) -> Self {
        register_fonts(&cc.egui_ctx);
        let wgpu_queue = init_wgpu_renderer(cc, &cloud.points);
        Self::default_state(wgpu_queue, cloud)
    }

    /// Native constructor: load data from the paths in `config`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_config(cc: &eframe::CreationContext<'_>, config: &crate::config::Config) -> Self {
        let label_pairs = config.label_pairs();
        let mut cloud = load_point_cloud_native(config, &label_pairs);
        let all_label_categories = load_all_label_categories(&mut cloud, &label_pairs);
        let color_maps = build_color_maps(&label_pairs, &all_label_categories, config);
        apply_label_sets_to_cloud(&mut cloud, &label_pairs, &all_label_categories, &color_maps);

        let mut app = Self::build(cc, cloud);
        app.label_files = label_pairs;
        app.all_label_categories = all_label_categories;
        app.color_maps = color_maps;
        app
    }

    /// WASM constructor: load data from the embedded binary blob.
    #[cfg(target_arch = "wasm32")]
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let cloud = PointCloud::from_bin(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/points.bin"
        )))
        .expect("failed to parse embedded points.bin");

        let label_files = cloud
            .label_set_names
            .iter()
            .map(|n| (n.clone(), String::new()))
            .collect();
        let all_label_categories = cloud.all_categories.clone();
        let color_maps = cloud.category_color_maps.clone();

        let mut app = Self::build(cc, cloud);
        app.label_files = label_files;
        app.all_label_categories = all_label_categories;
        app.color_maps = color_maps;
        app
    }

    /// Build a fresh `UmapApp` with all fields at their default/initial values.
    fn default_state(wgpu_queue: Arc<wgpu::Queue>, cloud: PointCloud) -> Self {
        Self {
            cloud,
            pan: Vec2::ZERO,
            zoom: 1.0,
            point_size: 4.0,
            alpha: 0.5,
            drag_start: None,
            hover_data_pos: None,
            hovered_point: None,
            mode: Mode::Navigate,
            poly_verts: Vec::new(),
            poly_closed: false,
            selected_indices: Vec::new(),
            focused_category: None,
            focus_size_scale: 2.0,
            category_histogram: Vec::new(),
            pinned_point: None,
            histogram_visible: true,
            table_visible: true,
            left_panel_visible: true,
            table_sort_col: SortCol::Row,
            table_sort_asc: true,
            sorted_rows: Vec::new(),
            wgpu_queue,
            label_files: Vec::new(),
            all_label_categories: Vec::new(),
            color_maps: Vec::new(),
            selected_label_idx: 0,
        }
    }

    fn screen_to_data(&self, screen: Pos2, rect: egui::Rect) -> (f32, f32) {
        let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
        let cx = (xmin + xmax) * 0.5 + self.pan.x;
        let cy = (ymin + ymax) * 0.5 + self.pan.y;
        let aspect = rect.width() / rect.height();
        let span_x = (xmax - xmin) * 0.5 / self.zoom;
        let span_y = (ymax - ymin) * 0.5 / self.zoom;
        let (half_x, half_y) = if aspect >= 1.0 {
            (span_x * aspect, span_y)
        } else {
            (span_x, span_y / aspect)
        };
        let nx = (screen.x - rect.left()) / rect.width() * 2.0 - 1.0;
        let ny = -((screen.y - rect.top()) / rect.height() * 2.0 - 1.0);
        (cx + nx * half_x, cy + ny * half_y)
    }

    fn data_to_screen(&self, dx: f32, dy: f32, rect: egui::Rect) -> Pos2 {
        let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
        let cx = (xmin + xmax) * 0.5 + self.pan.x;
        let cy = (ymin + ymax) * 0.5 + self.pan.y;
        let aspect = rect.width() / rect.height();
        let span_x = (xmax - xmin) * 0.5 / self.zoom;
        let span_y = (ymax - ymin) * 0.5 / self.zoom;
        let (half_x, half_y) = if aspect >= 1.0 {
            (span_x * aspect, span_y)
        } else {
            (span_x, span_y / aspect)
        };
        let nx = (dx - cx) / half_x;
        let ny = (dy - cy) / half_y;
        Pos2 {
            x: rect.left() + (nx + 1.0) / 2.0 * rect.width(),
            y: rect.top() + (-ny + 1.0) / 2.0 * rect.height(),
        }
    }

    fn hit_radius_data(&self, rect: egui::Rect) -> f32 {
        let [_xmin, _xmax, ymin, ymax] = self.cloud.bounds;
        let span_y = (ymax - ymin) * 0.5 / self.zoom;
        (self.point_size / rect.height()) * span_y
    }

    fn build_category_histogram(&self) -> Vec<(String, usize)> {
        let mut map: std::collections::HashMap<&str, usize> = Default::default();
        for &i in &self.selected_indices {
            let cat = self
                .cloud
                .categories
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("");
            *map.entry(cat).or_insert(0) += 1;
        }
        let mut counts: Vec<(String, usize)> = map
            .into_iter()
            .map(|(k, v)| {
                (
                    if k.is_empty() {
                        "(unlabeled)".to_string()
                    } else {
                        k.to_string()
                    },
                    v,
                )
            })
            .collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts
    }

    fn rebuild_sorted_rows(&mut self) {
        let mut order: Vec<usize> = (0..self.selected_indices.len()).collect();
        let cloud = &self.cloud;
        let selected = &self.selected_indices;
        match self.table_sort_col {
            SortCol::Row => {} // natural order
            SortCol::Category => order.sort_by(|&a, &b| {
                let ca = cloud
                    .categories
                    .get(selected[a])
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let cb = cloud
                    .categories
                    .get(selected[b])
                    .map(|s| s.as_str())
                    .unwrap_or("");
                ca.cmp(cb)
            }),
            SortCol::Label => order.sort_by(|&a, &b| {
                let la = cloud
                    .labels
                    .get(selected[a])
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let lb = cloud
                    .labels
                    .get(selected[b])
                    .map(|s| s.as_str())
                    .unwrap_or("");
                la.cmp(lb)
            }),
            SortCol::X => order.sort_by(|&a, &b| {
                cloud.points[selected[a]]
                    .x
                    .partial_cmp(&cloud.points[selected[b]].x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortCol::Y => order.sort_by(|&a, &b| {
                cloud.points[selected[a]]
                    .y
                    .partial_cmp(&cloud.points[selected[b]].y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
        }
        if !self.table_sort_asc {
            order.reverse();
        }
        self.sorted_rows = order;
    }

    /// Apply category focus highlight: within the selection, only points in
    /// `focused_category` (if set) get highlight=1.0; all others get 0.15.
    /// Clears focus when called with None (restores normal selection highlight).
    fn apply_category_focus(&mut self, frame: &eframe::Frame) {
        let selected_set: std::collections::HashSet<usize> =
            self.selected_indices.iter().copied().collect();
        for (i, p) in self.cloud.points.iter_mut().enumerate() {
            let (highlight, size) = match &self.focused_category {
                Some(cat) => {
                    let in_sel = selected_set.contains(&i);
                    let in_cat = self
                        .cloud
                        .categories
                        .get(i)
                        .map(|c| {
                            if cat == "(unlabeled)" {
                                c.is_empty()
                            } else {
                                c == cat
                            }
                        })
                        .unwrap_or(false);
                    if in_sel && in_cat {
                        (1.0, self.focus_size_scale)
                    } else {
                        (0.15, 1.0)
                    }
                }
                None => {
                    let in_sel = selected_set.is_empty() || selected_set.contains(&i);
                    let unlabeled = self
                        .cloud
                        .categories
                        .get(i)
                        .map(|c| c.is_empty())
                        .unwrap_or(false);
                    let base_h = if in_sel { 1.0 } else { 0.15 };
                    let h = if unlabeled { base_h * 0.5 } else { base_h };
                    (h, 1.0)
                }
            };
            p.highlight = highlight;
            p.size = size;
        }
        self.upload_points(frame);
    }

    fn shuffle_colors(&mut self, frame: &eframe::Frame) {
        use rand::seq::SliceRandom;
        let idx = self.selected_label_idx;
        let cats = match self.all_label_categories.get(idx) {
            Some(c) => c.clone(),
            None => return,
        };

        // Collect unique category names in first-seen order.
        let mut seen = std::collections::HashSet::new();
        let mut unique: Vec<&str> = Vec::new();
        for c in &cats {
            if seen.insert(c.as_str()) {
                unique.push(c.as_str());
            }
        }
        let n = unique.len();
        if n == 0 {
            return;
        }

        // Shuffle hue indices and build a new ColorMap.
        let mut hue_indices: Vec<usize> = (0..n).collect();
        hue_indices.shuffle(&mut rand::thread_rng());
        let new_cmap: crate::data::ColorMap = unique
            .iter()
            .zip(hue_indices.iter())
            .map(|(&name, &hi)| {
                (
                    name.to_string(),
                    crate::data::hue_to_rgb(hi as f32 / n as f32),
                )
            })
            .collect();

        // Persist the new map and re-apply colours to the cloud.
        while self.color_maps.len() <= idx {
            self.color_maps.push(crate::data::ColorMap::new());
        }
        self.color_maps[idx] = new_cmap.clone();
        self.cloud.apply_categories(cats, &new_cmap);

        if self.focused_category.is_some() {
            self.apply_category_focus(frame);
        } else {
            self.upload_points(frame);
        }
    }

    fn upload_points(&self, frame: &eframe::Frame) {
        if let Some(wgpu_rs) = frame.wgpu_render_state() {
            let res_lock = wgpu_rs.renderer.read();
            if let Some(res) = res_lock.callback_resources.get::<PointsCallbackResources>() {
                res.renderer
                    .update_points(&self.wgpu_queue, &self.cloud.points);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-panel UI methods
// ---------------------------------------------------------------------------

impl UmapApp {
    fn show_left_panel(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        // Collapsed tab
        if !self.left_panel_visible {
            egui::SidePanel::left("controls_tab")
                .resizable(false)
                .min_width(28.0)
                .max_width(28.0)
                .show(ctx, |ui| {
                    let rect = ui.available_rect_before_wrap();
                    // Whole strip is clickable to expand.
                    let resp = ui.allocate_rect(rect, egui::Sense::click());
                    if resp.on_hover_text("Show controls").clicked() {
                        self.left_panel_visible = true;
                    }
                    draw_rotated_tab_label(ui, rect, "▲  Controls");
                });
        }

        // ---- left panel: full controls ----
        if self.left_panel_visible {
            egui::SidePanel::left("controls")
                .min_width(200.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("UMAP Viewer");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("◀")
                                .on_hover_text("Hide controls")
                                .clicked()
                            {
                                self.left_panel_visible = false;
                            }
                        });
                    });
                    ui.separator();
                    ui.label(format!("Points: {}", self.cloud.points.len()));

                    // ---- label file selector ----
                    if self.label_files.len() > 1 {
                        ui.add_space(6.0);
                        ui.label("Label set:");
                        let mut changed = false;
                        for (i, (name, _path)) in self.label_files.iter().enumerate() {
                            if ui
                                .selectable_label(self.selected_label_idx == i, name)
                                .clicked()
                                && self.selected_label_idx != i
                            {
                                self.selected_label_idx = i;
                                changed = true;
                            }
                        }
                        if changed {
                            let new_cats =
                                self.all_label_categories[self.selected_label_idx].clone();
                            let empty = crate::data::ColorMap::new();
                            let cmap = self
                                .color_maps
                                .get(self.selected_label_idx)
                                .unwrap_or(&empty);
                            self.cloud.apply_categories(new_cats, cmap);
                            // Re-derive all highlights for the new label set (handles
                            // selection state, focus, and relative unlabeled dim).
                            self.apply_category_focus(frame);
                            if !self.selected_indices.is_empty() {
                                self.category_histogram = self.build_category_histogram();
                            }
                        }
                        ui.separator();
                    }

                    ui.add_space(8.0);
                    ui.label("Point size");
                    ui.add(egui::Slider::new(&mut self.point_size, 1.0..=20.0));
                    ui.label("Opacity");
                    ui.add(egui::Slider::new(&mut self.alpha, 0.01..=1.0));
                    ui.label("Focus size scale");
                    let focus_changed = ui
                        .add(egui::Slider::new(&mut self.focus_size_scale, 1.0..=16.0).step_by(1.0))
                        .changed();
                    if focus_changed && self.focused_category.is_some() {
                        self.apply_category_focus(frame);
                    }
                    ui.add_space(8.0);

                    ui.label(format!("Zoom: {:.2}x", self.zoom));
                    if ui.button("Reset view").clicked() {
                        self.cloud.clear_selection();
                        self.upload_points(frame);
                        self.selected_indices.clear();
                        self.focused_category = None;
                        self.category_histogram.clear();
                        self.sorted_rows.clear();
                        self.poly_verts.clear();
                        self.poly_closed = false;
                        self.mode = Mode::Navigate;
                        self.pan = Vec2::ZERO;
                        self.zoom = 1.0;
                    }
                    if ui
                        .button("Shuffle colors")
                        .on_hover_text("Randomly reassign category colors")
                        .clicked()
                    {
                        self.shuffle_colors(frame);
                        self.pinned_point = None;
                    }
                    ui.add_space(8.0);
                    ui.separator();

                    // ---- mode toggle ----
                    ui.label("Mode:");
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(self.mode == Mode::Navigate, "Navigate")
                            .clicked()
                        {
                            self.mode = Mode::Navigate;
                            self.poly_verts.clear();
                            self.poly_closed = false;
                        }
                        if ui
                            .selectable_label(self.mode == Mode::SelectPolygon, "Select")
                            .clicked()
                        {
                            self.mode = Mode::SelectPolygon;
                        }
                    });
                    ui.add_space(4.0);

                    if self.mode == Mode::SelectPolygon {
                        ui.label(format!("Vertices: {}", self.poly_verts.len()));
                        ui.label("Left-click  → add vertex");
                        ui.label("Near start  → close & select");
                        ui.label("Right-click → cancel");
                        ui.add_space(4.0);
                    }

                    if ui.button("Select All").clicked() {
                        let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
                        let margin_x = (xmax - xmin) * 0.05;
                        let margin_y = (ymax - ymin) * 0.05;
                        let bbox = vec![
                            [xmin - margin_x, ymin - margin_y],
                            [xmax + margin_x, ymin - margin_y],
                            [xmax + margin_x, ymax + margin_y],
                            [xmin - margin_x, ymax + margin_y],
                        ];
                        self.selected_indices = self.cloud.apply_polygon_selection(&bbox);
                        self.focused_category = None;
                        self.pinned_point = None;
                        self.category_histogram = self.build_category_histogram();
                        self.rebuild_sorted_rows();
                        self.upload_points(frame);
                        self.poly_verts = bbox;
                        self.poly_closed = true;
                        self.mode = Mode::Navigate;
                    }
                    ui.add_space(4.0);

                    if !self.selected_indices.is_empty() {
                        ui.separator();
                        ui.label(format!("Selected: {}", self.selected_indices.len()));
                        if ui.button("Clear selection").clicked() {
                            self.cloud.clear_selection();
                            self.upload_points(frame);
                            self.selected_indices.clear();
                            self.focused_category = None;
                            self.pinned_point = None;
                            self.category_histogram.clear();
                            self.sorted_rows.clear();
                            self.poly_verts.clear();
                            self.poly_closed = false;
                        }
                        if ui.button("Export IDs").clicked() {
                            let ids: Vec<String> = self
                                .selected_indices
                                .iter()
                                .filter_map(|&i| self.cloud.labels.get(i).cloned())
                                .collect();
                            let content = ids.join("\n");
                            export_ids(content);
                        }
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Cursor position:");
                    if let Some((x, y)) = self.hover_data_pos {
                        ui.monospace(format!("x = {:.4}", x));
                        ui.monospace(format!("y = {:.4}", y));
                    } else {
                        ui.monospace("x = —");
                        ui.monospace("y = —");
                    }
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Controls:");
                    ui.label("  Scroll → zoom");
                    if self.mode == Mode::Navigate {
                        ui.label("  Drag   → pan");
                    }
                });
        } // if left_panel_visible
    }

    fn show_bottom_panel(&mut self, ctx: &Context) {
        // Collapsed tab
        if !self.selected_indices.is_empty() && !self.table_visible {
            egui::TopBottomPanel::bottom("table_tab")
                .resizable(false)
                .min_height(24.0)
                .max_height(24.0)
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        if ui
                            .button("▲  Show table")
                            .on_hover_text("Show selected points table")
                            .clicked()
                        {
                            self.table_visible = true;
                        }
                    });
                });
        }

        if !self.selected_indices.is_empty() && self.table_visible {
            egui::TopBottomPanel::bottom("selected_points")
                .resizable(true)
                .min_height(10.0)
                .default_height(120.0)
                .max_height(300.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.strong(format!(
                            "Selected points ({}):",
                            self.selected_indices.len()
                        ));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("▼").on_hover_text("Hide table").clicked() {
                                self.table_visible = false;
                            }
                        });
                    });

                    let row_height = ui.text_style_height(&egui::TextStyle::Monospace) + 4.0;
                    let total = self.sorted_rows.len();
                    let sort_col = self.table_sort_col;
                    let sort_asc = self.table_sort_asc;
                    let selected_indices = &self.selected_indices;
                    let sorted_rows = &self.sorted_rows;
                    let cloud = &self.cloud;
                    let pinned_point = self.pinned_point;

                    let mut clicked_col: Option<SortCol> = None;
                    let mut clicked_point: Option<usize> = None;

                    let col_label = |name: &str, col: SortCol| -> String {
                        if sort_col == col {
                            if sort_asc {
                                format!("{name} ▲")
                            } else {
                                format!("{name} ▼")
                            }
                        } else {
                            name.to_string()
                        }
                    };

                    let avail = ui.available_size();
                    ui.allocate_ui(avail, |ui| {
                        egui::ScrollArea::horizontal().show(ui, |ui| {
                            TableBuilder::new(ui)
                                .striped(true)
                                .resizable(true)
                                .sense(egui::Sense::click())
                                .min_scrolled_height(0.0)
                                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                                .column(Column::initial(60.0).range(40.0..=120.0))
                                .column(Column::initial(140.0).range(60.0..=400.0))
                                .column(Column::initial(200.0).range(80.0..=f32::INFINITY))
                                .column(Column::initial(110.0).range(60.0..=200.0))
                                .column(Column::initial(110.0).range(60.0..=200.0))
                                .header(row_height, |mut header| {
                                    for (col, name) in [
                                        (SortCol::Row, "#"),
                                        (SortCol::Category, "Label"),
                                        (SortCol::Label, "ID"),
                                        (SortCol::X, "X"),
                                        (SortCol::Y, "Y"),
                                    ] {
                                        header.col(|ui| {
                                            if ui.button(col_label(name, col)).clicked() {
                                                clicked_col = Some(col);
                                            }
                                        });
                                    }
                                })
                                .body(|body| {
                                    body.rows(row_height, total, |mut row| {
                                        let row_idx = row.index();
                                        let sel_idx = sorted_rows[row_idx];
                                        let idx = selected_indices[sel_idx];
                                        let p = &cloud.points[idx];
                                        row.set_selected(pinned_point == Some(idx));
                                        let category = cloud
                                            .categories
                                            .get(idx)
                                            .map(|s| s.as_str())
                                            .unwrap_or("");
                                        let label =
                                            cloud.labels.get(idx).map(|s| s.as_str()).unwrap_or("");
                                        let category_display = if category.is_empty() {
                                            "(unlabeled)"
                                        } else {
                                            category
                                        };
                                        let label_display =
                                            if label.is_empty() { "(no id)" } else { label };
                                        row.col(|ui| {
                                            ui.monospace(format!("{}", sel_idx + 1));
                                        });
                                        row.col(|ui| {
                                            ui.label(category_display);
                                        });
                                        row.col(|ui| {
                                            ui.label(label_display);
                                        });
                                        row.col(|ui| {
                                            ui.monospace(format!("{:.6}", p.x));
                                        });
                                        row.col(|ui| {
                                            ui.monospace(format!("{:.6}", p.y));
                                        });
                                        if row.response().clicked() {
                                            clicked_point = Some(idx);
                                        }
                                    });
                                });
                        }); // ScrollArea::horizontal
                    }); // allocate_ui

                    if let Some(col) = clicked_col {
                        if self.table_sort_col == col {
                            self.table_sort_asc = !self.table_sort_asc;
                        } else {
                            self.table_sort_col = col;
                            self.table_sort_asc = true;
                        }
                        self.rebuild_sorted_rows();
                    }

                    if let Some(idx) = clicked_point {
                        // Toggle: clicking the same row again clears the pin.
                        if self.pinned_point == Some(idx) {
                            self.pinned_point = None;
                        } else {
                            self.pinned_point = Some(idx);
                            // Pan so the pinned point lands at the canvas center.
                            let p = &self.cloud.points[idx];
                            let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
                            self.pan.x = p.x - (xmin + xmax) * 0.5;
                            self.pan.y = p.y - (ymin + ymax) * 0.5;
                        }
                    }
                });
        }
    }

    /// Shows the right histogram panel. Returns the category name if one was clicked.
    fn show_histogram_panel(&mut self, ctx: &Context) -> Option<String> {
        // Collapsed tab
        if !self.category_histogram.is_empty() && !self.histogram_visible {
            egui::SidePanel::right("histogram_tab")
                .resizable(false)
                .min_width(28.0)
                .max_width(28.0)
                .show(ctx, |ui| {
                    let rect = ui.available_rect_before_wrap();
                    let resp = ui.allocate_rect(rect, egui::Sense::click());
                    if resp.on_hover_text("Show histogram").clicked() {
                        self.histogram_visible = true;
                    }
                    draw_rotated_tab_label(ui, rect, "▼  Selected Labels");
                });
        }

        let mut histogram_click: Option<String> = None;
        if !self.category_histogram.is_empty() && self.histogram_visible {
            // Compute default width from content so labels are fully visible on first show.
            const GAP_W_OUTER: f32 = 6.0;
            const COUNT_W_OUTER: f32 = 52.0;
            const LABEL_PAD_OUTER: f32 = 8.0;
            const BAR_DEFAULT: f32 = 80.0;
            let font_id_outer = egui::FontId::proportional(11.0);
            let max_label_w = self
                .category_histogram
                .iter()
                .map(|(cat, _)| {
                    ctx.fonts(|f| {
                        f.layout_no_wrap(cat.clone(), font_id_outer.clone(), egui::Color32::WHITE)
                            .size()
                            .x
                    })
                })
                .fold(0.0_f32, f32::max)
                + LABEL_PAD_OUTER;
            let computed_default = max_label_w + GAP_W_OUTER + COUNT_W_OUTER + BAR_DEFAULT;
            egui::SidePanel::right("histogram")
                .default_width(computed_default)
                .min_width(60.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new("Category histogram").heading())
                                .truncate(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("▶")
                                .on_hover_text("Hide histogram")
                                .clicked()
                            {
                                self.histogram_visible = false;
                            }
                        });
                    });
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new("Click a label to highlight its points")
                                .size(10.0)
                                .italics()
                                .color(ui.visuals().weak_text_color()),
                        )
                        .truncate(),
                    );
                    ui.separator();

                    let max_count = self
                        .category_histogram
                        .first()
                        .map(|(_, c)| *c)
                        .unwrap_or(1);
                    const GAP_W: f32 = 6.0;
                    const COUNT_W: f32 = 52.0;
                    const LABEL_PADDING: f32 = 8.0;
                    let font_id = egui::FontId::proportional(11.0);
                    let avail = ui.available_width();
                    let raw_label_w = self
                        .category_histogram
                        .iter()
                        .map(|(cat, _)| {
                            ui.fonts(|f| {
                                f.layout_no_wrap(cat.clone(), font_id.clone(), egui::Color32::WHITE)
                                    .size()
                                    .x
                            })
                        })
                        .fold(0.0_f32, f32::max)
                        + LABEL_PADDING;
                    // Cap label_w so that label_w + GAP_W + bar + COUNT_W <= avail for any bar >= 0.
                    let label_w = raw_label_w.min((avail - GAP_W - COUNT_W - 1.0).max(0.0));
                    let bar_max_w = (avail - label_w - GAP_W - COUNT_W).max(0.0);
                    let focused = &self.focused_category;

                    let mut clicked: Option<String> = None;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (cat, count) in &self.category_histogram {
                            let fraction = *count as f32 / max_count as f32;
                            let is_focused = focused.as_deref() == Some(cat.as_str());
                            // Look up the scatter plot color for this category from any matching point.
                            // "(unlabeled)" in the histogram corresponds to empty-string categories.
                            let bar_color = self
                                .cloud
                                .categories
                                .iter()
                                .zip(self.cloud.points.iter())
                                .find(|(c, _)| {
                                    if cat == "(unlabeled)" {
                                        c.is_empty()
                                    } else {
                                        c.as_str() == cat.as_str()
                                    }
                                })
                                .map(|(_, p)| {
                                    egui::Color32::from_rgb(
                                        (p.r * 255.0) as u8,
                                        (p.g * 255.0) as u8,
                                        (p.b * 255.0) as u8,
                                    )
                                })
                                .unwrap_or(egui::Color32::from_rgb(80, 140, 220));
                            ui.horizontal(|ui| {
                                // Zero item spacing so all widths are fully predictable.
                                ui.spacing_mut().item_spacing.x = 0.0;
                                // Clickable category label (right-justified).
                                let (label_rect, label_resp) = ui.allocate_exact_size(
                                    egui::vec2(label_w, 16.0),
                                    egui::Sense::click(),
                                );
                                if label_resp.clicked() {
                                    clicked = Some(cat.clone());
                                }
                                // Highlight background when focused or hovered.
                                if is_focused {
                                    ui.painter().rect_filled(
                                        label_rect,
                                        2.0,
                                        egui::Color32::from_rgba_premultiplied(255, 220, 50, 40),
                                    );
                                } else if label_resp.hovered() {
                                    ui.painter().rect_filled(
                                        label_rect,
                                        2.0,
                                        egui::Color32::from_white_alpha(15),
                                    );
                                }
                                let text_color = if is_focused {
                                    egui::Color32::BLACK
                                } else {
                                    ui.visuals().text_color()
                                };
                                ui.painter().text(
                                    label_rect.right_center(),
                                    egui::Align2::RIGHT_CENTER,
                                    cat,
                                    egui::FontId::proportional(11.0),
                                    text_color,
                                );
                                // Fixed gap.
                                ui.allocate_exact_size(
                                    egui::vec2(GAP_W, 16.0),
                                    egui::Sense::hover(),
                                );
                                // Bar.
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(bar_max_w, 14.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter()
                                    .rect_filled(rect, 2.0, egui::Color32::from_gray(40));
                                let filled = egui::Rect::from_min_size(
                                    rect.min,
                                    egui::vec2(rect.width() * fraction, rect.height()),
                                );
                                ui.painter().rect_filled(filled, 2.0, bar_color);
                                // Fixed-width count label.
                                let (count_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(COUNT_W, 16.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().text(
                                    count_rect.left_center(),
                                    egui::Align2::LEFT_CENTER,
                                    format!(" {}", count),
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().text_color(),
                                );
                            });
                        }
                    });
                    histogram_click = clicked;
                });
        }

        histogram_click
    }

    fn show_canvas(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::canvas(&ctx.style()))
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                let hover_screen = ui
                    .input(|i| i.pointer.hover_pos())
                    .filter(|p| rect.contains(*p));

                // ---- scroll zoom (both modes) ----
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 && response.hovered() {
                    let old_zoom = self.zoom;
                    let new_zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.05, 500.0);
                    let actual_f = new_zoom / old_zoom;
                    // Zoom toward cursor: adjust pan so the data point under the
                    // cursor stays fixed on screen.
                    if let Some(cursor) = hover_screen {
                        let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
                        let aspect = rect.width() / rect.height();
                        let span_x = (xmax - xmin) * 0.5 / old_zoom;
                        let span_y = (ymax - ymin) * 0.5 / old_zoom;
                        let (half_x, half_y) = if aspect >= 1.0 {
                            (span_x * aspect, span_y)
                        } else {
                            (span_x, span_y / aspect)
                        };
                        let nx = (cursor.x - rect.left()) / rect.width() * 2.0 - 1.0;
                        let ny = -((cursor.y - rect.top()) / rect.height() * 2.0 - 1.0);
                        self.pan.x += nx * half_x * (1.0 - 1.0 / actual_f);
                        self.pan.y += ny * half_y * (1.0 - 1.0 / actual_f);
                    }
                    self.zoom = new_zoom;
                }

                // ---- pinch-to-zoom + two-finger pan (WASM / touch devices) ----
                #[cfg(target_arch = "wasm32")]
                let multitouch_active = {
                    let mut active = false;
                    if let Some(touch) = ui.input(|i| i.multi_touch()) {
                        active = true;
                        let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
                        let aspect = rect.width() / rect.height();

                        // --- pinch zoom toward the gesture centroid ---
                        if touch.zoom_delta != 1.0 {
                            let old_zoom = self.zoom;
                            let new_zoom = (self.zoom * touch.zoom_delta).clamp(0.05, 500.0);
                            let actual_f = new_zoom / old_zoom;
                            // Use the current pointer position as the pinch centroid.
                            let centroid = ui
                                .input(|i| i.pointer.hover_pos())
                                .unwrap_or(rect.center())
                                .clamp(rect.min, rect.max);
                            let span_x = (xmax - xmin) * 0.5 / old_zoom;
                            let span_y = (ymax - ymin) * 0.5 / old_zoom;
                            let (half_x, half_y) = if aspect >= 1.0 {
                                (span_x * aspect, span_y)
                            } else {
                                (span_x, span_y / aspect)
                            };
                            let nx = (centroid.x - rect.left()) / rect.width() * 2.0 - 1.0;
                            let ny = -((centroid.y - rect.top()) / rect.height() * 2.0 - 1.0);
                            self.pan.x += nx * half_x * (1.0 - 1.0 / actual_f);
                            self.pan.y += ny * half_y * (1.0 - 1.0 / actual_f);
                            self.zoom = new_zoom;
                        }

                        // --- two-finger pan (centroid translation) ---
                        let td = touch.translation_delta;
                        if td.x != 0.0 || td.y != 0.0 {
                            let span_x = (xmax - xmin) / self.zoom;
                            let span_y = (ymax - ymin) / self.zoom;
                            self.pan.x -= td.x / rect.width() * span_x;
                            self.pan.y += td.y / rect.height() * span_y;
                        }
                    }
                    active
                };
                #[cfg(not(target_arch = "wasm32"))]
                let multitouch_active = false;

                match self.mode {
                    Mode::Navigate => {
                        // pan via drag (single finger / mouse); suppressed during multi-touch
                        if response.drag_started() && !multitouch_active {
                            self.drag_start =
                                response.interact_pointer_pos().map(|p| (p, self.pan));
                        }
                        if response.dragged() && !multitouch_active {
                            if let (Some((start_pos, start_pan)), Some(cur)) =
                                (self.drag_start, response.interact_pointer_pos())
                            {
                                let delta = cur - start_pos;
                                let [xmin, xmax, ymin, ymax] = self.cloud.bounds;
                                let span_x = (xmax - xmin) / self.zoom;
                                let span_y = (ymax - ymin) / self.zoom;
                                self.pan = start_pan
                                    - Vec2::new(
                                        delta.x / rect.width() * span_x,
                                        -delta.y / rect.height() * span_y,
                                    );
                            }
                        }
                        if response.drag_stopped() {
                            self.drag_start = None;
                        }

                        // hover hit-test
                        if let Some(screen) = hover_screen {
                            let dp = self.screen_to_data(screen, rect);
                            let radius = self.hit_radius_data(rect);
                            self.hovered_point = self.cloud.grid.query_nearest(
                                &self.cloud.points,
                                dp.0,
                                dp.1,
                                radius,
                            );
                            self.hover_data_pos = self.hovered_point.map(|i| {
                                let p = &self.cloud.points[i];
                                (p.x, p.y)
                            });
                        } else {
                            self.hovered_point = None;
                            self.hover_data_pos = None;
                        }
                    }

                    Mode::SelectPolygon => {
                        self.drag_start = None;
                        self.hovered_point = None;
                        self.hover_data_pos = None;

                        // Right-click → cancel polygon
                        if response.secondary_clicked() {
                            self.poly_verts.clear();
                            self.poly_closed = false;
                        }

                        // Left-click → add vertex or close
                        if response.clicked() {
                            if let Some(screen) = response.interact_pointer_pos() {
                                if !rect.contains(screen) {
                                    return;
                                }

                                // Check if close to first vertex (close the polygon)
                                let close = self.poly_verts.first().map(|&v| {
                                    let first_screen = self.data_to_screen(v[0], v[1], rect);
                                    (screen - first_screen).length() < 12.0
                                        && self.poly_verts.len() >= 3
                                }).unwrap_or(false);

                                if close {
                                    let poly = self.poly_verts.clone();
                                    self.selected_indices =
                                        self.cloud.apply_polygon_selection(&poly);
                                    self.focused_category = None;
                                    self.pinned_point = None;
                                    self.category_histogram = self.build_category_histogram();
                                    self.rebuild_sorted_rows();
                                    self.upload_points(frame);
                                    self.poly_closed = true;
                                    self.mode = Mode::Navigate;
                                } else {
                                    if self.poly_closed {
                                        // Start a new polygon: keep old selection visible, just reset polygon
                                        self.poly_verts.clear();
                                        self.poly_closed = false;
                                    }
                                    let dp = self.screen_to_data(screen, rect);
                                    self.poly_verts.push([dp.0, dp.1]);
                                }
                            }
                        }
                    }
                }

                // ---- wgpu draw ----
                let transform = build_transform(
                    [self.pan.x, self.pan.y],
                    self.zoom,
                    rect.width(),
                    rect.height(),
                    self.cloud.bounds,
                );
                let clip_point_size = self.point_size / (rect.height() * 0.5);

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    PointsCallback {
                        transform,
                        point_size: clip_point_size,
                        viewport_aspect: rect.width() / rect.height(),
                        alpha: self.alpha,
                    },
                ));

                // ---- polygon overlay ----
                if !self.poly_verts.is_empty() {
                    let painter = ui.painter();
                    let screen_verts: Vec<Pos2> = self
                        .poly_verts
                        .iter()
                        .map(|&[x, y]| self.data_to_screen(x, y, rect))
                        .collect();

                    // Edges
                    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 220, 50));
                    for w in screen_verts.windows(2) {
                        painter.line_segment([w[0], w[1]], stroke);
                    }
                    if self.poly_closed {
                        // Closing edge: last vertex → first vertex
                        if screen_verts.len() >= 2 {
                            painter.line_segment(
                                [*screen_verts.last().unwrap(), screen_verts[0]],
                                stroke,
                            );
                        }
                    } else {
                        // Preview edge: last vertex → cursor
                        if let Some(cursor) = hover_screen {
                            painter.line_segment(
                                [*screen_verts.last().unwrap(), cursor],
                                egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgba_premultiplied(255, 220, 50, 120),
                                ),
                            );
                        }
                    }
                    // Vertices
                    for (i, &sv) in screen_verts.iter().enumerate() {
                        let (radius, color) = if !self.poly_closed && i == 0 {
                            (6.0, egui::Color32::from_rgb(255, 80, 80)) // first = red while open
                        } else {
                            (4.0, egui::Color32::from_rgb(255, 220, 50))
                        };
                        painter.circle_filled(sv, radius, color);
                    }
                }

                // ---- pinned point ring ----
                if let Some(pin_idx) = self.pinned_point {
                    if let Some(p) = self.cloud.points.get(pin_idx) {
                        let screen_pos = self.data_to_screen(p.x, p.y, rect);
                        let painter = ui.painter();
                        let radius = self.point_size * 3.0;
                        // Outer bright ring
                        painter.circle_stroke(
                            screen_pos,
                            radius,
                            egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 240, 60)),
                        );
                        // Small crosshair lines
                        let arm = radius * 0.6;
                        let c = egui::Color32::from_rgb(255, 240, 60);
                        let s = egui::Stroke::new(1.5, c);
                        painter.line_segment(
                            [screen_pos - egui::vec2(arm + radius, 0.0),
                             screen_pos - egui::vec2(radius, 0.0)], s);
                        painter.line_segment(
                            [screen_pos + egui::vec2(radius, 0.0),
                             screen_pos + egui::vec2(arm + radius, 0.0)], s);
                        painter.line_segment(
                            [screen_pos - egui::vec2(0.0, arm + radius),
                             screen_pos - egui::vec2(0.0, radius)], s);
                        painter.line_segment(
                            [screen_pos + egui::vec2(0.0, radius),
                             screen_pos + egui::vec2(0.0, arm + radius)], s);
                    }
                }

                // ---- tooltip ----
                if let Some((x, y)) = self.hover_data_pos {
                    if let Some(cursor) = hover_screen {
                        let tooltip_pos = cursor + egui::vec2(12.0, -24.0);
                        let painter = ui.painter();
                        let category = self
                            .hovered_point
                            .and_then(|i| self.cloud.categories.get(i))
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        let label = self
                            .hovered_point
                            .and_then(|i| self.cloud.labels.get(i))
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        let text = match (category.is_empty(), label.is_empty()) {
                            (true, true) => format!("({:.4}, {:.4})", x, y),
                            (false, true) => format!("label: {}\n({:.4}, {:.4})", category, x, y),
                            (true, false) => format!("id: {}\n({:.4}, {:.4})", label, x, y),
                            (false, false) => {
                                format!("label: {}\nid: {}\n({:.4}, {:.4})", category, label, x, y)
                            }
                        };
                        let galley = painter.layout(
                            text,
                            egui::FontId::monospace(11.0),
                            egui::Color32::WHITE,
                            400.0,
                        );
                        let bg = egui::Rect::from_min_size(
                            tooltip_pos - egui::vec2(3.0, 3.0),
                            galley.size() + egui::vec2(6.0, 6.0),
                        );
                        painter.rect_filled(bg, 3.0, egui::Color32::from_black_alpha(200));
                        painter.galley(tooltip_pos, galley, egui::Color32::WHITE);
                    }
                }
            });
    }
}

// ---------------------------------------------------------------------------
// eframe::App — thin orchestrator calling the per-panel methods above
// ---------------------------------------------------------------------------

impl eframe::App for UmapApp {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        self.show_left_panel(ctx, frame);
        self.show_bottom_panel(ctx);
        if let Some(cat) = self.show_histogram_panel(ctx) {
            if self.focused_category.as_deref() == Some(cat.as_str()) {
                self.focused_category = None;
            } else {
                self.focused_category = Some(cat);
            }
            self.apply_category_focus(frame);
        }
        self.show_canvas(ctx, frame);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Paint `text` rotated 90° clockwise and centered within `rect`.
/// Used for collapsed side-panel tabs.
fn draw_rotated_tab_label(ui: &mut egui::Ui, rect: egui::Rect, text: &str) {
    let font_id = egui::FontId::proportional(12.0);
    let color = ui.visuals().text_color();
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font_id, color));
    let w = galley.size().x;
    let h = galley.size().y;
    // After π/2 CW rotation around `pos`, the visual center is at pos + (-h/2, w/2).
    // Solve for pos so that center == rect.center().
    let c = rect.center();
    let pos = egui::pos2(c.x + h * 0.5, c.y - w * 0.5);
    ui.painter().add(egui::epaint::TextShape {
        pos,
        galley,
        underline: egui::Stroke::NONE,
        fallback_color: color,
        override_text_color: None,
        opacity_factor: 1.0,
        angle: std::f32::consts::FRAC_PI_2,
    });
}

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn export_ids(content: String) {
    let mut dialog = rfd::FileDialog::new()
        .set_file_name("selected_ids.txt")
        .add_filter("Text file", &["txt"]);
    if let Some(downloads) = dirs::home_dir().map(|h| h.join("Downloads")) {
        if downloads.exists() {
            dialog = dialog.set_directory(&downloads);
        }
    }
    if let Some(path) = dialog.save_file() {
        let _ = std::fs::write(&path, content);
    }
}

#[cfg(target_arch = "wasm32")]
fn export_ids(content: String) {
    use wasm_bindgen::JsCast as _;
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();

    let array = js_sys::Array::new();
    array.push(&wasm_bindgen::JsValue::from_str(&content));
    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type("text/plain");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&array, &opts).unwrap();
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

    let a = document
        .create_element("a")
        .unwrap()
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .unwrap();
    a.set_href(&url);
    a.set_download("selected_ids.txt");
    a.click();
    let _ = web_sys::Url::revoke_object_url(&url);
}
