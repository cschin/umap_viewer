use rand::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;

/// Default colour for points with no category label (mid-gray).
pub const UNLABELED_COLOR: (f32, f32, f32) = (0.5, 0.5, 0.5);

// ---------------------------------------------------------------------------
// Spatial grid for O(1) hover hit-testing
// ---------------------------------------------------------------------------

pub struct SpatialGrid {
    cells: Vec<Vec<u32>>,
    xmin: f32,
    ymin: f32,
    cell_w: f32,
    cell_h: f32,
    cols: usize,
    rows: usize,
}

impl SpatialGrid {
    pub fn build(points: &[Point], bounds: [f32; 4]) -> Self {
        let [xmin, xmax, ymin, ymax] = bounds;
        let cols = 512usize;
        let rows = 512usize;
        let cell_w = (xmax - xmin) / cols as f32;
        let cell_h = (ymax - ymin) / rows as f32;
        let mut cells = vec![Vec::new(); cols * rows];
        for (i, p) in points.iter().enumerate() {
            let c = ((p.x - xmin) / cell_w) as usize;
            let r = ((p.y - ymin) / cell_h) as usize;
            let c = c.min(cols - 1);
            let r = r.min(rows - 1);
            cells[r * cols + c].push(i as u32);
        }
        Self {
            cells,
            xmin,
            ymin,
            cell_w,
            cell_h,
            cols,
            rows,
        }
    }

    /// Return the index of the closest point within `radius` (data-space), or `None`.
    pub fn query_nearest(&self, points: &[Point], x: f32, y: f32, radius: f32) -> Option<usize> {
        let r2 = radius * radius;
        let c0 = ((x - self.xmin) / self.cell_w) as isize;
        let r0 = ((y - self.ymin) / self.cell_h) as isize;
        let dc = (radius / self.cell_w).ceil() as isize + 1;
        let dr = (radius / self.cell_h).ceil() as isize + 1;

        let mut best: Option<(usize, f32)> = None;
        for row in (r0 - dr)..=(r0 + dr) {
            for col in (c0 - dc)..=(c0 + dc) {
                if col < 0 || row < 0 || col >= self.cols as isize || row >= self.rows as isize {
                    continue;
                }
                for &pi in &self.cells[row as usize * self.cols + col as usize] {
                    let p = &points[pi as usize];
                    let d2 = (p.x - x).powi(2) + (p.y - y).powi(2);
                    if d2 <= r2 && best.as_ref().map_or_else(|| true, |&(_, bd)| d2 < bd) {
                        best = Some((pi as usize, d2));
                    }
                }
            }
        }
        best.map(|(i, _)| i)
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    /// 1.0 = normal/selected, 0.15 = dimmed (not selected)
    pub highlight: f32,
    /// Size multiplier relative to global point_size (1.0 = normal, 2.0 = double)
    pub size: f32,
}

/// Maps a category name to an RGB colour (0.0–1.0 per channel).
pub type ColorMap = std::collections::HashMap<String, (f32, f32, f32)>;

pub struct PointCloud {
    pub points: Vec<Point>,
    pub bounds: [f32; 4], // xmin, xmax, ymin, ymax
    pub grid: SpatialGrid,
    /// Per-point ID string. May be empty if not loaded.
    pub labels: Vec<String>,
    /// Per-point category string used for colouring and tooltips (active label set).
    pub categories: Vec<String>,
    /// Per-point info string loaded from the optional `info` column of the primary
    /// labels parquet.  Plain text or a Markdown-style link `[text](url)`.
    /// Empty when the column is absent.
    pub info: Vec<String>,
    /// Display names for all available label sets.
    pub label_set_names: Vec<String>,
    /// Per-point categories for every label set (index matches label_set_names).
    pub all_categories: Vec<Vec<String>>,
    /// Per-label-set colour maps: category name → (r, g, b).  Empty map = use hue defaults.
    pub category_color_maps: Vec<ColorMap>,
}

impl PointCloud {
    /// Generate 500k synthetic UMAP-like points: Gaussian clusters at random centers.
    pub fn generate_test(n_points: usize, n_clusters: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut points = Vec::with_capacity(n_points);

        // Cluster centers spread across [-12, 12]^2
        let centers: Vec<(f32, f32)> = (0..n_clusters)
            .map(|_| {
                (
                    rng.gen_range(-12.0_f32..12.0_f32),
                    rng.gen_range(-12.0_f32..12.0_f32),
                )
            })
            .collect();

        // Each cluster gets a random spread so sizes vary like real UMAP
        let spreads: Vec<f32> = (0..n_clusters)
            .map(|_| rng.gen_range(0.3_f32..1.4_f32))
            .collect();

        // Evenly spaced hues
        let colors: Vec<(f32, f32, f32)> = (0..n_clusters)
            .map(|i| hue_to_rgb(i as f32 / n_clusters as f32))
            .collect();

        for i in 0..n_points {
            let c = i % n_clusters;
            let (cx, cy) = centers[c];
            let sigma = spreads[c];
            let (r, g, b) = colors[c];
            // Box-Muller Gaussian sample
            let (dx, dy) = box_muller(&mut rng, sigma);
            points.push(Point {
                x: cx + dx,
                y: cy + dy,
                r,
                g,
                b,
                highlight: 1.0,
                size: 1.0,
            });
        }

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

        let bounds = [xmin, xmax, ymin, ymax];
        let grid = SpatialGrid::build(&points, bounds);
        Self {
            points,
            bounds,
            grid,
            labels: Vec::new(),
            categories: Vec::new(),
            info: Vec::new(),
            label_set_names: Vec::new(),
            all_categories: Vec::new(),
            category_color_maps: Vec::new(),
        }
    }

    /// Apply a polygon selection: selected points get highlight=1.0, others 0.15.
    /// Unlabeled points are dimmed to 0.15 even when selected.
    /// Returns the indices of selected points.
    pub fn apply_polygon_selection(&mut self, poly: &[[f32; 2]]) -> Vec<usize> {
        let mut indices = Vec::new();
        for (i, p) in self.points.iter_mut().enumerate() {
            let sel = point_in_polygon(p.x, p.y, poly);
            p.highlight = if sel { 1.0 } else { 0.15 };
            p.size = 1.0;
            if sel {
                indices.push(i);
            }
        }
        self.dim_unlabeled();
        indices
    }

    /// Parse a pre-built binary blob (works on all targets including WASM).
    ///
    /// Format (multiple label sets):
    ///   magic        : b"UMAP" (4 bytes)
    ///   n_points     : u32 le
    ///   n_label_sets : u32 le
    ///   id_stride    : u32 le
    ///   for each label set:
    ///     set_name : u32 le len + utf-8 bytes
    ///     n_cats   : u32 le
    ///     for each category: u32 le len + utf-8 bytes
    ///     cat_idx  : [u32 le; n_points]
    ///   x       : [f32 le; n_points]
    ///   y       : [f32 le; n_points]
    ///   id_blob : [u8; n_points * id_stride]
    pub fn from_bin(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut p = 0usize;
        macro_rules! read_u32 {
            () => {{
                let v = u32::from_le_bytes(data[p..p + 4].try_into()?);
                p += 4;
                v
            }};
        }
        macro_rules! read_f32 {
            () => {{
                let v = f32::from_le_bytes(data[p..p + 4].try_into()?);
                p += 4;
                v
            }};
        }
        macro_rules! read_str {
            () => {{
                let n = read_u32!() as usize;
                let s = std::str::from_utf8(&data[p..p + n])?.to_string();
                p += n;
                s
            }};
        }

        if &data[p..p + 4] != b"UMAP" {
            return Err("bad magic".into());
        }
        p += 4;

        let n_points = read_u32!() as usize;
        let n_label_sets = read_u32!() as usize;
        let id_stride = read_u32!() as usize;

        let mut label_set_names: Vec<String> = Vec::with_capacity(n_label_sets);
        let mut all_cat_names: Vec<Vec<String>> = Vec::with_capacity(n_label_sets);
        let mut all_cat_indices: Vec<Vec<usize>> = Vec::with_capacity(n_label_sets);

        let mut category_color_maps: Vec<ColorMap> = Vec::with_capacity(n_label_sets);
        for _ in 0..n_label_sets {
            label_set_names.push(read_str!());
            let n_cats = read_u32!() as usize;
            let mut cat_names: Vec<String> = Vec::with_capacity(n_cats);
            let mut color_map = ColorMap::new();
            for _ in 0..n_cats {
                let name = read_str!();
                let r = data[p] as f32 / 255.0;
                p += 1;
                let g = data[p] as f32 / 255.0;
                p += 1;
                let b = data[p] as f32 / 255.0;
                p += 1;
                color_map.insert(name.clone(), (r, g, b));
                cat_names.push(name);
            }
            let mut indices: Vec<usize> = Vec::with_capacity(n_points);
            for _ in 0..n_points {
                indices.push(read_u32!() as usize);
            }
            all_cat_names.push(cat_names);
            all_cat_indices.push(indices);
            category_color_maps.push(color_map);
        }

        let mut xs = Vec::with_capacity(n_points);
        let mut ys = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            xs.push(read_f32!());
        }
        for _ in 0..n_points {
            ys.push(read_f32!());
        }

        let mut labels = Vec::with_capacity(n_points);
        if id_stride > 0 {
            let blob = &data[p..p + n_points * id_stride];
            for i in 0..n_points {
                let raw = &blob[i * id_stride..(i + 1) * id_stride];
                let trimmed = raw
                    .iter()
                    .rposition(|&b| b != 0)
                    .map(|q| &raw[..=q])
                    .unwrap_or(&raw[..0]);
                labels.push(std::str::from_utf8(trimmed)?.to_string());
            }
        }

        // Colour points by the first label set using embedded colours.
        let first_names = &all_cat_names[0];
        let first_idx = &all_cat_indices[0];
        let first_colors = &category_color_maps[0];
        let n_cats_f = first_names.len().max(1) as f32;
        let mut points = Vec::with_capacity(n_points);
        let mut categories = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let name = &first_names[first_idx[i]];
            let (r, g, b) = if name.is_empty() {
                UNLABELED_COLOR
            } else {
                first_colors
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| hue_to_rgb(first_idx[i] as f32 / n_cats_f))
            };
            points.push(Point {
                x: xs[i],
                y: ys[i],
                r,
                g,
                b,
                highlight: 1.0,
                size: 1.0,
            });
            categories.push(name.clone());
        }

        let all_categories: Vec<Vec<String>> = all_cat_names
            .iter()
            .zip(all_cat_indices.iter())
            .map(|(names, indices)| indices.iter().map(|&i| names[i].clone()).collect())
            .collect();

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let bounds = [xmin, xmax, ymin, ymax];
        let grid = SpatialGrid::build(&points, bounds);
        let mut cloud = Self {
            points,
            bounds,
            grid,
            labels,
            categories,
            info: Vec::new(), // binary format does not carry info
            label_set_names,
            all_categories,
            category_color_maps,
        };
        cloud.dim_unlabeled();
        Ok(cloud)
    }

    /// Export to the UMAP binary format consumed by `from_bin`.
    /// `label_pairs` is `(display_name, labels_parquet_path)` per set.
    /// `color_files` is a parallel slice of optional CSV paths (`None` = use hue defaults).
    /// Run the native binary with `--export-bin` to generate `data/points.bin`
    /// before doing a WASM build.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_to_bin(
        coords_path: &str,
        label_pairs: &[(String, String)],
        color_files: &[Option<String>],
        out_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let primary_path = label_pairs.first().map(|(_, p)| p.as_str()).unwrap_or("");
        let cloud = Self::load_from_parquet(coords_path, primary_path)?;
        let n = cloud.points.len();
        let id_stride = cloud.labels.iter().map(|s| s.len()).max().unwrap_or(0);

        // Deduplicate categories, preserving first-seen order.
        let encode_cats = |cats: &[String]| -> (Vec<String>, Vec<u32>) {
            let mut map: HashMap<&str, u32> = HashMap::new();
            let mut names: Vec<String> = Vec::new();
            let mut indices: Vec<u32> = Vec::with_capacity(cats.len());
            for cat in cats {
                let next = map.len() as u32;
                let idx = *map.entry(cat.as_str()).or_insert_with(|| {
                    names.push(cat.clone());
                    next
                });
                indices.push(idx);
            }
            (names, indices)
        };

        // Collect all label sets with their resolved colours.
        let mut sets: Vec<(String, Vec<String>, Vec<u32>, ColorMap)> = Vec::new();
        for (i, (set_name, labels_path)) in label_pairs.iter().enumerate() {
            let cats = if labels_path == primary_path {
                cloud.categories.clone()
            } else {
                cloud
                    .load_categories_from_parquet(labels_path)
                    .unwrap_or_else(|e| {
                        eprintln!("Warning: could not load {labels_path}: {e}");
                        vec![String::new(); n]
                    })
            };
            let (names, indices) = encode_cats(&cats);

            // Load custom colours if a CSV was provided, otherwise empty map (hue fallback).
            let color_map = color_files
                .get(i)
                .and_then(|opt| opt.as_deref())
                .map(|path| {
                    Self::load_color_csv(path).unwrap_or_else(|e| {
                        eprintln!("Warning: could not load color CSV {path}: {e}");
                        ColorMap::new()
                    })
                })
                .unwrap_or_default();

            sets.push((set_name.clone(), names, indices, color_map));
        }

        let write_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"UMAP");
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(sets.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(id_stride as u32).to_le_bytes());

        for (set_idx, (set_name, cat_names, cat_indices, color_map)) in sets.iter().enumerate() {
            let n_cats = cat_names.len();
            let n_cats_f = n_cats.max(1) as f32;
            // Precompute hue colours so each cat has a deterministic default.
            let hue_colors: Vec<(f32, f32, f32)> = (0..n_cats)
                .map(|i| hue_to_rgb(i as f32 / n_cats_f))
                .collect();

            write_str(&mut buf, set_name);
            buf.extend_from_slice(&(n_cats as u32).to_le_bytes());
            for (ci, name) in cat_names.iter().enumerate() {
                write_str(&mut buf, name);
                let (r, g, b) = if name.is_empty() {
                    UNLABELED_COLOR
                } else {
                    color_map.get(name).copied().unwrap_or(hue_colors[ci])
                };
                buf.push((r * 255.0).round() as u8);
                buf.push((g * 255.0).round() as u8);
                buf.push((b * 255.0).round() as u8);
            }
            for idx in cat_indices {
                buf.extend_from_slice(&idx.to_le_bytes());
            }
            let _ = set_idx; // suppress unused warning
        }

        for pt in &cloud.points {
            buf.extend_from_slice(&pt.x.to_le_bytes());
        }
        for pt in &cloud.points {
            buf.extend_from_slice(&pt.y.to_le_bytes());
        }

        if id_stride > 0 {
            for id in &cloud.labels {
                let bytes = id.as_bytes();
                buf.extend_from_slice(bytes);
                buf.resize(buf.len() + (id_stride - bytes.len()), 0u8);
            }
        }

        std::fs::write(out_path, &buf)?;
        println!(
            "Exported {} points, {} label set(s), id_stride={} → {}",
            n,
            sets.len(),
            id_stride,
            out_path
        );
        Ok(())
    }

    /// Dim points with an empty category to 75 % of their current highlight.
    /// Must be called once after highlights have been freshly set (not repeatedly).
    pub fn dim_unlabeled(&mut self) {
        for (p, cat) in self.points.iter_mut().zip(self.categories.iter()) {
            if cat.is_empty() {
                p.highlight *= 0.5;
            }
        }
    }

    /// Reset all highlights to 1.0 (clears any selection), then dim unlabeled points.
    pub fn clear_selection(&mut self) {
        for p in &mut self.points {
            p.highlight = 1.0;
            p.size = 1.0;
        }
        self.dim_unlabeled();
    }

    /// Load categories from a labels parquet file (joined by id) and return
    /// them in the same order as `self.labels` (the point ids).
    /// Parquet schema: `id` (Utf8), `labels` (Utf8).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_categories_from_parquet(
        &self,
        labels_path: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use polars::prelude::*;

        let lf = LazyFrame::scan_parquet(labels_path.into(), Default::default())?
            .select([col("id"), col("labels")])
            .collect()?;

        let ids_col = lf.column("id")?.str()?;
        let cats_col = lf.column("labels")?.str()?;

        let mut label_map: HashMap<String, String> = HashMap::new();
        for i in 0..lf.height() {
            if let Some(id) = ids_col.get(i) {
                let cat = cats_col.get(i).unwrap_or("").to_string();
                label_map.insert(id.to_string(), cat);
            }
        }

        Ok(self
            .labels
            .iter()
            .map(|id| label_map.get(id).cloned().unwrap_or_default())
            .collect())
    }

    /// Replace `self.categories` with `new_categories` and recompute point colours.
    /// `color_map` maps category name → (r,g,b); pass an empty map to use hue defaults.
    /// `new_categories` must have the same length as `self.points`.
    pub fn apply_categories(&mut self, new_categories: Vec<String>, color_map: &ColorMap) {
        let mut category_order: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut category_idx: Vec<usize> = Vec::with_capacity(new_categories.len());
        for cat in &new_categories {
            let next = category_order.len();
            let idx = *category_order.entry(cat.clone()).or_insert(next);
            category_idx.push(idx);
        }
        let n_categories = category_order.len().max(1);
        for (i, p) in self.points.iter_mut().enumerate() {
            let cat = &new_categories[i];
            let (r, g, b) = if cat.is_empty() {
                UNLABELED_COLOR
            } else {
                color_map
                    .get(cat)
                    .copied()
                    .unwrap_or_else(|| hue_to_rgb(category_idx[i] as f32 / n_categories as f32))
            };
            p.r = r;
            p.g = g;
            p.b = b;
        }
        self.categories = new_categories;
        // Note: highlights are not reset here; the caller is responsible for
        // re-applying highlight state (e.g. via apply_category_focus or clear_selection).
    }

    /// Load a colour CSV file: two columns `label,color` where color is `#RRGGBB`.
    /// Lines starting with `#` are treated as comments and skipped.
    /// A header row `label,color` (case-insensitive) is also skipped automatically.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_color_csv(path: &str) -> Result<ColorMap, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let mut map = ColorMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.splitn(2, ',');
            let label = match parts.next() {
                Some(l) => l.trim(),
                None => continue,
            };
            let color = match parts.next() {
                Some(c) => c.trim(),
                None => continue,
            };
            // Skip header row
            if label.eq_ignore_ascii_case("label") {
                continue;
            }
            let hex = color.trim_start_matches('#');
            if hex.len() != 6 {
                continue;
            }
            let r = u8::from_str_radix(&hex[0..2], 16)?;
            let g = u8::from_str_radix(&hex[2..4], 16)?;
            let b = u8::from_str_radix(&hex[4..6], 16)?;
            map.insert(
                label.to_string(),
                (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0),
            );
        }
        Ok(map)
    }

    /// Load UMAP coordinates and labels from parquet files.
    /// `coords_path`: schema `id` (Utf8), `coordinates` (FixedSizeList<f32, 2>).
    /// `labels_path`: schema `id` (Utf8), `labels` (Utf8).
    /// The two files are joined on `id`; points are coloured by their label.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_parquet(
        coords_path: &str,
        labels_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use polars::prelude::*;

        // Accept two coord schemas:
        //   - Legacy: `coordinates` FixedSizeList<f32,2>  (original parquet files)
        //   - Flat:   separate `x` and `y` float columns  (csv_to_input_package output)
        let coords_has_array: bool =
            LazyFrame::scan_parquet(coords_path.into(), Default::default())
                .ok()
                .and_then(|mut lf| lf.collect_schema().ok())
                .map(|s| s.get("coordinates").is_some())
                .unwrap_or(false);

        let coords = if coords_has_array {
            LazyFrame::scan_parquet(coords_path.into(), Default::default())?.select([
                col("id"),
                col("coordinates").arr().get(lit(0i64), true).alias("x"),
                col("coordinates").arr().get(lit(1i64), true).alias("y"),
            ])
        } else {
            LazyFrame::scan_parquet(coords_path.into(), Default::default())?.select([
                col("id"),
                col("x"),
                col("y"),
            ])
        };

        // Check whether the labels parquet has an optional `info` column.
        let has_info: bool = LazyFrame::scan_parquet(labels_path.into(), Default::default())
            .ok()
            .and_then(|mut lf| lf.collect_schema().ok())
            .map(|s| s.get("info").is_some())
            .unwrap_or(false);

        let mut labels_cols = vec![col("id"), col("labels")];
        if has_info {
            labels_cols.push(col("info"));
        }
        let labels_lf = LazyFrame::scan_parquet(labels_path.into(), Default::default())?
            .select(labels_cols);

        let df = coords
            .join(
                labels_lf,
                [col("id")],
                [col("id")],
                JoinArgs::new(JoinType::Left),
            )
            .collect()?;

        let n = df.height();
        let ids = df.column("id")?.str()?;
        let xs = df.column("x")?.cast(&DataType::Float32)?;
        let ys = df.column("y")?.cast(&DataType::Float32)?;
        let xs = xs.f32()?;
        let ys = ys.f32()?;
        let cats_col = df.column("labels")?.str()?;

        // Collect info strings up front (before the point loop consumes other iterators).
        let info_vec: Vec<String> = if has_info {
            let ic = df.column("info")?.cast(&DataType::String)?;
            ic.str()?
                .into_iter()
                .map(|v| v.unwrap_or("").to_string())
                .collect()
        } else {
            vec![String::new(); n]
        };

        // Assign a hue per category value.
        let mut category_map: HashMap<String, usize> = HashMap::new();
        let mut category_idx: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let cat = cats_col.get(i).unwrap_or("").to_string();
            let next = category_map.len();
            let idx = *category_map.entry(cat).or_insert(next);
            category_idx.push(idx);
        }
        let n_categories = category_map.len().max(1);

        let mut labels = Vec::with_capacity(n);
        let mut categories = Vec::with_capacity(n);
        let mut points = Vec::with_capacity(n);
        for ((id, x), (y, (cat, idx))) in ids
            .into_iter()
            .zip(xs.into_no_null_iter())
            .zip(ys.into_no_null_iter().zip(cats_col.into_iter().zip(category_idx.into_iter())))
        {
            let cat_str = cat.unwrap_or("").to_string();
            let (r, g, b) = if cat_str.is_empty() {
                UNLABELED_COLOR
            } else {
                hue_to_rgb(idx as f32 / n_categories as f32)
            };
            labels.push(id.unwrap_or("").to_string());
            categories.push(cat_str.clone());
            points.push(Point {
                x,
                y,
                r,
                g,
                b,
                highlight: 1.0,
                size: 1.0,
            });
        }

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

        let bounds = [xmin, xmax, ymin, ymax];
        let grid = SpatialGrid::build(&points, bounds);
        let all_categories = vec![categories.clone()];
        let label_set_names = Vec::new(); // populated by the caller (with_config)
                                          // Build default hue colour map for the initial label set.
        let default_color_map: ColorMap = category_map
            .iter()
            .map(|(name, &idx)| (name.clone(), hue_to_rgb(idx as f32 / n_categories as f32)))
            .collect();
        let category_color_maps = vec![default_color_map];
        let mut cloud = Self {
            points,
            bounds,
            grid,
            labels,
            categories,
            info: info_vec,
            label_set_names,
            all_categories,
            category_color_maps,
        };
        cloud.dim_unlabeled();
        Ok(cloud)
    }
}

/// Ray-casting point-in-polygon test (data space).
pub fn point_in_polygon(px: f32, py: f32, poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Box-Muller transform: two uniform samples → two Gaussian samples.
fn box_muller(rng: &mut impl Rng, sigma: f32) -> (f32, f32) {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>();
    let mag = sigma * (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    (mag * theta.cos(), mag * theta.sin())
}

pub fn hue_to_rgb(h: f32) -> (f32, f32, f32) {
    let h = h * 6.0;
    let i = h as u32;
    let f = h - i as f32;
    let q = 1.0 - f;
    match i % 6 {
        0 => (1.0, f, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, q, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, q),
    }
}
