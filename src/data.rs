#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
use rand::prelude::*;

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
        Self { cells, xmin, ymin, cell_w, cell_h, cols, rows }
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
                    if d2 <= r2 {
                        if best.map_or(true, |(_, bd)| d2 < bd) {
                            best = Some((pi as usize, d2));
                        }
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
}

pub struct PointCloud {
    pub points: Vec<Point>,
    pub bounds: [f32; 4], // xmin, xmax, ymin, ymax
    pub grid: SpatialGrid,
    /// Per-point ID string. May be empty if not loaded.
    pub labels: Vec<String>,
    /// Per-point category string used for colouring and tooltips.
    pub categories: Vec<String>,
}

impl PointCloud {
    /// Generate 500k synthetic UMAP-like points: Gaussian clusters at random centers.
    pub fn generate_test(n_points: usize, n_clusters: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut points = Vec::with_capacity(n_points);

        // Cluster centers spread across [-12, 12]^2
        let centers: Vec<(f32, f32)> = (0..n_clusters)
            .map(|_| (rng.gen_range(-12.0_f32..12.0_f32), rng.gen_range(-12.0_f32..12.0_f32)))
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
            points.push(Point { x: cx + dx, y: cy + dy, r, g, b, highlight: 1.0 });
        }

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

        let bounds = [xmin, xmax, ymin, ymax];
        let grid = SpatialGrid::build(&points, bounds);
        Self { points, bounds, grid, labels: Vec::new(), categories: Vec::new() }
    }

    /// Apply a polygon selection: selected points get highlight=1.0, others 0.15.
    /// Returns the indices of selected points.
    pub fn apply_polygon_selection(&mut self, poly: &[[f32; 2]]) -> Vec<usize> {
        let mut indices = Vec::new();
        for (i, p) in self.points.iter_mut().enumerate() {
            let sel = point_in_polygon(p.x, p.y, poly);
            p.highlight = if sel { 1.0 } else { 0.15 };
            if sel { indices.push(i); }
        }
        indices
    }

    /// Parse a pre-built binary blob (works on all targets including WASM).
    ///
    /// Format:
    ///   magic    : b"UMAP" (4 bytes)
    ///   n_points : u32 le
    ///   n_cats   : u32 le
    ///   for each category: name_len u32 le + utf-8 bytes
    ///   id_stride: u32 le  (fixed byte length of each ID; 0 = no IDs stored)
    ///   x        : [f32 le; n_points]
    ///   y        : [f32 le; n_points]
    ///   cat_idx  : [u32 le; n_points]
    ///   id_blob  : [u8; n_points * id_stride]  (only present when id_stride > 0)
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

        if &data[p..p + 4] != b"UMAP" { return Err("bad magic".into()); }
        p += 4;

        let n_points  = read_u32!() as usize;
        let n_cats    = read_u32!() as usize;
        let id_stride = read_u32!() as usize;

        let mut cat_names: Vec<String> = Vec::with_capacity(n_cats);
        for _ in 0..n_cats {
            let len = read_u32!() as usize;
            cat_names.push(std::str::from_utf8(&data[p..p + len])?.to_string());
            p += len;
        }

        let mut xs = Vec::with_capacity(n_points);
        let mut ys = Vec::with_capacity(n_points);
        for _ in 0..n_points { xs.push(read_f32!()); }
        for _ in 0..n_points { ys.push(read_f32!()); }

        let mut cat_indices: Vec<usize> = Vec::with_capacity(n_points);
        for _ in 0..n_points { cat_indices.push(read_u32!() as usize); }

        // IDs: fixed-stride flat blob
        let mut labels = Vec::with_capacity(n_points);
        if id_stride > 0 {
            let blob = &data[p..p + n_points * id_stride];
            for i in 0..n_points {
                let raw = &blob[i * id_stride..(i + 1) * id_stride];
                let trimmed = raw.iter().rposition(|&b| b != 0).map(|p| &raw[..=p]).unwrap_or(&raw[..0]);
                labels.push(std::str::from_utf8(trimmed)?.to_string());
            }
        }

        let n_cats_f = n_cats.max(1) as f32;
        let mut points     = Vec::with_capacity(n_points);
        let mut categories = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let (r, g, b) = hue_to_rgb(cat_indices[i] as f32 / n_cats_f);
            points.push(Point { x: xs[i], y: ys[i], r, g, b, highlight: 1.0 });
            categories.push(cat_names[cat_indices[i]].clone());
        }

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let bounds = [xmin, xmax, ymin, ymax];
        let grid   = SpatialGrid::build(&points, bounds);
        Ok(Self { points, bounds, grid, labels, categories })
    }

    /// Export to the compact binary format consumed by `from_bin`.
    /// Run the native binary with `--export-bin` to generate `data/points.bin`
    /// before doing a WASM build.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_to_bin(
        coords_path: &str,
        labels_path: &str,
        out_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cloud = Self::load_from_parquet(coords_path, labels_path)?;

        // Build a deduplicated category catalog preserving first-seen order.
        let mut cat_map: std::collections::HashMap<&str, u32> = Default::default();
        let mut cat_names: Vec<&str> = Vec::new();
        let mut cat_indices: Vec<u32> = Vec::with_capacity(cloud.points.len());
        for cat in &cloud.categories {
            let next = cat_map.len() as u32;
            let idx = *cat_map.entry(cat.as_str()).or_insert_with(|| {
                cat_names.push(cat.as_str());
                next
            });
            cat_indices.push(idx);
        }

        let n = cloud.points.len();
        // Use the max ID byte length as stride; shorter IDs are null-padded.
        let id_stride = cloud.labels.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut buf: Vec<u8> = Vec::with_capacity(4 + 4 + 4 + 4 + n * (12 + id_stride));
        buf.extend_from_slice(b"UMAP");
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(cat_names.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(id_stride as u32).to_le_bytes());
        for name in &cat_names {
            let bytes = name.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        for p in &cloud.points { buf.extend_from_slice(&p.x.to_le_bytes()); }
        for p in &cloud.points { buf.extend_from_slice(&p.y.to_le_bytes()); }
        for idx in &cat_indices { buf.extend_from_slice(&idx.to_le_bytes()); }
        if id_stride > 0 {
            for id in &cloud.labels {
                let bytes = id.as_bytes();
                buf.extend_from_slice(bytes);
                buf.resize(buf.len() + (id_stride - bytes.len()), 0u8);
            }
        }

        std::fs::write(out_path, &buf)?;
        println!(
            "Exported {} points ({} categories, id_stride={}) → {}",
            n, cat_names.len(), id_stride, out_path
        );
        Ok(())
    }

    /// Reset all highlights to 1.0 (clears any selection).
    pub fn clear_selection(&mut self) {
        for p in &mut self.points {
            p.highlight = 1.0;
        }
    }

    /// Load UMAP coordinates and labels from parquet files.
    /// `coords_path`: schema `id` (Utf8), `coordinates` (FixedSizeList<f32, 2>).
    /// `labels_path`: schema `id` (Utf8), `labels` (Utf8).
    /// The two files are joined on `id`; points are coloured by their label.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_parquet(coords_path: &str, labels_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use polars::prelude::*;

        let coords = LazyFrame::scan_parquet(coords_path.into(), Default::default())?
            .select([
                col("id"),
                col("coordinates").arr().get(lit(0i64), true).alias("x"),
                col("coordinates").arr().get(lit(1i64), true).alias("y"),
            ]);

        let labels_lf = LazyFrame::scan_parquet(labels_path.into(), Default::default())?
            .select([col("id"), col("labels")]);

        let df = coords
            .join(labels_lf, [col("id")], [col("id")], JoinArgs::new(JoinType::Left))
            .collect()?;

        let n = df.height();
        let ids      = df.column("id")?.str()?;
        let xs       = df.column("x")?.cast(&DataType::Float32)?;
        let ys       = df.column("y")?.cast(&DataType::Float32)?;
        let xs       = xs.f32()?;
        let ys       = ys.f32()?;
        let cats_col = df.column("labels")?.str()?;

        // Assign a hue per category value.
        let mut category_map: HashMap<String, usize> = HashMap::new();
        let mut category_idx: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let cat  = cats_col.get(i).unwrap_or("").to_string();
            let next = category_map.len();
            let idx  = *category_map.entry(cat).or_insert(next);
            category_idx.push(idx);
        }
        let n_categories = category_map.len().max(1);

        let mut labels     = Vec::with_capacity(n);
        let mut categories = Vec::with_capacity(n);
        let mut points     = Vec::with_capacity(n);
        for i in 0..n {
            let x = xs.get(i).unwrap_or(0.0);
            let y = ys.get(i).unwrap_or(0.0);
            let (r, g, b) = hue_to_rgb(category_idx[i] as f32 / n_categories as f32);
            labels.push(ids.get(i).unwrap_or("").to_string());
            categories.push(cats_col.get(i).unwrap_or("").to_string());
            points.push(Point { x, y, r, g, b, highlight: 1.0 });
        }

        let xmin = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let xmax = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let ymin = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

        let bounds = [xmin, xmax, ymin, ymax];
        let grid   = SpatialGrid::build(&points, bounds);
        Ok(Self { points, bounds, grid, labels, categories })
    }
}

/// Ray-casting point-in-polygon test (data space).
pub fn point_in_polygon(px: f32, py: f32, poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    if n < 3 { return false; }
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

fn hue_to_rgb(h: f32) -> (f32, f32, f32) {
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
