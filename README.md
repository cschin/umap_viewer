# LatentAnalyzer

An interactive GPU-accelerated viewer for 2D latent-space embeddings (UMAP, t-SNE, PCA, etc.). Supports hundreds of thousands of points with smooth pan/zoom, polygon selection, category filtering, a sticky tooltip with clickable links, and a sortable data table. Runs as a native desktop app or in the browser via WebAssembly.

## 🚀 [Try the live demo](https://cschin.github.io/umap_viewer/)

No install required — runs entirely in your browser via WebAssembly. The demo loads a pre-built dataset of arXiv ML paper embeddings with multiple label sets. Works on desktop and mobile (pinch to zoom, two-finger pan).

---

![Controls: pan, zoom, polygon select, category histogram, sortable table](misc/umap_viewer_screenshot.gif)

---

## Features

- **GPU rendering** — instanced quad rendering via wgpu; antialiased circles; handles 500k+ points at interactive frame rates
- **Pan / zoom** — drag to pan, scroll wheel to zoom (0.05× – 500×)
- **Polygon selection** — click to place vertices, close near the start to select; right-click to cancel
- **Multiple label sets** — load several label/category files and switch between them instantly; colours update live
- **Custom colours** — optional per-label-set colour CSV files (`label,#RRGGBB`); unspecified labels fall back to evenly-spaced hues
- **Unlabeled point styling** — points with no assigned label rendered in mid-gray at 50% opacity; histogram bar matches
- **Category histogram** — right panel shows distribution of selected points; click a category to highlight it at 2× size; collapsible
- **Sortable table** — bottom panel lists selected points with sortable columns; collapsible
- **Sticky tooltip with links** — hover over a point to see its label, ID, and info. Click the point to **pin** the tooltip. The pinned tooltip tracks the point as you pan/zoom, stays within the canvas, and shows a callout triangle pointing toward the point. If the `info` field contains a `[text](url)` markdown link it is rendered as a clickable hyperlink. Close with **✕** or by clicking empty canvas
- **Info search** — the search box in the bottom panel filters by content in the `info` column as well as label and ID
- **Go to selected / Clear focus** — buttons in the bottom panel scroll back to the pinned row or remove category focus without losing the selection
- **Direct CSV loading** — load a joined CSV with no intermediate conversion steps. Accepted formats: `.csv`, `.csv.gz` (gzip), `.zip` (first `.csv` entry). Works on native and WASM
- **CSV format help** — click the **?** button next to "Load CSV…" for a description of required columns and an example row
- **Inline load errors** — if a CSV fails to parse, a red error message appears in the left panel with a Dismiss button; the app never crashes
- **Empty-state welcome** — when no data is loaded the canvas shows a prompt to use Load CSV…
- **Export selected IDs** — save the IDs of all selected points to a text file; native opens a save dialog; WASM triggers a browser download
- **Collapsible control panel** — collapses to a narrow tab to maximise canvas space
- **Touch / mobile support** *(WASM)* — pinch to zoom, two-finger drag to pan

---

## Architecture

```
umap_viewer/
├── src/
│   ├── main.rs            # Entry points: native CLI + WASM init
│   ├── lib.rs             # Module re-exports
│   ├── config.rs          # config.yaml loader (native only)
│   ├── data.rs            # PointCloud, SpatialGrid, parquet/binary/CSV I/O, ColorMap
│   ├── app.rs             # egui application: UI panels, interaction, rendering
│   ├── point_renderer.rs  # wgpu pipeline, vertex buffers, uniforms
│   └── shaders/
│       └── points.wgsl    # WGSL vertex + fragment shaders
├── fonts/
│   └── .gitkeep           # placeholder — SFNSMono.ttf not tracked (see Prerequisites)
├── data/
│   ├── arxiv_ml_data_map.parquet
│   ├── arxiv_ml_layer[0-4]_cluster_labels.parquet
│   ├── cluster_colors.csv
│   └── points.bin                         # pre-serialised binary for WASM (--export-bin)
├── support_scripts/
│   └── test_data/
│       ├── gen_joined_csv.py              # generates a test joined CSV
│       └── test_joined.csv               # 100-point test file for Load CSV
├── misc/
│   └── umap_viewer_screenshot.{gif,mp4,png}
├── build.rs               # detects data/points.bin at compile time for WASM embedding
├── Cargo.toml
├── Trunk.toml
├── config.yaml
├── index.html
├── install_fonts.sh
└── README.md
```

### Key modules

| Module | Responsibility |
|---|---|
| `data.rs` | Loads parquet files (native), a compact binary blob (WASM), or a joined CSV on any target. Defines `ColorMap` and `PointCloud`. Builds a `SpatialGrid` for O(1) hover hit-testing. Auto-detects gzip/zip compression when loading CSV. |
| `config.rs` | Deserialises `config.yaml`: coordinates path, one or more label parquet files, optional colour CSVs. Returns `None` if no config is found so the app starts empty. |
| `point_renderer.rs` | Creates the wgpu render pipeline. Each point is an instanced quad. Per-instance data: position, RGB colour, highlight factor. |
| `shaders/points.wgsl` | Vertex shader applies pan/zoom transform and quad offset. Fragment shader renders antialiased circles via distance. |
| `app.rs` | Immediate-mode egui UI: left control panel, right histogram panel, bottom table panel, central wgpu canvas. Manages selection, sort, polygon, tooltip, CSV upload, and label-set switching. |

### Data flow

```
Parquet / CSV / .bin ──► PointCloud ──► Vec<Point> (x,y,r,g,b,highlight)
        +                     │                    │
   colour CSVs          SpatialGrid          wgpu vertex buffer
   (ColorMap)           (hover lookup)       (re-created on reload / selection change)
                              │
                       all_categories        label selector UI
                       (pre-loaded)          (recolours points live)
```

---

## Data formats

### Joined CSV (recommended — works on native and WASM)

A single CSV file with all data. Compressed variants (`.csv.gz`, `.zip`) are decompressed automatically.

| Column | Required | Description |
|---|---|---|
| `id` | yes | Unique identifier for each point |
| `x` | yes | Embedding x coordinate (float) |
| `y` | yes | Embedding y coordinate (float) |
| `labels` | yes | Primary category label |
| `info` | no | Plain text or `[text](url)` markdown link shown in the tooltip |
| `labels_<name>` | no | Additional label sets (e.g. `labels_cluster`) |

Example row:
```
cell_001,3.14,-1.27,TypeA,"see [paper](https://example.com)",clust_5
```

A test file is provided at `support_scripts/test_data/test_joined.csv`. To regenerate it:
```bash
python3 support_scripts/test_data/gen_joined_csv.py
```

### Parquet (native builds only)

One coordinates file plus one or more label files, all joined on `id`:

| File | Required columns | Arrow / Parquet type |
|---|---|---|
| `coords_parquet` | `id` (Utf8), `coordinates` (FixedSizeList\<Float32\>[2]) | |
| `labels_parquet` | `id` (Utf8), `labels` (Utf8) | |

#### Generating parquet files with Python

```python
import polars as pl

coords = pl.DataFrame({
    "id": [f"pt{i}" for i in range(n)],
    "coordinates": pl.Series(embedding.astype("float32").tolist())
                     .cast(pl.Array(pl.Float32, 2)),
})
coords.write_parquet("data/umap_coordinate.parquet")

labels = pl.DataFrame({"id": [f"pt{i}" for i in range(n)], "labels": category_strings})
labels.write_parquet("data/umap_label.parquet")
```

### Colour CSV (optional, native builds)

```csv
label,color
TypeA,#E74C3C
TypeB,#3498DB
```

Rules: header row is optional; lines starting with `#` are comments; colour must be `#RRGGBB`; unlisted categories use hue defaults.

### `points.bin` (WASM builds)

Compact binary produced by `--export-bin`. Bakes all label sets and resolved colours. The WASM build includes this file at compile time via `include_bytes!` (controlled by `build.rs`). If `data/points.bin` does not exist the WASM build starts empty and loads CSV from the browser.

```
[0:4]    magic        "UMAP"
[4:8]    u32le        n_points
[8:12]   u32le        n_label_sets
[12:16]  u32le        id_stride

for each label set:
  u32le + utf-8      set display name
  u32le              n_categories
  for each category:
    u32le + utf-8    category name
    u8, u8, u8       R, G, B

[u32le; n_points]              per-point category index (per label set)
[f32le; n_points]              x values
[f32le; n_points]              y values
[u8; n_points × id_stride]     ID strings (null-padded; omitted when id_stride = 0)
```

---

## Building & running

### Prerequisites

```bash
rustup target add wasm32-unknown-unknown   # for web builds
cargo install trunk                        # for web builds
```

**Font** — `SFNSMono.ttf` is not included. Copy SF Mono from macOS system fonts:

```bash
bash install_fonts.sh
```

On Linux/Windows, copy any monospace TTF with Unicode symbol coverage to `fonts/SFNSMono.ttf`.

### Native desktop

```bash
cargo run --release
```

Command-line options:

| Flag | Description |
|---|---|
| `--config <path>` | Path to config file (default: `config.yaml`) |
| `--export-bin` | Export `data/points.bin` and exit (required before WASM builds with embedded data) |
| `-h`, `--help` | Show help and exit |

If no `config.yaml` is found the app starts with an empty canvas — use **Load CSV…** to load data.

### Web / WASM

**Option A — Load CSV in the browser (no pre-build step needed):**

```bash
trunk build --release --public-url ./
```

Open `dist/index.html` and use **Load CSV…**.

**Option B — Embed data at compile time:**

```bash
# Step 1: export binary blob from config.yaml
cargo run --release -- --export-bin

# Step 2: build with embedded data
trunk build --release --public-url ./
```

Development server with hot reload:

```bash
trunk serve
```

`--public-url ./` emits relative asset paths for GitHub Pages / subdirectory hosting.

---

## Configuration (`config.yaml`)

```yaml
coords_parquet: data/arxiv_ml_data_map.parquet

labels_parquet:
  Layer 0: data/arxiv_ml_layer0_cluster_labels.parquet
  Layer 1: data/arxiv_ml_layer1_cluster_labels.parquet
  Layer 2: data/arxiv_ml_layer2_cluster_labels.parquet
  Layer 3: data/arxiv_ml_layer3_cluster_labels.parquet
  Layer 4: data/arxiv_ml_layer4_cluster_labels.parquet

# Optional per-label-set colour files (key must match labels_parquet key)
# label_colors:
#   Layer 0: data/cluster_colors.csv

output_bin: data/points.bin
```

Simple single-file config:

```yaml
coords_parquet: data/umap_coordinate.parquet
labels_parquet: data/umap_label.parquet
output_bin:     data/points.bin
```

---

## Usage

| Action | How |
|---|---|
| Pan | Navigate mode → drag (mouse or single finger on touch) |
| Zoom | Scroll wheel; pinch gesture on touch |
| Reset view | **Reset view** button |
| Load data | **Load CSV…** in the left panel (`.csv`, `.csv.gz`, `.zip`) |
| CSV format help | **?** button next to Load CSV… |
| Switch label set | Click a label name in the **Label set** selector |
| Enter selection mode | Click **Select** in the mode toggle |
| Add polygon vertex | Left-click on canvas |
| Close polygon | Left-click near the first vertex (≥ 3 vertices) |
| Cancel polygon | Right-click |
| Clear selection | **Clear selection** button |
| Export selected IDs | **Export IDs** button |
| Pin tooltip | Click a point — tooltip stays visible and tracks the point |
| Close tooltip | Click **✕** in the tooltip, or click empty canvas |
| Search table | Type in the search box (matches label, ID, and info) |
| Go to pinned row | **Go to selected** button in the bottom panel |
| Clear category focus | **Clear focus** button in the bottom panel |
| Focus a category | Click a category label in the histogram |
| Sort table | Click a column header; click again to reverse |
| Collapse left panel | Click **◀**; click the **▲ Controls** tab to expand |
| Collapse histogram | Click **▶**; click the **▼ Selected Labels** tab to expand |
| Collapse table | Click **▼**; click the **▲ Show table** tab to expand |

---

## Dependencies

| Crate | Purpose |
|---|---|
| `eframe` / `egui` | Cross-platform windowed app + immediate-mode UI |
| `egui_extras` | `TableBuilder` for the sortable bottom panel |
| `egui-wgpu` | egui ↔ wgpu paint callback integration |
| `wgpu` | GPU rendering (Metal / Vulkan / DX12 / WebGL2) |
| `bytemuck` | Zero-copy struct → GPU buffer casting |
| `csv` | Joined CSV parsing (all targets including WASM) |
| `flate2` | Gzip decompression for `.csv.gz` files |
| `zip` | Zip archive decompression for `.zip` files |
| `polars` | Parquet loading (native only) |
| `serde` / `serde_yaml` | `config.yaml` deserialisation (native only) |
| `indexmap` | Order-preserving map for multi-label config (native only) |
| `rfd` | Native file dialogs (native only) |
| `dirs` | Resolves `~/Downloads` (native only) |
| `wasm-bindgen` / `web-sys` / `js-sys` | Rust ↔ browser bindings (WASM only) |

---

## Example data

The files `data/arxiv_ml*` are sourced from the [datamapplot](https://github.com/TutteInstitute/datamapplot) project. Refer to that repository for the original licence and attribution.
