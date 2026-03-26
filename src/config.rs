#[cfg(not(target_arch = "wasm32"))]
use indexmap::IndexMap;
#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

/// Accepts either a bare path string (legacy single file) or a YAML mapping of
/// `"Display Name": path/to/file.parquet` (multiple files, order preserved).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
#[serde(untagged)]
enum LabelParquetConfig {
    Single(String),
    Map(IndexMap<String, String>),
}

#[cfg(not(target_arch = "wasm32"))]
impl LabelParquetConfig {
    fn primary_path(&self) -> &str {
        match self {
            LabelParquetConfig::Single(p) => p.as_str(),
            LabelParquetConfig::Map(map) => map.values().next().map(|s| s.as_str()).unwrap_or(""),
        }
    }
}

/// Runtime configuration loaded from a YAML file.
///
/// Default config file: `config.yaml` in the current working directory.
/// Override with: `--config <path>`
///
/// Single label file (legacy):
/// ```yaml
/// coords_parquet: data/umap_coordinate.parquet
/// labels_parquet: data/umap_label.parquet
/// output_bin:     data/points.bin
/// ```
///
/// Multiple label files (key = UI label, value = path):
/// ```yaml
/// coords_parquet: data/umap_coordinate.parquet
/// labels_parquet:
///   Cell Type: data/umap_label.parquet
///   Cluster:   data/umap_cluster.parquet
/// output_bin: data/points.bin
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
pub struct Config {
    /// Path to the UMAP coordinates parquet (columns: id, coordinates).
    pub coords_parquet: String,
    /// Label parquet file(s). Either a bare path or a `"Name": path` mapping.
    labels_parquet: LabelParquetConfig,
    /// Optional per-label-set colour CSV files.
    /// Key = label set display name (must match a key in `labels_parquet`).
    /// Value = path to a CSV with columns `label,color` where color is `#RRGGBB`.
    /// Label sets without an entry here use the default hue-based colouring.
    #[serde(default)]
    label_colors: IndexMap<String, String>,
    /// Output path used by --export-bin.
    #[serde(default = "default_output_bin")]
    pub output_bin: String,
    /// When true (set via --export-bin CLI flag), export points.bin and exit.
    #[serde(default)]
    pub export_bin: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn default_output_bin() -> String {
    "data/points.bin".to_string()
}

#[cfg(not(target_arch = "wasm32"))]
impl Config {
    /// Load from a YAML file.  Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let cfg = serde_yaml::from_str(&text)?;
        Ok(cfg)
    }

    /// Resolve the config file path from args (`--config <path>`, default `config.yaml`)
    /// and load it.
    ///
    /// Returns:
    ///   `Ok(Some(cfg))` — config found and loaded.
    ///   `Ok(None)`      — no `--config` flag given and `config.yaml` does not exist;
    ///                     the caller should start with an empty dataset.
    ///   `Err(e)`        — `--config <path>` was given explicitly but failed to load.
    pub fn from_args() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().collect();

        if args.iter().any(|a| a == "--help" || a == "-h") {
            let bin = args.first().map(|s| s.as_str()).unwrap_or("umap_viewer");
            println!(
                "Usage: {bin} [OPTIONS]

Options:
  --config <path>   Path to the YAML config file (default: config.yaml)
  --export-bin      Export data as points.bin for the WASM build, then exit
  -h, --help        Show this help message and exit

Config file (YAML):
  coords_parquet    Path to the parquet file with UMAP coordinates
  labels_parquet    Path(s) to parquet file(s) with point labels
  output_bin        Output path for --export-bin (default: data/points.bin)

Starting without a config:
  If no config file is found the viewer starts with an empty canvas.
  Use the \"Load CSV\" button to load a joined CSV file
  (columns: id, x, y, labels[, info][, labels_*]).

WASM build:
  The WASM version can embed data at compile time via data/points.bin.
  Without that file it starts empty — use \"Load CSV\" in the browser.
  To build with embedded data:
    1. Run with --export-bin to write points.bin from the desired config
    2. Rebuild with: ./webapp_build.sh
"
            );
            std::process::exit(0);
        }

        let explicit = args.windows(2).find(|w| w[0] == "--config").map(|w| w[1].clone());
        let path = explicit.as_deref().unwrap_or("config.yaml");

        // If no explicit path was given and the default doesn't exist, start empty.
        if explicit.is_none() && !std::path::Path::new(path).exists() {
            return Ok(None);
        }

        let mut cfg = Self::from_file(path)?;
        cfg.export_bin = args.contains(&"--export-bin".to_string());
        Ok(Some(cfg))
    }

    /// Ordered list of `(display_name, path)` pairs for all label files.
    pub fn label_pairs(&self) -> Vec<(String, String)> {
        match &self.labels_parquet {
            LabelParquetConfig::Single(path) => {
                let name = std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("labels")
                    .to_string();
                vec![(name, path.clone())]
            }
            LabelParquetConfig::Map(map) => {
                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            }
        }
    }

    /// Path of the first (primary) label file — used for initial load and --export-bin.
    pub fn primary_labels_path(&self) -> &str {
        self.labels_parquet.primary_path()
    }

    /// Return the colour CSV path for a given label set name, if configured.
    pub fn color_file_for(&self, set_name: &str) -> Option<&str> {
        self.label_colors.get(set_name).map(|s| s.as_str())
    }
}
