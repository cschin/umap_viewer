#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

/// Runtime configuration loaded from a YAML file.
///
/// Default config file: `config.yaml` in the current working directory.
/// Override with: `--config <path>`
///
/// Example config.yaml:
/// ```yaml
/// coords_parquet: data/sample_umap_coordinates.parquet
/// labels_parquet: data/sample_labels.parquet
/// output_bin:     data/points.bin
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
pub struct Config {
    /// Path to the UMAP coordinates parquet (columns: id, coordinates).
    pub coords_parquet: String,
    /// Path to the labels/categories parquet (columns: id, labels).
    pub labels_parquet: String,
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
    pub fn from_args() -> Result<Self, Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().collect();
        let path = args.windows(2)
            .find(|w| w[0] == "--config")
            .map(|w| w[1].as_str())
            .unwrap_or("config.yaml");
        let mut cfg = Self::from_file(path)?;
        cfg.export_bin = args.contains(&"--export-bin".to_string());
        Ok(cfg)
    }
}
