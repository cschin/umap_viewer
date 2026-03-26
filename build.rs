// build.rs — set a cfg flag when data/points.bin exists at compile time.
// The WASM entry point uses this to conditionally embed the binary blob.
fn main() {
    let bin = std::path::Path::new("data/points.bin");
    if bin.exists() {
        println!("cargo:rustc-cfg=has_embedded_bin");
    }
    // Tell the compiler this cfg flag is intentionally set by this build script.
    println!("cargo::rustc-check-cfg=cfg(has_embedded_bin)");
    // Re-run this script if the bin appears or disappears.
    println!("cargo:rerun-if-changed=data/points.bin");
}
