struct Uniforms {
    transform: mat4x4<f32>,
    point_size: f32,
    viewport_aspect: f32,
    alpha: f32,
    _pad0: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) highlight: f32,
    @location(3) quad_offset: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) highlight: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let clip_center = uniforms.transform * vec4<f32>(in.pos, 0.0, 1.0);
    let pixel_offset = vec2<f32>(
        in.quad_offset.x * uniforms.point_size / uniforms.viewport_aspect,
        in.quad_offset.y * uniforms.point_size,
    );
    out.clip_pos = clip_center + vec4<f32>(pixel_offset, 0.0, 0.0);
    out.color = in.color;
    out.uv = in.quad_offset * 2.0;
    out.highlight = in.highlight;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if dist > 1.0 { discard; }

    let alpha = 1.0 - smoothstep(0.85, 1.0, dist);
    let outline = smoothstep(0.72, 0.78, dist) * (1.0 - smoothstep(0.79, 0.85, dist));
    let color = mix(in.color, vec3<f32>(0.05, 0.05, 0.05), outline * 0.7);

    return vec4<f32>(color, alpha * uniforms.alpha * in.highlight);
}
