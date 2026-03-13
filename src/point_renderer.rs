use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::data::Point;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub transform: [[f32; 4]; 4],
    pub point_size: f32,
    pub viewport_aspect: f32,
    pub alpha: f32,
    pub _pad: [f32; 1],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadVertex {
    offset: [f32; 2],
}

const QUAD_VERTICES: &[QuadVertex] = &[
    QuadVertex { offset: [-0.5, -0.5] },
    QuadVertex { offset: [ 0.5, -0.5] },
    QuadVertex { offset: [ 0.5,  0.5] },
    QuadVertex { offset: [-0.5, -0.5] },
    QuadVertex { offset: [ 0.5,  0.5] },
    QuadVertex { offset: [-0.5,  0.5] },
];

pub struct PointRenderer {
    pipeline: wgpu::RenderPipeline,
    point_buf: wgpu::Buffer,
    quad_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    n_points: u32,
}

impl PointRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        points: &[Point],
    ) -> Self {
        let shader_src = include_str!("shaders/points.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let point_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_buf"),
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let quad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_buf"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buf"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    // buffer slot 0: per-instance point data (x,y,r,g,b,highlight)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Point>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2, // pos
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3, // color
                            },
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,   // highlight
                            },
                        ],
                    },
                    // buffer slot 1: per-vertex quad corner
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self { pipeline, point_buf, quad_buf, uniform_buf, bind_group, n_points: points.len() as u32 }
    }

    /// Replace GPU point data.
    pub fn update_points(&self, queue: &wgpu::Queue, points: &[Point]) {
        queue.write_buffer(&self.point_buf, 0, bytemuck::cast_slice(points));
    }

    /// Expose the uniform buffer so `app.rs` can write to it from `prepare()`.
    pub fn uniform_buf(&self) -> &wgpu::Buffer {
        &self.uniform_buf
    }

    /// Record draw calls — called from `paint()`.
    pub fn draw<'rp>(&'rp self, rpass: &mut wgpu::RenderPass<'rp>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.point_buf.slice(..));
        rpass.set_vertex_buffer(1, self.quad_buf.slice(..));
        rpass.draw(0..6, 0..self.n_points);
    }
}

// ---------------------------------------------------------------------------
// View transform helper
// ---------------------------------------------------------------------------

/// Returns a column-major 4×4 orthographic transform for pan/zoom.
pub fn build_transform(
    pan: [f32; 2],
    zoom: f32,
    viewport_width: f32,
    viewport_height: f32,
    data_bounds: [f32; 4], // xmin, xmax, ymin, ymax
) -> [[f32; 4]; 4] {
    let [xmin, xmax, ymin, ymax] = data_bounds;
    let cx = (xmin + xmax) * 0.5;
    let cy = (ymin + ymax) * 0.5;

    // Half-span in data space at current zoom
    let span_x = (xmax - xmin) * 0.5 / zoom;
    let span_y = (ymax - ymin) * 0.5 / zoom;

    // Scale so data fills the viewport, corrected for aspect ratio
    let aspect = viewport_width / viewport_height;
    let (sx, sy) = if aspect >= 1.0 {
        (1.0 / (span_x * aspect), 1.0 / span_y)
    } else {
        (1.0 / span_x, aspect / span_y)
    };

    let tx = -(cx + pan[0]) * sx;
    let ty = -(cy + pan[1]) * sy;

    [
        [sx,  0.0, 0.0, 0.0],
        [0.0, sy,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx,  ty,  0.0, 1.0],
    ]
}
