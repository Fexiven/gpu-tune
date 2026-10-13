//! The test loads. They run in a separate process (`gpu-tune --worker '<json>'`), so a
//! driver crash only kills the test, not the tuner.
//!
//! Everything is Vulkan, like games: NVIDIA runs Vulkan in P0 (game clocks), while CUDA
//! would force P2. Every pass is checked bit-for-bit on the GPU against a reference.

use crate::proto::{Kind, Params, WorkResult};
use anyhow::{Context, Result, anyhow, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

static LOST: AtomicBool = AtomicBool::new(false);

const RENDER_WGSL: &str = r#"
struct P { iters: u32, seed: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Game-shader-like mix: FP32 math (FMA, transcendental), texture sampling with filtering,
// random reads from a storage buffer (caches and memory), integer hashing.
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<u32> {
    var a = vec4<f32>(fract(pos.x * 0.0137), fract(pos.y * 0.0191), 0.5, 0.25);
    var h = u32(pos.x) * 1973u + u32(pos.y) * 9277u + prm.seed;
    for (var i = 0u; i < prm.iters; i = i + 1u) {
        h = h * 1664525u + 1013904223u;
        let uv = vec2<f32>(f32(h & 1023u), f32((h >> 10u) & 1023u)) * (1.0 / 1024.0) + a.xy;
        let t = textureSampleLevel(tex, smp, uv, 0.0);
        let l = lut[(h >> 16u) & 65535u];
        a = fma(a, t + vec4<f32>(0.5), l);
        a = fract(a * 1.37 + vec4<f32>(sin(a.x * 6.2), cos(a.y * 6.2), a.z * a.w, sqrt(a.w + 0.01)));
    }
    return bitcast<vec4<u32>>(a) ^ vec4<u32>(h, h >> 7u, h << 3u, ~h);
}
"#;

const COMPARE_WGSL: &str = r#"
@group(0) @binding(0) var a: texture_2d<u32>;
@group(0) @binding(1) var b: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>, 2>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let d = textureDimensions(a);
    if (g.x >= d.x || g.y >= d.y) { return; }
    let p = vec2<i32>(g.xy);
    if (any(textureLoad(a, p, 0) != textureLoad(b, p, 0))) { atomicAdd(&cnt[0], 1u); }
}
"#;

// Address-dependent pattern: every word depends on its own index and buffer, so data
// written to or read from the wrong address shows up as an error too.
const MEM_WGSL: &str = r#"
struct M { buf: u32, pad0: u32, pad1: u32, pad2: u32 }
fn pattern(i: u32, b: u32) -> u32 { return (i * 2654435761u) ^ (0x9E3779B9u + b * 0x85EBCA6Bu); }

@group(0) @binding(0) var<uniform> m: M;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(256) fn fill(@builtin(global_invocation_id) g: vec3<u32>,
                                      @builtin(num_workgroups) n: vec3<u32>) {
    let stride = n.x * 256u;
    let len = arrayLength(&data);
    for (var i = g.x; i < len; i = i + stride) { data[i] = pattern(i, m.buf); }
}
"#;

const CHECK_WGSL: &str = r#"
struct M { buf: u32, pad0: u32, pad1: u32, pad2: u32 }
fn pattern(i: u32, b: u32) -> u32 { return (i * 2654435761u) ^ (0x9E3779B9u + b * 0x85EBCA6Bu); }

@group(0) @binding(0) var<uniform> m: M;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>, 2>;
@compute @workgroup_size(256) fn check(@builtin(global_invocation_id) g: vec3<u32>,
                                       @builtin(num_workgroups) n: vec3<u32>) {
    let stride = n.x * 256u;
    let len = arrayLength(&data);
    var bad = 0u;
    for (var i = g.x; i < len; i = i + stride) {
        if (data[i] != pattern(i, m.buf)) { bad = bad + 1u; }
    }
    if (bad > 0u) { atomicAdd(&cnt[1], bad); }
}
"#;

const MEM_GROUPS: u32 = 1024;

struct Gfx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    name: String,
}

fn open(bus: &str) -> Result<Gfx> {
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::VULKAN));
    if adapters.is_empty() {
        bail!("no Vulkan device found (is the Vulkan loader / NVIDIA ICD installed?)");
    }
    let list = || {
        adapters
            .iter()
            .map(|a| {
                let i = a.get_info();
                format!("{} [{}]", i.name, i.device_pci_bus_id)
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let adapter = if bus == "any" {
        adapters
            .iter()
            .find(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .or(adapters.first())
    } else {
        adapters
            .iter()
            .find(|a| a.get_info().device_pci_bus_id.eq_ignore_ascii_case(bus))
    }
    .ok_or_else(|| anyhow!("no Vulkan device at PCI {bus}; found: {}", list()))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu-tune"),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .context("can't open the Vulkan device")?;
    device.set_device_lost_callback(|reason, msg| {
        LOST.store(true, Ordering::SeqCst);
        eprintln!("device lost ({reason:?}): {msg}");
    });
    device.on_uncaptured_error(Arc::new(|e| {
        eprintln!("GPU error: {e}");
        std::process::exit(4);
    }));
    Ok(Gfx {
        device,
        queue,
        limits,
        name: format!("{} ({})", info.name, info.driver_info),
    })
}

impl Gfx {
    fn wait(&self, idx: wgpu::SubmissionIndex) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(Duration::from_secs(20)),
            })
            .map_err(|e| anyhow!("GPU stopped responding: {e}"))?;
        if LOST.load(Ordering::SeqCst) {
            bail!("device lost");
        }
        Ok(())
    }

    fn submit_wait(&self, enc: wgpu::CommandEncoder) -> Result<()> {
        let idx = self.queue.submit([enc.finish()]);
        self.wait(idx)
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.encoder();
        enc.copy_buffer_to_buffer(src, 0, &rb, 0, size);
        self.submit_wait(enc)?;
        self.map_read(&rb)
    }

    fn map_read(&self, rb: &wgpu::Buffer) -> Result<Vec<u8>> {
        let (tx, rx) = std::sync::mpsc::channel();
        rb.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(20)),
            })
            .map_err(|e| anyhow!("GPU stopped responding: {e}"))?;
        rx.recv()?.map_err(|e| anyhow!("map failed: {e}"))?;
        let v = rb
            .get_mapped_range(..)
            .map_err(|e| anyhow!("{e:?}"))?
            .to_vec();
        rb.unmap();
        Ok(v)
    }
}

/// Small deterministic generator for the input data (same on every run).
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn fnv64(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for c in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h ^= u64::from_le_bytes(w);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// ------------------------------------------------------------------ render load
struct Render {
    pipe: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    /// self-test only: same image with a different seed, used for every 8th pass
    bad_bind: Option<wgpu::BindGroup>,
    passes: std::cell::Cell<u64>,
    target: wgpu::TextureView,
    reference: wgpu::Texture,
    ref_view: wgpu::TextureView,
    cmp_pipe: wgpu::ComputePipeline,
    cmp_bind: wgpu::BindGroup,
    w: u32,
    h: u32,
    iters: u32,
}

fn counters(g: &Gfx) -> wgpu::Buffer {
    g.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("counters"),
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

impl Render {
    fn new(g: &Gfx, p: &Params, cnt: &wgpu::Buffer) -> Result<Self> {
        let d = &g.device;
        let (w, h) = (p.width.max(16) / 16 * 16, p.height.max(16)); // row bytes stay 256-aligned
        let mut rng = XorShift(0x2545F4914F6CDD1D);

        let noise: Vec<u8> = (0..1024 * 1024 * 4)
            .map(|_| (rng.next() >> 56) as u8)
            .collect();
        let tex = d.create_texture_with_data(
            &g.queue,
            &wgpu::TextureDescriptor {
                label: Some("noise"),
                size: wgpu::Extent3d {
                    width: 1024,
                    height: 1024,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &noise,
        );
        let lut: Vec<f32> = (0..65536 * 4).map(|_| rng.unit() - 0.5).collect();
        let lut_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lut"),
            contents: as_bytes(&lut),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uni = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: as_bytes(&[p.iters, 0x1234_5678u32, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(RENDER_WGSL.into()),
        });
        let fmt = wgpu::TextureFormat::Rgba32Uint;
        let pipe = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let tex_view = tex.create_view(&Default::default());
        let mk_bind = |u: &wgpu::Buffer| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lut_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let bind = mk_bind(&uni);
        let bad_bind = p.inject.then(|| {
            mk_bind(&d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: as_bytes(&[p.iters, 0x1234_5679u32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            }))
        });
        let mk_target = |label| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let target_tex = mk_target("target");
        let reference = mk_target("reference");
        let target = target_tex.create_view(&Default::default());
        let ref_view = reference.create_view(&Default::default());

        let cmp_mod = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compare"),
            source: wgpu::ShaderSource::Wgsl(COMPARE_WGSL.into()),
        });
        let cmp_pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compare"),
            layout: None,
            module: &cmp_mod,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cmp_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cmp_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ref_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cnt.as_entire_binding(),
                },
            ],
        });
        let _ = target_tex; // the view keeps the texture alive
        Ok(Render {
            pipe,
            bind,
            bad_bind,
            passes: Default::default(),
            target,
            reference,
            ref_view,
            cmp_pipe,
            cmp_bind,
            w,
            h,
            iters: p.iters,
        })
    }

    fn draw(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        self.draw_with(enc, view, &self.bind)
    }

    fn draw_with(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        bind: &wgpu::BindGroup,
    ) {
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&self.pipe);
        rp.set_bind_group(0, bind, &[]);
        rp.draw(0..3, 0..1);
    }

    /// one pass: render the image and compare it with the reference, on the GPU
    fn pass(&self, enc: &mut wgpu::CommandEncoder) {
        let n = self.passes.get() + 1;
        self.passes.set(n);
        match &self.bad_bind {
            Some(bad) if n.is_multiple_of(8) => self.draw_with(enc, &self.target, bad),
            _ => self.draw(enc, &self.target),
        }
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cp.set_pipeline(&self.cmp_pipe);
        cp.set_bind_group(0, &self.cmp_bind, &[]);
        cp.dispatch_workgroups(self.w.div_ceil(16), self.h.div_ceil(16), 1);
    }

    /// render the reference once and hash it, so different runs can be compared too
    fn make_reference(&self, g: &Gfx) -> Result<String> {
        let mut enc = g.encoder();
        self.draw(&mut enc, &self.ref_view);
        let size = self.w as u64 * self.h as u64 * 16;
        let rb = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            self.reference.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.w * 16),
                    rows_per_image: Some(self.h),
                },
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
        g.submit_wait(enc)?;
        Ok(fnv64(&g.map_read(&rb)?))
    }

    fn work_per_pass(&self) -> f64 {
        self.w as f64 * self.h as f64 * self.iters as f64
    }
}

// ------------------------------------------------------------------ memory load
struct Mem {
    src: Vec<wgpu::Buffer>,
    dst: Vec<wgpu::Buffer>,
    check_src: Vec<wgpu::BindGroup>,
    check_dst: Vec<wgpu::BindGroup>,
    check_pipe: wgpu::ComputePipeline,
    bytes: u64,
    inject: bool,
}

impl Mem {
    fn new(g: &Gfx, mb: u64, cnt: &wgpu::Buffer, inject: bool) -> Result<Self> {
        let d = &g.device;
        let total = mb.max(16) * 1024 * 1024;
        let max = g
            .limits
            .max_storage_buffer_binding_size
            .min(g.limits.max_buffer_size)
            .min(1 << 30);
        let chunk = (max / (4 << 20)) * (4 << 20);
        if chunk == 0 {
            bail!("device allows only tiny buffers ({max} bytes)");
        }
        let n = total.div_ceil(chunk) as usize;
        let sizes: Vec<u64> = (0..n)
            .map(|i| {
                if i + 1 < n {
                    chunk
                } else {
                    total - chunk * (n as u64 - 1)
                }
            })
            .collect();
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mk = |label, s: u64| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: s,
                usage,
                mapped_at_creation: false,
            })
        };
        let src: Vec<_> = sizes.iter().map(|&s| mk("src", s)).collect();
        let dst: Vec<_> = sizes.iter().map(|&s| mk("dst", s)).collect();
        let unis: Vec<_> = (0..n)
            .map(|i| {
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: as_bytes(&[i as u32, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

        let compute = |src_code: &str, entry: &str| {
            let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src_code.into()),
            });
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fill_pipe = compute(MEM_WGSL, "fill");
        let check_pipe = compute(CHECK_WGSL, "check");

        // fill the source with the pattern
        let mut enc = g.encoder();
        for (i, b) in src.iter().enumerate() {
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &fill_pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: unis[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&fill_pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(MEM_GROUPS, 1, 1);
        }
        g.submit_wait(enc)?;
        let check_bind = |bufs: &Vec<wgpu::Buffer>| -> Vec<wgpu::BindGroup> {
            bufs.iter()
                .enumerate()
                .map(|(i, b)| {
                    d.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &check_pipe.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: unis[i].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cnt.as_entire_binding(),
                            },
                        ],
                    })
                })
                .collect()
        };
        let check_src = check_bind(&src);
        let check_dst = check_bind(&dst);
        Ok(Mem {
            src,
            dst,
            check_src,
            check_dst,
            check_pipe,
            bytes: total,
            inject,
        })
    }

    /// `count` rounds of copying everything; returns (bytes moved, seconds)
    fn copies(&self, g: &Gfx, count: u32) -> Result<(f64, f64)> {
        let mut enc = g.encoder();
        for _ in 0..count {
            for (s, d) in self.src.iter().zip(&self.dst) {
                enc.copy_buffer_to_buffer(s, 0, d, 0, s.size());
            }
        }
        let t0 = Instant::now();
        g.submit_wait(enc)?;
        if self.inject {
            g.queue
                .write_buffer(&self.dst[0], 4096, &[0xde, 0xad, 0xbe, 0xef]);
        }
        Ok((
            2.0 * self.bytes as f64 * count as f64,
            t0.elapsed().as_secs_f64(),
        ))
    }

    fn check(&self, g: &Gfx, full: bool) -> Result<()> {
        let mut enc = g.encoder();
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.check_pipe);
            let sets: Vec<&wgpu::BindGroup> = self
                .check_dst
                .iter()
                .chain(if full {
                    self.check_src.iter()
                } else {
                    [].iter()
                })
                .collect();
            for bg in sets {
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(MEM_GROUPS, 1, 1);
            }
        }
        g.submit_wait(enc)
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn read_counters(g: &Gfx, cnt: &wgpu::Buffer) -> Result<(u64, u64)> {
    let v = g.read_buffer(cnt, 16)?;
    let w = |i: usize| u32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap()) as u64;
    Ok((w(0), w(1)))
}

// ------------------------------------------------------------------ the tests
pub fn run(p: &Params) -> Result<WorkResult> {
    let g = open(&p.bus)?;
    let cnt = counters(&g);
    let mut r = WorkResult {
        adapter: g.name.clone(),
        ..Default::default()
    };
    let deadline = |s: f64| Instant::now() + Duration::from_secs_f64(s.max(0.0));
    match p.kind {
        Kind::Ping => {
            let mem = Mem::new(&g, 16, &cnt, false)?;
            mem.copies(&g, 1)?;
            mem.check(&g, true)?;
        }
        Kind::Heavy => {
            let rd = Render::new(&g, p, &cnt)?;
            r.hash = Some(rd.make_reference(&g)?);
            let batch = |n: u32| -> Result<()> {
                let mut enc = g.encoder();
                for _ in 0..n {
                    rd.pass(&mut enc);
                }
                g.submit_wait(enc)
            };
            let end = deadline(p.warmup);
            while Instant::now() < end {
                batch(4)?;
            }
            let (t0, end) = (Instant::now(), deadline(p.seconds));
            let mut passes = 0u64;
            while Instant::now() < end {
                batch(4)?;
                passes += 4;
            }
            r.perf = Some(passes as f64 * rd.work_per_pass() / t0.elapsed().as_secs_f64() / 1e9);
        }
        Kind::Paced => {
            let rd = Render::new(&g, p, &cnt)?;
            r.hash = Some(rd.make_reference(&g)?);
            let frame = 1.0 / p.fps.max(1) as f64;
            let (mut frames, mut missed, mut busy) = (0u64, 0u64, 0.0);
            let end = deadline(p.seconds);
            while Instant::now() < end {
                let fs = Instant::now();
                let mut enc = g.encoder();
                for _ in 0..p.per_frame.max(1) {
                    rd.pass(&mut enc);
                }
                g.submit_wait(enc)?;
                let dt = fs.elapsed().as_secs_f64();
                busy += dt;
                frames += 1;
                if dt > frame {
                    missed += 1;
                } else {
                    std::thread::sleep(Duration::from_secs_f64(frame - dt));
                }
            }
            r.frames = Some(frames);
            r.missed = Some(missed);
            r.busy = Some(busy / frames.max(1) as f64 / frame);
        }
        Kind::Mem => {
            let mem = Mem::new(&g, p.mem_mb, &cnt, p.inject)?;
            let (mut moved, mut secs) = (0.0, 0.0);
            let end = deadline(p.seconds);
            while Instant::now() < end {
                let (b, s) = mem.copies(&g, 10)?;
                moved += b;
                secs += s;
                mem.check(&g, false)?;
            }
            mem.check(&g, true)?;
            r.bw = Some(moved / secs.max(1e-9) / 1e9);
        }
        Kind::Combo => {
            let rd = Render::new(&g, p, &cnt)?;
            r.hash = Some(rd.make_reference(&g)?);
            let mem = Mem::new(&g, p.mem_mb, &cnt, p.inject)?;
            let core = |n: u32| -> Result<()> {
                let mut enc = g.encoder();
                for _ in 0..n {
                    rd.pass(&mut enc);
                }
                g.submit_wait(enc)
            };
            let end = deadline(p.warmup);
            while Instant::now() < end {
                core(4)?;
            }
            let (mut moved, mut secs) = (0.0, 0.0);
            let end = deadline(p.seconds);
            while Instant::now() < end {
                core(3)?;
                let (b, s) = mem.copies(&g, 3)?;
                moved += b;
                secs += s;
                mem.check(&g, false)?;
            }
            mem.check(&g, true)?;
            r.bw = Some(moved / secs.max(1e-9) / 1e9);
        }
    }
    let (c, m) = read_counters(&g, &cnt)?;
    r.core_errors = c;
    r.mem_errors = m;
    Ok(r)
}

/// Entry point of the worker process: prints one RESULT line, or exits non-zero.
pub fn worker_main(json: &str) -> ! {
    let p: Params = match serde_json::from_str(json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bad worker parameters: {e}");
            std::process::exit(2)
        }
    };
    match run(&p) {
        Ok(r) => {
            println!("RESULT {}", serde_json::to_string(&r).unwrap());
            std::process::exit(0)
        }
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(3)
        }
    }
}
