#[cxx::bridge(namespace = "dxil_spirv_shader_recompiler")]
mod ffi {
    #[derive(Clone, Debug)]
    struct VertexInput {
        semantic: String,
        semantic_index: u32,
        location: u32,
    }

    struct Converted {
        spirv: Vec<u8>,
        vertex_inputs: Vec<VertexInput>,
    }

    unsafe extern "C++" {
        include!("dxil-spirv-shader-recompiler-sys/include/bridge.h");

        fn convert_dxil(dxbc: &[u8]) -> Result<Converted>;
        fn link(words: &[u32], lengths: &[usize]) -> Result<Vec<u32>>;
        fn optimize(words: &[u32], flags: &[String]) -> Result<Vec<u32>>;
    }
}

pub use ffi::{Converted, VertexInput, convert_dxil, link, optimize};
