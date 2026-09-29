use anyhow::{Result, anyhow};
use rspirv::{
    binary::Assemble,
    dr::{Module, load_words},
};

use crate::link::merge_builtins;

/// The `spirv-opt` passes run on the linked module: HLSL legalization, then capability trimming.
pub const LEGALIZATION: &[&str] = &["--legalize-hlsl", "--trim-capabilities"];

/// Joins `shader`, as [`link::link`](crate::link::link) returns it, with `stub`, resolving the
/// shader's imports of the stub's providers.
pub fn join(shader: Module, stub: &Module) -> Result<Module> {
    let (shader, stub) = (shader.assemble(), stub.assemble());
    let lengths = [shader.len(), stub.len()];
    let linked = dxil_spirv_shader_recompiler_sys::link(&[shader, stub].concat(), &lengths)?;
    let mut linked = load(linked)?;
    // The stub and the shader may each declare the same builtin input.
    merge_builtins(&mut linked);

    let flags: Vec<String> = LEGALIZATION.iter().map(|flag| flag.to_string()).collect();
    load(dxil_spirv_shader_recompiler_sys::optimize(
        &linked.assemble(),
        &flags,
    )?)
}

fn load(words: Vec<u32>) -> Result<Module> {
    load_words(words).map_err(|err| anyhow!("reading SPIR-V: {err:?}"))
}
