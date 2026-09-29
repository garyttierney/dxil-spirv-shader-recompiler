use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rspirv::{
    dr::{Module, Operand},
    spirv::{Decoration, Op, StorageClass, Word},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct DxilSpirv;

/// A converted shader.
#[derive(Debug, Clone)]
pub struct Converted {
    pub spirv: Vec<u8>,
    /// A vertex shader's inputs, empty for other stages
    pub vertex_inputs: Vec<VertexInput>,
}

/// A vertex shader input: its Direct3D semantic, where it's read from and what it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexInput {
    /// e.g. `TEXCOORD`.
    pub semantic: String,
    pub semantic_index: u32,
    pub location: u32,
    pub component_type: ComponentType,
    pub components: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    Float,
    Sint,
    Uint,
}

impl DxilSpirv {
    pub fn new() -> Self {
        Self
    }

    /// Converts a shader in a DXBC container.
    pub fn convert(&self, dxbc: &[u8]) -> Result<Converted> {
        let converted = dxil_spirv_shader_recompiler_sys::convert_dxil(dxbc)?;
        let vertex_inputs = if converted.vertex_inputs.is_empty() {
            Vec::new()
        } else {
            let module = rspirv::dr::load_bytes(&converted.spirv)
                .map_err(|err| anyhow!("reading the converted SPIR-V: {err:?}"))?;
            let types = input_types(&module);
            converted
                .vertex_inputs
                .into_iter()
                .map(|input| {
                    let &(component_type, components) = types
                        .get(&input.location)
                        .ok_or_else(|| anyhow!("no input at location {}", input.location))?;
                    Ok(VertexInput {
                        semantic: input.semantic,
                        semantic_index: input.semantic_index,
                        location: input.location,
                        component_type,
                        components,
                    })
                })
                .collect::<Result<_>>()?
        };
        Ok(Converted {
            spirv: converted.spirv,
            vertex_inputs,
        })
    }
}

fn input_types(module: &Module) -> HashMap<u32, (ComponentType, u32)> {
    let defs: HashMap<Word, _> = module
        .types_global_values
        .iter()
        .filter_map(|inst| Some((inst.result_id?, inst)))
        .collect();
    let locations: HashMap<Word, u32> = module
        .annotations
        .iter()
        .filter_map(|inst| match inst.operands.as_slice() {
            [
                Operand::IdRef(target),
                Operand::Decoration(Decoration::Location),
                Operand::LiteralBit32(location),
            ] => Some((*target, *location)),
            _ => None,
        })
        .collect();
    let scalar = |id: Word| {
        let def = defs.get(&id)?;
        match def.class.opcode {
            Op::TypeFloat => Some(ComponentType::Float),
            Op::TypeInt if def.operands.get(1) == Some(&Operand::LiteralBit32(1)) => {
                Some(ComponentType::Sint)
            }
            Op::TypeInt => Some(ComponentType::Uint),
            _ => None,
        }
    };

    module
        .types_global_values
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Input))
        })
        .filter_map(|variable| {
            let location = *locations.get(&variable.result_id?)?;
            // The variable's pointer type, then what it points to.
            let pointer = defs.get(&variable.result_type?)?;
            let pointee = pointer.operands.get(1)?.id_ref_any()?;
            let def = defs.get(&pointee)?;
            let (component_type, components) = match def.class.opcode {
                Op::TypeVector => (
                    scalar(def.operands.first()?.id_ref_any()?)?,
                    def.operands.get(1)?.unwrap_literal_bit32(),
                ),
                _ => (scalar(pointee)?, 1),
            };
            Some((location, (component_type, components)))
        })
        .collect()
}
