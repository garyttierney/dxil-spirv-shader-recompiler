use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::{Builder, Function, Instruction, Operand},
    spirv::{Decoration, ExecutionModel, Op, StorageClass, Word},
};

use crate::{
    link::{Provider, emit, store},
    pipeline, util,
};

pub const VS_PROLOGUE: &str = "vs_prologue";
pub const VS_EPILOGUE: &str = "vs_epilogue";
pub const PS_PROLOGUE: &str = "ps_prologue";
pub const PS_EPILOGUE: &str = "ps_epilogue";

pub(crate) fn apply(
    b: &mut Builder,
    exports: &HashMap<String, Provider>,
    varyings: u32,
) -> Result<()> {
    let model = util::execution_model(b.module_ref())?;
    let (prologue, epilogue) = match model {
        ExecutionModel::Vertex => (VS_PROLOGUE, VS_EPILOGUE),
        ExecutionModel::Fragment => (PS_PROLOGUE, PS_EPILOGUE),
        _ => return Ok(()),
    };
    let (prologue, epilogue) = (exports.get(prologue), exports.get(epilogue));
    if prologue.is_none() && epilogue.is_none() {
        return Ok(());
    }
    let fragment = model == ExecutionModel::Fragment;
    let m = b.module_ref();
    let (inputs, outputs) = if fragment {
        (varyings, pipeline::next_location(m, StorageClass::Output)?)
    } else {
        (pipeline::next_location(m, StorageClass::Input)?, varyings)
    };
    let mut stage = Stage {
        defs: util::global_defs(m),
        fragment,
        interface: Vec::new(),
    };
    let entry = m
        .entry_points
        .first()
        .and_then(|ep| ep.operands.get(1))
        .and_then(Operand::id_ref_any)
        .context("no entry point")?;

    let prologue = prologue.map(|p| stage.prologue(b, p, inputs)).transpose()?;
    let epilogue = epilogue
        .map(|e| stage.epilogue_outputs(b, e, outputs))
        .transpose()?;

    let mut funcs = std::mem::take(&mut b.module_mut().functions);
    let r = funcs
        .iter_mut()
        .find(|f| f.def_id() == Some(entry))
        .context("no entry point function")
        .and_then(|f| stage.insert(b, f, prologue, epilogue));
    b.module_mut().functions = funcs;
    r?;

    let ep = &mut b.module_mut().entry_points[0];
    ep.operands
        .extend(stage.interface.into_iter().map(Operand::IdRef));
    Ok(())
}

struct Stage {
    defs: HashMap<Word, Instruction>,
    fragment: bool,
    interface: Vec<Word>,
}

impl Stage {
    fn insert(
        &self,
        b: &mut Builder,
        entry: &mut Function,
        prologue: Option<(Vec<Instruction>, Vec<Instruction>)>,
        epilogue: Option<(&Provider, Returned)>,
    ) -> Result<()> {
        if let Some((locals, code)) = prologue {
            let first = &mut entry
                .blocks
                .first_mut()
                .context("entry point without a body")?
                .instructions;
            let at = first
                .iter()
                .take_while(|i| i.class.opcode == Op::Variable)
                .count();
            first.splice(at..at, code);
            first.splice(0..0, locals);
        }
        if let Some((e, outputs)) = epilogue {
            for block in &mut entry.blocks {
                let Some(at) = block
                    .instructions
                    .iter()
                    .position(|i| i.class.opcode == Op::Return)
                else {
                    continue;
                };
                let code = self.epilogue(b, e, &outputs);
                block.instructions.splice(at..at, code);
            }
        }
        Ok(())
    }

    /// The prologue call and the function variables it passes arguments through.
    fn prologue(
        &mut self,
        b: &mut Builder,
        p: &Provider,
        mut location: u32,
    ) -> Result<(Vec<Instruction>, Vec<Instruction>)> {
        let (mut locals, mut code) = (Vec::new(), Vec::new());
        let mut args = vec![p.func];
        for &param in &p.params {
            let (ty, pointer) = match self.defs.get(&param) {
                Some(t) if t.class.opcode == Op::TypePointer => {
                    ensure!(
                        t.operands[0] == Operand::StorageClass(StorageClass::Function),
                        "hook parameters must be values or function pointers"
                    );
                    (t.operands[1].unwrap_id_ref(), Some(param))
                }
                _ => (param, None),
            };
            let (members, composite) = self.members(ty)?;
            let mut values = Vec::with_capacity(members.len());
            for m in members {
                let var = self.variable(b, StorageClass::Input, m, location)?;
                location += 1;
                values.push(emit(&mut code, b, Op::Load, m, &[var]));
            }
            let value = if composite {
                emit(&mut code, b, Op::CompositeConstruct, ty, &values)
            } else {
                values[0]
            };
            args.push(match pointer {
                None => value,
                Some(pointer) => {
                    let var = b.id();
                    locals.push(Instruction::new(
                        Op::Variable,
                        Some(pointer),
                        Some(var),
                        vec![Operand::StorageClass(StorageClass::Function)],
                    ));
                    code.push(store(var, value));
                    var
                }
            });
        }
        emit(&mut code, b, Op::FunctionCall, p.ret, &args);
        Ok((locals, code))
    }

    /// The epilogue and the output variables for each value it returns.
    fn epilogue_outputs<'p>(
        &mut self,
        b: &mut Builder,
        e: &'p Provider,
        mut location: u32,
    ) -> Result<(&'p Provider, Returned)> {
        ensure!(e.params.is_empty(), "epilogues take no parameters");
        let void = self
            .defs
            .get(&e.ret)
            .is_some_and(|t| t.class.opcode == Op::TypeVoid);
        if void {
            return Ok((e, Returned::Nothing));
        }
        let (members, composite) = self.members(e.ret)?;
        let mut vars = Vec::with_capacity(members.len());
        for m in members {
            vars.push((self.variable(b, StorageClass::Output, m, location)?, m));
            location += 1;
        }
        Ok((
            e,
            if composite {
                Returned::Members(vars)
            } else {
                Returned::Value(vars[0].0)
            },
        ))
    }

    fn epilogue(&self, b: &mut Builder, e: &Provider, outputs: &Returned) -> Vec<Instruction> {
        let mut code = Vec::new();
        let value = emit(&mut code, b, Op::FunctionCall, e.ret, &[e.func]);
        match outputs {
            Returned::Nothing => {}
            Returned::Value(var) => code.push(store(*var, value)),
            Returned::Members(vars) => {
                for (i, &(var, ty)) in vars.iter().enumerate() {
                    let id = b.id();
                    code.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(ty),
                        Some(id),
                        vec![Operand::IdRef(value), Operand::LiteralBit32(i as u32)],
                    ));
                    code.push(store(var, id));
                }
            }
        }
        code
    }

    /// The types taking one location each that make up a hook value, and whether it is a struct
    /// of them.
    fn members(&self, ty: Word) -> Result<(Vec<Word>, bool)> {
        let t = self
            .defs
            .get(&ty)
            .context("hook value type is not defined")?;
        let (members, composite) = match t.class.opcode {
            Op::TypeStruct => (
                t.operands.iter().map(Operand::unwrap_id_ref).collect(),
                true,
            ),
            _ => (vec![ty], false),
        };
        for &m in &members {
            ensure!(
                self.scalar(m)
                    .is_some_and(|s| s.operands[0] == Operand::LiteralBit32(32)),
                "hook values must be 32-bit scalars or vectors, or structs of them"
            );
        }
        Ok((members, composite))
    }

    /// The scalar type of a scalar or vector.
    fn scalar(&self, ty: Word) -> Option<&Instruction> {
        let t = self.defs.get(&ty)?;
        let s = match t.class.opcode {
            Op::TypeVector => self.defs.get(&t.operands[0].unwrap_id_ref())?,
            _ => t,
        };
        matches!(s.class.opcode, Op::TypeInt | Op::TypeFloat).then_some(s)
    }

    fn variable(
        &mut self,
        b: &mut Builder,
        class: StorageClass,
        ty: Word,
        location: u32,
    ) -> Result<Word> {
        let pointer = b.type_pointer(None, class, ty);
        let var = b.variable(pointer, None, class, None);
        b.decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
        let integer = match self.scalar(ty) {
            Some(s) => s.class.opcode == Op::TypeInt,
            None => bail!("hook value type is not a scalar or vector"),
        };
        if self.fragment && class == StorageClass::Input && integer {
            b.decorate(var, Decoration::Flat, []);
        }
        self.interface.push(var);
        Ok(var)
    }
}

enum Returned {
    Nothing,
    Value(Word),
    /// Each member's output variable and type.
    Members(Vec<(Word, Word)>),
}
