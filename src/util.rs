use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::{Instruction, Module, Operand},
    spirv::{Decoration, ExecutionModel, Op, StorageClass, Word},
};

pub fn op_names(m: &Module) -> HashMap<Word, String> {
    m.debug_names
        .iter()
        .filter(|n| n.class.opcode == Op::Name)
        .filter_map(|n| Some((n.operands[0].unwrap_id_ref(), string_operand(n, 1)?)))
        .collect()
}

pub fn string_operand(i: &Instruction, idx: usize) -> Option<String> {
    match i.operands.get(idx)? {
        Operand::LiteralString(s) => Some(s.clone()),
        _ => None,
    }
}

pub fn global_defs(m: &Module) -> HashMap<Word, Instruction> {
    m.types_global_values
        .iter()
        .filter_map(|i| Some((i.result_id?, i.clone())))
        .collect()
}

pub fn constant_values(m: &Module) -> HashMap<Word, u64> {
    m.types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Constant)
        .filter_map(|i| {
            let v = match i.operands.first()? {
                Operand::LiteralBit32(v) => *v as u64,
                Operand::LiteralBit64(v) => *v,
                _ => return None,
            };
            Some((i.result_id?, v))
        })
        .collect()
}

type DecorationTarget = (Word, Option<u32>);

pub struct Decorations(HashMap<DecorationTarget, Vec<(Decoration, Option<u32>)>>);

impl Decorations {
    pub fn new(m: &Module) -> Self {
        let mut map: HashMap<DecorationTarget, Vec<_>> = HashMap::new();
        for a in &m.annotations {
            let (key, rest) = match a.class.opcode {
                Op::Decorate => ((a.operands[0].unwrap_id_ref(), None), &a.operands[1..]),
                Op::MemberDecorate => {
                    let Operand::LiteralBit32(i) = a.operands[1] else {
                        continue;
                    };
                    ((a.operands[0].unwrap_id_ref(), Some(i)), &a.operands[2..])
                }
                _ => continue,
            };
            let Some(Operand::Decoration(d)) = rest.first() else {
                continue;
            };
            let lit = match rest.get(1) {
                Some(Operand::LiteralBit32(v)) => Some(*v),
                Some(Operand::BuiltIn(b)) => Some(*b as u32),
                _ => None,
            };
            map.entry(key).or_default().push((*d, lit));
        }
        Self(map)
    }

    pub fn get(&self, id: Word, member: Option<u32>, d: Decoration) -> Option<Option<u32>> {
        self.0
            .get(&(id, member))?
            .iter()
            .find(|(x, _)| *x == d)
            .map(|(_, v)| *v)
    }

    pub fn value(&self, id: Word, member: Option<u32>, d: Decoration) -> Option<u32> {
        self.get(id, member, d).flatten()
    }

    pub fn has(&self, id: Word, member: Option<u32>, d: Decoration) -> bool {
        self.get(id, member, d).is_some()
    }
}

#[derive(Debug, Clone)]
pub struct CBuffer {
    pub var: Word,
    pub ty: Word,
    pub name: String,
    pub set: Option<u32>,
    pub binding: Option<u32>,
}

pub struct NamedCBuffer {
    pub name: String,
    pub views: Vec<CBuffer>,
}

impl NamedCBuffer {
    pub fn first(&self) -> &CBuffer {
        &self.views[0]
    }
}

pub fn named_cbuffers(m: &Module) -> Result<Vec<NamedCBuffer>> {
    let mut out: Vec<NamedCBuffer> = Vec::new();
    for cb in cbuffers(m)? {
        match out.iter_mut().find(|n| n.name == cb.name) {
            Some(n) => n.views.push(cb),
            None => out.push(NamedCBuffer {
                name: cb.name.clone(),
                views: vec![cb],
            }),
        }
    }
    Ok(out)
}

pub fn variables<'a>(
    m: &'a Module,
    defs: &'a HashMap<Word, Instruction>,
    class: StorageClass,
) -> impl Iterator<Item = (Word, Word)> + 'a {
    m.types_global_values
        .iter()
        .filter(move |v| {
            v.class.opcode == Op::Variable && v.operands[0] == Operand::StorageClass(class)
        })
        .filter_map(|v| {
            Some((
                v.result_id?,
                defs.get(&v.result_type?)?.operands.get(1)?.id_ref_any()?,
            ))
        })
}

pub fn cbuffers(m: &Module) -> Result<Vec<CBuffer>> {
    let names = op_names(m);
    let defs = global_defs(m);
    let decos = Decorations::new(m);
    let mut out = Vec::new();
    for (var, ty) in variables(m, &defs, StorageClass::Uniform) {
        if defs[&ty].class.opcode != Op::TypeStruct {
            bail!("cbuffer is not a plain struct (arrays of CBVs are unsupported)");
        }
        let name = [var, ty]
            .iter()
            .filter_map(|id| names.get(id))
            .find(|n| !n.is_empty())
            .cloned()
            .context("cbuffer and its block type are both unnamed")?;
        out.push(CBuffer {
            var,
            ty,
            name,
            set: decos.value(var, None, Decoration::DescriptorSet),
            binding: decos.value(var, None, Decoration::Binding),
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    Srv,
    Uav,
    Sampler,
}

impl RegisterClass {
    pub fn letter(self) -> &'static str {
        match self {
            Self::Srv => "t",
            Self::Uav => "u",
            Self::Sampler => "s",
        }
    }
}

pub struct Resource {
    pub var: Word,
    pub ty: Word,
    pub name: String,
    pub register: u32,
    pub class: RegisterClass,
}

pub fn resources(m: &Module) -> Result<Vec<Resource>> {
    let names = op_names(m);
    let defs = global_defs(m);
    let decos = Decorations::new(m);
    let mut out = Vec::new();
    for (var, ty) in variables(m, &defs, StorageClass::UniformConstant) {
        let name = names.get(&var).cloned().unwrap_or_default();
        let class = match &defs[&ty] {
            t if t.class.opcode == Op::TypeSampler => RegisterClass::Sampler,
            t if t.class.opcode == Op::TypeImage && t.operands[5] == Operand::LiteralBit32(2) => {
                RegisterClass::Uav
            }
            t if t.class.opcode == Op::TypeImage => RegisterClass::Srv,
            _ => bail!("unsupported resource type (e.g. a resource array)"),
        };
        let space = decos
            .value(var, None, Decoration::DescriptorSet)
            .unwrap_or(0);
        ensure!(space == 0, "only register space 0 is supported");
        let register = decos
            .value(var, None, Decoration::Binding)
            .context("resource has no binding")?;
        out.push(Resource {
            var,
            ty,
            name,
            register,
            class,
        });
    }
    Ok(out)
}

pub struct TypeTables {
    pub defs: HashMap<Word, Instruction>,
    pub consts: HashMap<Word, u64>,
    pub decos: Decorations,
}

impl TypeTables {
    pub fn new(m: &Module) -> Self {
        Self {
            defs: global_defs(m),
            consts: constant_values(m),
            decos: Decorations::new(m),
        }
    }
}

pub fn same_type(a: &TypeTables, ta: Word, b: &TypeTables, tb: Word, layout: bool) -> bool {
    let (Some(x), Some(y)) = (a.defs.get(&ta), b.defs.get(&tb)) else {
        return false;
    };
    if x.class.opcode != y.class.opcode {
        return false;
    }
    let lit = |i: usize| (x.operands.get(i), y.operands.get(i));
    let same_lit = |i: usize| matches!(lit(i), (Some(p), Some(q)) if p == q);
    let same_sub = |i: usize| match (x.operands.get(i), y.operands.get(i)) {
        (Some(Operand::IdRef(p)), Some(Operand::IdRef(q))) => same_type(a, *p, b, *q, layout),
        _ => false,
    };
    let deco =
        |t: &TypeTables, ty: Word, member: Option<u32>, d: Decoration| t.decos.value(ty, member, d);
    let flag =
        |t: &TypeTables, ty: Word, member: Option<u32>, d: Decoration| t.decos.has(ty, member, d);
    match x.class.opcode {
        Op::TypeBool => true,
        Op::TypeFloat => same_lit(0),
        Op::TypeInt => same_lit(0) && same_lit(1),
        Op::TypeVector | Op::TypeMatrix => same_lit(1) && same_sub(0),
        Op::TypeArray => {
            let len = |t: &TypeTables, i: &Instruction| {
                t.consts.get(&i.operands[1].unwrap_id_ref()).copied()
            };
            len(a, x) == len(b, y)
                && (!layout
                    || deco(a, ta, None, Decoration::ArrayStride)
                        == deco(b, tb, None, Decoration::ArrayStride))
                && same_sub(0)
        }
        Op::TypeStruct => {
            x.operands.len() == y.operands.len()
                && (0..x.operands.len()).all(|i| {
                    let m = Some(i as u32);
                    same_sub(i)
                        && (!layout
                            || (deco(a, ta, m, Decoration::Offset)
                                == deco(b, tb, m, Decoration::Offset)
                                && flag(a, ta, m, Decoration::RowMajor)
                                    == flag(b, tb, m, Decoration::RowMajor)
                                && deco(a, ta, m, Decoration::MatrixStride)
                                    == deco(b, tb, m, Decoration::MatrixStride)))
                })
        }
        _ => true,
    }
}

pub fn execution_model(m: &Module) -> Result<ExecutionModel> {
    match m.entry_points.as_slice() {
        [ep] => Ok(ep.operands[0].unwrap_execution_model()),
        _ => bail!("expected exactly one entry point"),
    }
}
