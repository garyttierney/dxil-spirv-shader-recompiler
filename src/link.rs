use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::{Block, Builder, Function, Instruction, Module, Operand},
    spirv::{
        AddressingModel, Capability, Decoration, Dim, ImageFormat, LinkageType, Op, StorageClass,
        Word,
    },
};

use crate::{
    hooks,
    layout::{Leaf, MatrixLayout, Types},
    util::{self, Decorations, NamedCBuffer, RegisterClass, Resource},
};

pub const DEFAULT_PREFIX: &str = "provide_";

pub fn link(
    mut shader: Module,
    lib: &Module,
    prefix: &str,
    matrices: MatrixLayout,
    varyings: u32,
) -> Result<Module> {
    match_addressing_model(&mut shader, lib)?;
    let lib_version = lib
        .header
        .as_ref()
        .context("library has no header")?
        .version();
    let header = shader.header.as_mut().context("shader has no header")?;
    let version = header.version().max(lib_version);
    header.set_version(version.0, version.1);

    let cbuffers = util::named_cbuffers(&shader)?;
    let resources = util::resources(&shader)?;
    let decos = Decorations::new(&shader);
    let bound: Vec<Word> = shader
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| i.result_id)
        .filter(|&v| decos.value(v, None, Decoration::Binding).is_some())
        .collect();
    let mut b = Builder::new_from_module(shader);
    let exports = import_library(&mut b, lib)?;
    let roles = Roles::assign(&exports, prefix, &cbuffers, &resources)?;
    let defs = util::global_defs(b.module_ref());

    for cb in &cbuffers {
        let p = roles.cbuffer(&cb.name);
        if p.whole.is_none() && p.elems.is_empty() {
            continue;
        }
        for view in &cb.views {
            replace_cbuffer(&mut b, view, &p, matrices).context("replacing a cbuffer")?;
        }
    }
    for r in &resources {
        let buffer = defs[&r.ty].operands.get(1) == Some(&Operand::Dim(Dim::DimBuffer));
        match roles.named.get(&Role::Resource(&r.name)) {
            Some(p) if buffer => replace_structured(&mut b, r, p, matrices)
                .context("replacing a structured buffer")?,
            Some(p) => {
                ensure!(
                    same_resource_type(&defs[&p.ret], &defs[&r.ty]),
                    "a resource's provider returns another type of resource"
                );
                replace_resource(&mut b, r, p, false).context("replacing a resource")?;
            }
            None => {
                let p = register_provider(&defs, &roles.registers, r)?;
                replace_resource(&mut b, r, p, true).context("replacing a register")?;
            }
        }
    }

    hooks::apply(&mut b, &exports, varyings).context("calling the stage hooks")?;

    let mut shader = b.module();
    remove_unreachable_functions(&mut shader);
    remove_unreferenced_globals(&mut shader);
    ensure!(
        !shader
            .types_global_values
            .iter()
            .any(|i| i.result_id.is_some_and(|id| bound.contains(&id))),
        "a bound resource has no provider"
    );
    Ok(shader)
}

pub fn merge_builtins(m: &mut Module) {
    let mut first: HashMap<Operand, Word> = HashMap::new();
    let mut merged: HashMap<Word, Word> = HashMap::new();
    let inputs: HashSet<Word> = m
        .types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(StorageClass::Input))
        })
        .filter_map(|i| i.result_id)
        .collect();
    for a in &m.annotations {
        let (
            Op::Decorate,
            [
                Operand::IdRef(var),
                Operand::Decoration(Decoration::BuiltIn),
                builtin,
            ],
        ) = (a.class.opcode, a.operands.as_slice())
        else {
            continue;
        };
        if !inputs.contains(var) {
            continue;
        }
        match first.get(builtin) {
            Some(&kept) => {
                merged.insert(*var, kept);
            }
            None => {
                first.insert(builtin.clone(), *var);
            }
        }
    }
    if merged.is_empty() {
        return;
    }
    let gone: HashSet<Word> = merged.keys().copied().collect();
    m.types_global_values
        .retain(|i| !i.result_id.is_some_and(|id| gone.contains(&id)));
    remove_targeting(m, &gone);
    for ep in &mut m.entry_points {
        ep.operands
            .retain(|o| !matches!(o, Operand::IdRef(id) if gone.contains(id)));
    }
    for i in m.functions.iter_mut().flat_map(|f| f.all_inst_iter_mut()) {
        remap_refs(i, |id| merged.get(&id).copied().unwrap_or(id));
    }
}

pub(crate) fn match_addressing_model(shader: &mut Module, lib: &Module) -> Result<()> {
    let (Some(s), Some(l)) = (&mut shader.memory_model, &lib.memory_model) else {
        bail!("shader or library has no OpMemoryModel");
    };
    if s.operands == l.operands {
        return Ok(());
    }
    let psb = Operand::AddressingModel(AddressingModel::PhysicalStorageBuffer64);
    let logical = Operand::AddressingModel(AddressingModel::Logical);
    ensure!(
        s.operands[1] == l.operands[1] && s.operands[0] == logical && l.operands[0] == psb,
        "shader and library memory models are incompatible"
    );
    s.operands[0] = psb;
    let cap = Operand::Capability(Capability::PhysicalStorageBufferAddresses);
    if !shader.capabilities.iter().any(|c| c.operands[0] == cap) {
        shader
            .capabilities
            .push(Instruction::new(Op::Capability, None, None, vec![cap]));
    }
    Ok(())
}

/// The module's 32-bit `OpConstant` of type `ty` with value `bits`, added if there is none
/// (`Builder::constant_bit32` always adds one).
fn constant(b: &mut Builder, ty: Word, bits: u32) -> Word {
    let value = [Operand::LiteralBit32(bits)];
    let existing = b.module_ref().types_global_values.iter().find(|i| {
        i.class.opcode == Op::Constant && i.result_type == Some(ty) && i.operands == value
    });
    match existing.and_then(|i| i.result_id) {
        Some(id) => id,
        None => b.constant_bit32(ty, bits),
    }
}

pub(crate) fn emit(
    out: &mut Vec<Instruction>,
    b: &mut Builder,
    op: Op,
    ty: Word,
    operands: &[Word],
) -> Word {
    let id = b.id();
    out.push(Instruction::new(
        op,
        Some(ty),
        Some(id),
        operands.iter().copied().map(Operand::IdRef).collect(),
    ));
    id
}

pub(crate) fn store(pointer: Word, value: Word) -> Instruction {
    Instruction::new(
        Op::Store,
        None,
        None,
        vec![Operand::IdRef(pointer), Operand::IdRef(value)],
    )
}

pub(crate) fn id_refs(inst: &Instruction) -> impl Iterator<Item = Word> + '_ {
    inst.operands.iter().filter_map(Operand::id_ref_any)
}

struct Tables {
    defs: HashMap<Word, Instruction>,
    consts: HashMap<Word, u64>,
    decos: Decorations,
    masks: HashSet<Word>,
}

impl Tables {
    fn new(m: &Module) -> Self {
        let consts = util::constant_values(m);
        Self {
            defs: util::global_defs(m),
            masks: address_masks(m, &consts),
            consts,
            decos: Decorations::new(m),
        }
    }

    fn types(&self) -> Types<'_> {
        Types {
            defs: &self.defs,
            consts: &self.consts,
            decos: &self.decos,
        }
    }

    fn index_exprs(&self, f: &Function) -> IndexExprs<'_> {
        let local = f
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| Some((i.result_id?, i.clone())))
            .collect();
        IndexExprs {
            local,
            consts: &self.consts,
            masks: &self.masks,
        }
    }
}

fn value_types(f: &Function, globals: &[Instruction]) -> HashMap<Word, Word> {
    let body = f.blocks.iter().flat_map(|b| &b.instructions);
    f.parameters
        .iter()
        .chain(body)
        .chain(globals)
        .filter_map(|i| Some((i.result_id?, i.result_type?)))
        .collect()
}

fn rewrite_functions(
    b: &mut Builder,
    mut each: impl FnMut(&mut Builder, &mut Function, &mut Locals) -> Result<()>,
) -> Result<()> {
    let mut funcs = std::mem::take(&mut b.module_mut().functions);
    let r = funcs.iter_mut().try_for_each(|f| {
        let mut locals = Locals::default();
        each(b, f, &mut locals)?;
        locals.prepend_to(f);
        Ok(())
    });
    b.module_mut().functions = funcs;
    r
}

fn rewrite_block(
    b: &mut Block,
    mut each: impl FnMut(Instruction, &mut Vec<Instruction>) -> Result<()>,
) -> Result<()> {
    let mut out = Vec::with_capacity(b.instructions.len());
    for inst in std::mem::take(&mut b.instructions) {
        each(inst, &mut out)?;
    }
    b.instructions = out;
    Ok(())
}

#[derive(Default)]
struct Locals {
    vars: Vec<Instruction>,
    args: HashMap<Word, Word>,
}

impl Locals {
    fn declare(&mut self, var: Word, pointer_ty: Word) {
        self.vars.push(Instruction::new(
            Op::Variable,
            Some(pointer_ty),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
    }

    fn spill(&mut self, b: &mut Builder, ty: Word) -> Word {
        let var = b.id();
        let pointer_ty = b.type_pointer(None, StorageClass::Function, ty);
        self.declare(var, pointer_ty);
        var
    }

    fn index_arg(
        &mut self,
        b: &mut Builder,
        out: &mut Vec<Instruction>,
        func: Word,
        param: IndexParam,
        mut index: Word,
        index_ty: Word,
    ) -> Word {
        if index_ty != param.ty {
            index = emit(out, b, Op::Bitcast, param.ty, &[index]);
        }
        let Some(pointer_ty) = param.pointer else {
            return index;
        };
        let var = match self.args.get(&func) {
            Some(&v) => v,
            None => {
                let v = b.id();
                self.declare(v, pointer_ty);
                self.args.insert(func, v);
                v
            }
        };
        out.push(store(var, index));
        var
    }

    fn prepend_to(self, f: &mut Function) {
        if let Some(b) = f.blocks.first_mut() {
            b.instructions.splice(0..0, self.vars);
        }
    }
}

pub(crate) struct Provider {
    pub(crate) func: Word,
    pub(crate) ret: Word,
    pub(crate) params: Vec<Word>,
}

/// Declares each of the library's exported functions as an import, keyed by its linkage name.
fn import_library(b: &mut Builder, lib: &Module) -> Result<HashMap<String, Provider>> {
    let mut map: HashMap<Word, Word> = HashMap::new();
    fn translate(map: &mut HashMap<Word, Word>, b: &mut Builder, id: Word) -> Word {
        *map.entry(id).or_insert_with(|| b.id())
    }

    let lib_decos = decorations_by_target(lib);
    let mut table = TypeTable::new(b.module_ref());
    let mut copied = HashSet::new();
    for inst in &lib.types_global_values {
        let Some(old) = inst.result_id else { continue };
        if inst.class.opcode == Op::Variable {
            continue;
        }
        let mut inst = inst.clone();
        remap_refs(&mut inst, |id| translate(&mut map, b, id));
        let decos = lib_decos.get(&old).cloned().unwrap_or_default();
        if let Some(existing) = table.find(&inst, &decos) {
            map.insert(old, existing);
            continue;
        }
        let new = translate(&mut map, b, old);
        inst.result_id = Some(new);
        table.insert(&inst, decos);
        copied.insert(old);
        b.module_mut().types_global_values.push(inst);
    }

    let mut providers = HashMap::new();
    for a in &lib.annotations {
        let Some((old_f, name, ty)) = linkage_of(a) else {
            continue;
        };
        ensure!(ty == LinkageType::Export, "library has imports");
        let f = lib
            .functions
            .iter()
            .find(|f| f.def_id() == Some(old_f))
            .context("library export is not a function")?;
        let mut def = f.def.clone().context("function without OpFunction")?;
        remap_refs(&mut def, |id| translate(&mut map, b, id));
        let func = b.id();
        def.result_id = Some(func);
        let ret = def.result_type.context("OpFunction without a type")?;
        let mut parameters = f.parameters.clone();
        for p in &mut parameters {
            remap_refs(p, |id| translate(&mut map, b, id));
            p.result_id = Some(b.id());
        }
        let params = parameters
            .iter()
            .map(|p| p.result_type.context("parameter without a type"))
            .collect::<Result<_>>()?;
        b.module_mut().functions.push(Function {
            def: Some(def),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters,
            blocks: vec![],
        });
        b.decorate(
            func,
            Decoration::LinkageAttributes,
            [
                Operand::LiteralString(name.clone()),
                Operand::LinkageType(LinkageType::Import),
            ],
        );
        b.name(func, name.as_str());
        providers.insert(name, Provider { func, ret, params });
    }

    for a in lib.annotations.iter().chain(&lib.debug_names) {
        if let Some(Operand::IdRef(t)) = a.operands.first()
            && copied.contains(t)
            && id_refs(a).all(|id| map.contains_key(&id))
        {
            let mut a = a.clone();
            remap_refs(&mut a, |id| map[&id]);
            let m = b.module_mut();
            if matches!(a.class.opcode, Op::Name | Op::MemberName) {
                m.debug_names.push(a);
            } else {
                m.annotations.push(a);
            }
        }
    }

    let linkage = Operand::Capability(Capability::Linkage);
    if !b
        .module_ref()
        .capabilities
        .iter()
        .any(|c| c.operands[0] == linkage)
    {
        b.capability(Capability::Linkage);
    }
    Ok(providers)
}

pub(crate) type DecorationSet = Vec<(Op, Vec<Operand>)>;

pub(crate) fn decorations_by_target(m: &Module) -> HashMap<Word, DecorationSet> {
    let mut out: HashMap<Word, DecorationSet> = HashMap::new();
    for a in &m.annotations {
        if let Some(Operand::IdRef(t)) = a.operands.first() {
            out.entry(*t)
                .or_default()
                .push((a.class.opcode, a.operands[1..].to_vec()));
        }
    }
    out
}

fn same_decorations(a: &DecorationSet, b: &DecorationSet) -> bool {
    a.len() == b.len() && a.iter().all(|d| b.contains(d))
}

fn interchangeable(op: Op) -> bool {
    use Op::*;
    matches!(
        op,
        TypeVoid
            | TypeBool
            | TypeInt
            | TypeFloat
            | TypeVector
            | TypeMatrix
            | TypeImage
            | TypeSampler
            | TypeSampledImage
            | TypeArray
            | TypeRuntimeArray
            | TypeStruct
            | TypePointer
            | TypeFunction
            | Constant
            | ConstantTrue
            | ConstantFalse
            | ConstantComposite
            | ConstantNull
            | Undef
    )
}

pub(crate) struct TypeTable(HashMap<Op, Vec<(Instruction, DecorationSet)>>);

impl TypeTable {
    pub(crate) fn new(m: &Module) -> Self {
        let mut decos = decorations_by_target(m);
        let mut t = Self(HashMap::new());
        for i in &m.types_global_values {
            if let Some(id) = i.result_id {
                t.insert(i, decos.remove(&id).unwrap_or_default());
            }
        }
        t
    }

    pub(crate) fn insert(&mut self, i: &Instruction, decos: DecorationSet) {
        if interchangeable(i.class.opcode) {
            self.0
                .entry(i.class.opcode)
                .or_default()
                .push((i.clone(), decos));
        }
    }

    pub(crate) fn find(&self, i: &Instruction, decos: &DecorationSet) -> Option<Word> {
        if !interchangeable(i.class.opcode) {
            return None;
        }
        self.0.get(&i.class.opcode)?.iter().find_map(|(e, d)| {
            (e.result_type == i.result_type
                && e.operands == i.operands
                && same_decorations(d, decos))
            .then_some(e.result_id)
            .flatten()
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Role<'a> {
    Whole(&'a str),

    Tail(&'a str),
    Element(&'a str, u32),
    /// A structured buffer (by element) or a texture or sampler, by name.
    Resource(&'a str),
    Register(RegisterClass),
}

struct Roles<'a> {
    named: HashMap<Role<'a>, &'a Provider>,
    registers: Vec<(RegisterClass, &'a Provider)>,
}

impl<'a> Roles<'a> {
    fn assign(
        exports: &'a HashMap<String, Provider>,
        prefix: &str,
        cbuffers: &'a [NamedCBuffer],
        resources: &'a [Resource],
    ) -> Result<Self> {
        let mut roles = Self {
            named: HashMap::new(),
            registers: vec![],
        };
        for (name, p) in exports {
            let Some(key) = name.strip_prefix(prefix) else {
                continue;
            };
            let mut candidates: Vec<Role> = Vec::new();
            for cb in cbuffers {
                let x = cb.name.as_str();
                let rest = key.strip_prefix(x);
                if rest == Some("") {
                    candidates.push(Role::Whole(x));
                } else if rest == Some("_tail") {
                    candidates.push(Role::Tail(x));
                } else if let Some(n) = rest
                    .and_then(|r| r.strip_prefix("_m"))
                    .and_then(member_index)
                {
                    candidates.push(Role::Element(x, n));
                }
            }
            for r in resources
                .iter()
                .filter(|r| !r.name.is_empty() && r.name == *key)
            {
                let role = Role::Resource(&r.name);
                if !candidates.contains(&role) {
                    candidates.push(role);
                }
            }
            if candidates.is_empty() {
                candidates.extend(register_class(key).map(Role::Register));
            }
            match candidates.as_slice() {
                [] => {}
                [Role::Register(class)] => roles.registers.push((*class, p)),
                [role] => {
                    roles.named.insert(*role, p);
                }
                _ => bail!("a provider's name fits more than one resource"),
            }
        }
        Ok(roles)
    }

    fn cbuffer(&self, name: &'a str) -> CBufferProviders<'a> {
        CBufferProviders {
            whole: self.named.get(&Role::Whole(name)).copied(),
            tail: self.named.get(&Role::Tail(name)).copied(),
            elems: self
                .named
                .iter()
                .filter_map(|(role, p)| match role {
                    Role::Element(x, n) if *x == name => Some((*n, *p)),
                    _ => None,
                })
                .collect(),
        }
    }
}

fn member_index(digits: &str) -> Option<u32> {
    let canonical = digits == "0" || !digits.starts_with('0');
    digits
        .bytes()
        .all(|b| b.is_ascii_digit())
        .then(|| digits.parse().ok())
        .flatten()
        .filter(|_| canonical)
}

fn register_class(key: &str) -> Option<RegisterClass> {
    [
        RegisterClass::Srv,
        RegisterClass::Uav,
        RegisterClass::Sampler,
    ]
    .into_iter()
    .find(|c| {
        key.strip_prefix(c.letter())
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
    })
}

#[derive(Debug, Clone, Copy)]
struct IndexParam {
    ty: Word,
    pointer: Option<Word>,
}

impl IndexParam {
    /// The provider's index parameter, if it takes one.
    fn of(defs: &HashMap<Word, Instruction>, p: &Provider) -> Result<Option<Self>> {
        Ok(match p.params.as_slice() {
            [] => None,
            &[ty] => {
                let t = defs
                    .get(&ty)
                    .context("provider parameter type is not defined")?;
                let param = match t.class.opcode {
                    Op::TypePointer => Self {
                        ty: t.operands[1].unwrap_id_ref(),
                        pointer: Some(ty),
                    },
                    _ => Self { ty, pointer: None },
                };
                ensure!(
                    defs.get(&param.ty)
                        .is_some_and(|t| t.class.opcode == Op::TypeInt),
                    "provider index must be an integer"
                );
                Some(param)
            }
            _ => bail!("provider has too many parameters"),
        })
    }
}

fn register_provider<'a>(
    defs: &HashMap<Word, Instruction>,
    providers: &[(RegisterClass, &'a Provider)],
    r: &Resource,
) -> Result<&'a Provider> {
    let class = r.class;
    let matching: Vec<&Provider> = providers
        .iter()
        .filter(|(c, p)| {
            *c == class
                && defs
                    .get(&p.ret)
                    .is_some_and(|t| same_resource_type(t, &defs[&r.ty]))
        })
        .map(|(_, p)| *p)
        .collect();
    match matching.as_slice() {
        [one] => Ok(*one),
        [] => bail!("register has no provider of a matching type"),
        _ => bail!("register matches several providers"),
    }
}

fn same_resource_type(a: &Instruction, b: &Instruction) -> bool {
    match (a.class.opcode, b.class.opcode) {
        (Op::TypeSampler, Op::TypeSampler) => true,
        (Op::TypeImage, Op::TypeImage) => {
            let unknown = Operand::ImageFormat(ImageFormat::Unknown);
            let (x, y) = (&a.operands, &b.operands);
            x.len() == y.len()
                && (0..x.len()).all(|i| match i {
                    2 => true,
                    6 => x[i] == y[i] || x[i] == unknown || y[i] == unknown,
                    _ => x[i] == y[i],
                })
        }
        _ => false,
    }
}

fn replace_structured(
    b: &mut Builder,
    r: &Resource,
    p: &Provider,
    matrices: MatrixLayout,
) -> Result<()> {
    let t = Tables::new(b.module_ref());
    let img = &t.defs[&r.ty];
    ensure!(
        img.class.opcode == Op::TypeImage && img.operands[1] == Operand::Dim(Dim::DimBuffer),
        "structured buffer provider is named after a resource that isn't a buffer"
    );
    let word_ty = img.operands[0].unwrap_id_ref();
    let index =
        IndexParam::of(&t.defs, p)?.context("structured buffer provider must take a uint index")?;
    let (leaves, size) = t
        .types()
        .structured_leaves(p.ret, matrices)
        .context("laying out a structured buffer provider's return type")?;
    ensure!(
        size.is_multiple_of(4),
        "structured buffer provider's return type is not whole words"
    );
    let k = (size / 4) as u64;
    let (zero, one) = (constant(b, word_ty, 0), constant(b, word_ty, 1));

    rewrite_functions(b, |b, f, locals| {
        let exprs = t.index_exprs(f);
        let value_types = value_types(f, &b.module_ref().types_global_values);
        let mut loaded: HashSet<Word> = HashSet::new();
        for block in &mut f.blocks {
            let mut calls: HashMap<Word, Word> = HashMap::new();
            rewrite_block(block, |inst, out| {
                if inst.class.opcode == Op::Load && inst.operands[0] == Operand::IdRef(r.var) {
                    loaded.insert(inst.result_id.context("OpLoad without a result")?);
                    return Ok(());
                }
                if !id_refs(&inst).any(|id| loaded.contains(&id)) {
                    out.push(inst);
                    return Ok(());
                }
                ensure!(
                    inst.class.opcode == Op::ImageFetch
                        && inst.operands.len() == 2
                        && loaded.contains(&inst.operands[0].unwrap_id_ref()),
                    "structured buffer is used other than by element reads"
                );
                let addr = inst.operands[1].unwrap_id_ref();
                let s = exprs
                    .split(addr, k)
                    .context("structured buffer word index is not `x * words + w`")?;
                ensure!(
                    s.step == k,
                    "structured buffer element size differs from its provider's return type"
                );
                ensure!(
                    s.dynamic.is_empty(),
                    "structured buffer is indexed dynamically within an element"
                );
                ensure!(s.first < k, "structured buffer read is past one element");
                let call = match calls.get(&s.x) {
                    Some(&c) => c,
                    None => {
                        let x_ty = *value_types
                            .get(&s.x)
                            .context("element index without a type")?;
                        let arg = locals.index_arg(b, out, p.func, index, s.x, x_ty);
                        let c = emit(out, b, Op::FunctionCall, p.ret, &[p.func, arg]);
                        calls.insert(s.x, c);
                        c
                    }
                };
                let off = s.first as u32 * 4;
                let leaf = leaves
                    .get(&off)
                    .context("structured buffer read not covered by its provider")?;
                let v = extract(call, leaf, out, b);
                let word = convert(b, &t.defs, out, v, leaf.ty, word_ty)
                    .context("converting a structured buffer word")?;
                let parts = [word, zero, zero, one].map(Operand::IdRef).to_vec();
                out.push(Instruction::new(
                    Op::CompositeConstruct,
                    inst.result_type,
                    inst.result_id,
                    parts,
                ));
                Ok(())
            })?;
        }
        Ok(())
    })
}

/// Replaces loads of a texture or sampler with calls to its provider, passing the register if
/// `by_register`.
fn replace_resource(b: &mut Builder, r: &Resource, p: &Provider, by_register: bool) -> Result<()> {
    let defs = util::global_defs(b.module_ref());
    let param = IndexParam::of(&defs, p)?;
    ensure!(
        param.is_some() == by_register,
        "register providers take a uint register and named resource providers nothing"
    );
    let reg = param.map(|param| (param, constant(b, param.ty, r.register)));

    rewrite_functions(b, |b, f, locals| {
        let mut retyped: HashMap<Word, Word> = HashMap::new();
        for block in &mut f.blocks {
            rewrite_block(block, |inst, out| {
                if !inst.operands.contains(&Operand::IdRef(r.var)) {
                    out.push(inst);
                    return Ok(());
                }
                ensure!(
                    inst.class.opcode == Op::Load,
                    "a texture or sampler is used other than by loads"
                );
                let mut call = vec![Operand::IdRef(p.func)];
                if let Some((param, reg)) = reg {
                    let arg = locals.index_arg(b, out, p.func, param, reg, param.ty);
                    call.push(Operand::IdRef(arg));
                }
                let result = inst.result_id.context("OpLoad without a result")?;
                out.push(Instruction::new(
                    Op::FunctionCall,
                    Some(p.ret),
                    Some(result),
                    call,
                ));
                if inst.result_type != Some(p.ret) {
                    retyped.insert(result, p.ret);
                }
                Ok(())
            })?;
        }
        retype_derived_images(b, f, retyped)
    })
}

fn retype_derived_images(
    b: &mut Builder,
    f: &mut Function,
    mut retyped: HashMap<Word, Word>,
) -> Result<()> {
    let mut changed = !retyped.is_empty();
    while changed {
        changed = false;
        for inst in f.blocks.iter_mut().flat_map(|b| &mut b.instructions) {
            let Some(result) = inst.result_id else {
                continue;
            };
            if retyped.contains_key(&result) {
                continue;
            }
            let Some(new) = id_refs(inst).find_map(|id| retyped.get(&id).copied()) else {
                continue;
            };
            let ty = match inst.class.opcode {
                Op::SampledImage => b.type_sampled_image(new),
                Op::Image => {
                    let sampled = b
                        .module_ref()
                        .types_global_values
                        .iter()
                        .find(|t| t.result_id == Some(new))
                        .context("sampled image type")?;
                    sampled.operands[0].unwrap_id_ref()
                }
                Op::CopyObject | Op::Phi | Op::Select => new,
                _ => continue,
            };
            inst.result_type = Some(ty);
            retyped.insert(result, ty);
            changed = true;
        }
    }
    Ok(())
}

struct CBufferProviders<'a> {
    whole: Option<&'a Provider>,
    tail: Option<&'a Provider>,
    elems: HashMap<u32, &'a Provider>,
}

struct Source {
    func: Word,
    ret: Word,
    leaves: BTreeMap<u32, Leaf>,
}

impl Source {
    fn new(types: &Types, p: &Provider, matrices: MatrixLayout) -> Result<Self> {
        let leaves = types
            .packed_leaves(p.ret, matrices)
            .context("laying out a provider's return type")?;
        Ok(Self {
            func: p.func,
            ret: p.ret,
            leaves,
        })
    }
}

struct LegacyScalar {
    offset: u32,
    ty: Word,
}

type LegacyLeaves = HashMap<Vec<u32>, LegacyScalar>;

fn by_path(leaves: BTreeMap<u32, Leaf>) -> LegacyLeaves {
    leaves
        .into_iter()
        .map(|(offset, l)| (l.path, LegacyScalar { offset, ty: l.ty }))
        .collect()
}

struct Whole {
    src: Source,
    legacy: LegacyLeaves,
    /// The provider's call in each function reading the cbuffer, by function.
    calls: HashMap<Word, WholeCall>,
}

/// A whole cbuffer provider's call at the top of a function reading the cbuffer.
struct WholeCall {
    call: Word,
    /// The call's result spilled to a variable, for dynamically indexed reads.
    var: Option<Word>,
}

struct Elem {
    src: Source,
    index: IndexParam,
    legacy: LegacyLeaves,
    stride: u32,
    group: Option<u32>,
}

struct Tail {
    func: Word,
    ret: Word,
    index: IndexParam,
    base: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    Whole,
    Element(u32),
}

#[derive(Debug, Clone, Copy)]
struct DynamicIndex {
    index: Word,
    ty: Word,
    stride: u32,
}

struct Chain {
    value: Word,
    region: Region,
    path: Vec<u32>,
    base: u32,
    dynamic: Option<DynamicIndex>,
}

struct CBufferRewrite<'a> {
    cb: &'a util::CBuffer,
    tables: &'a Tables,
    types: Types<'a>,
    matrices: MatrixLayout,
    whole: Option<Whole>,
    tail: Option<Tail>,
    elems: HashMap<u32, Elem>,
    chains: HashMap<Word, Chain>,
}

struct FunctionRewrite<'a> {
    exprs: IndexExprs<'a>,
    value_types: HashMap<Word, Word>,
    used: HashMap<Word, HashSet<u32>>,
    locals: &'a mut Locals,
}

impl FunctionRewrite<'_> {
    fn value_type(&self, id: Word) -> Result<Word> {
        self.value_types
            .get(&id)
            .copied()
            .context("value has no type")
    }
}

fn member_array(t: &Tables, block: Word, n: u32) -> Result<(Word, u32)> {
    let member = t.defs[&block]
        .operands
        .get(n as usize)
        .context("cbuffer has no such member")?
        .unwrap_id_ref();
    ensure!(
        t.defs
            .get(&member)
            .is_some_and(|a| a.class.opcode == Op::TypeArray),
        "cbuffer member is not an array"
    );
    let stride = t
        .decos
        .value(member, None, Decoration::ArrayStride)
        .context("cbuffer member has no ArrayStride")?;
    Ok((member, stride))
}

fn replace_cbuffer(
    b: &mut Builder,
    cb: &util::CBuffer,
    p: &CBufferProviders,
    matrices: MatrixLayout,
) -> Result<()> {
    let t = Tables::new(b.module_ref());
    let types = t.types();

    let whole = match p.whole {
        Some(w) => {
            let src = Source::new(&types, w, matrices)?;
            ensure!(
                IndexParam::of(&t.defs, w)?.is_none(),
                "whole cbuffer providers take no parameters"
            );
            let legacy = by_path(types.leaves(cb.ty, cb.ty)?);
            Some(Whole {
                src,
                legacy,
                calls: HashMap::new(),
            })
        }
        None => None,
    };
    let tail = match (p.tail, &p.whole) {
        (None, _) => None,
        (Some(_), None) => bail!("tail provider without a whole cbuffer provider"),
        (Some(tp), Some(w)) => {
            ensure!(
                t.defs.get(&tp.ret).is_some_and(|r| matches!(
                    r.class.opcode,
                    Op::TypeInt | Op::TypeFloat
                ) && r.operands[0]
                    == Operand::LiteralBit32(32)),
                "tail provider must return a 32-bit scalar"
            );
            let index = IndexParam::of(&t.defs, tp)?
                .context("tail provider must take a uint word index")?;
            let base = types.packed_size(w.ret, matrices)?.next_multiple_of(16);
            Some(Tail {
                func: tp.func,
                ret: tp.ret,
                index,
                base,
            })
        }
    };
    let mut elems: HashMap<u32, Elem> = HashMap::new();
    for (&n, &e) in &p.elems {
        let (arr, stride) =
            member_array(&t, cb.ty, n).context("element provider's cbuffer member")?;
        let src = Source::new(&types, e, matrices)?;
        let index =
            IndexParam::of(&t.defs, e)?.context("element provider must take a uint index")?;

        let size = src.leaves.last_key_value().map_or(0, |(off, _)| off + 4);
        let group = (size > stride).then(|| size.div_ceil(stride));
        if let Some(k) = group {
            let len = types.array_len(arr)?;
            ensure!(
                len.is_multiple_of(k),
                "element provider's type is not a whole cbuffer array element"
            );
        }
        let legacy = by_path(types.member_element_leaves(cb.ty, n)?);
        elems.insert(
            n,
            Elem {
                src,
                index,
                legacy,
                stride,
                group,
            },
        );
    }

    let mut rw = CBufferRewrite {
        cb,
        tables: &t,
        types,
        matrices,
        whole,
        tail,
        elems,
        chains: HashMap::new(),
    };

    rewrite_functions(b, |b, f, locals| {
        let func = f.def_id().context("function without an id")?;
        let value_types = value_types(f, &b.module_ref().types_global_values);
        let mut fr = FunctionRewrite {
            used: used_components(f, &t.defs, &value_types),
            exprs: t.index_exprs(f),
            value_types,
            locals,
        };
        for block in &mut f.blocks {
            rewrite_block(block, |inst, out| {
                let refs = |id: Word| id == cb.var || rw.chains.contains_key(&id);
                if !id_refs(&inst).any(refs) {
                    out.push(inst);
                    return Ok(());
                }
                match inst.class.opcode {
                    Op::AccessChain | Op::InBoundsAccessChain
                        if inst.operands[0] == Operand::IdRef(cb.var) =>
                    {
                        rw.access_chain(b, &mut fr, out, &inst, func)
                    }
                    Op::Load => rw.load(b, &mut fr, out, &inst),
                    _ => bail!("unsupported cbuffer use"),
                }
            })?;
        }
        Ok(())
    })?;

    if let Some(w) = &rw.whole {
        // Called once at the top of each function reading the cbuffer.
        for (&func, wc) in &w.calls {
            let mut call = vec![Instruction::new(
                Op::FunctionCall,
                Some(w.src.ret),
                Some(wc.call),
                vec![Operand::IdRef(w.src.func)],
            )];
            if let Some(var) = wc.var {
                call.push(store(var, wc.call));
            }
            let f = b
                .module_mut()
                .functions
                .iter_mut()
                .find(|f| f.def_id() == Some(func))
                .context("cbuffer read in a missing function")?;
            let first = &mut f
                .blocks
                .first_mut()
                .context("function without a body")?
                .instructions;
            let at = first
                .iter()
                .take_while(|i| i.class.opcode == Op::Variable)
                .count();
            first.splice(at..at, call);
        }
    }
    let m = b.module_mut();

    let var = Operand::IdRef(cb.var);
    m.annotations.retain(|a| a.operands[0] != var);
    m.debug_names.retain(|a| a.operands[0] != var);
    m.types_global_values
        .retain(|i| i.result_id != Some(cb.var));
    for ep in &mut m.entry_points {
        ep.operands.retain(|o| *o != var);
    }
    Ok(())
}

impl CBufferRewrite<'_> {
    fn region(&self, region: Region) -> Result<(&Source, &LegacyLeaves)> {
        match region {
            Region::Element(n) => self.elems.get(&n).map(|e| (&e.src, &e.legacy)),
            Region::Whole => self.whole.as_ref().map(|w| (&w.src, &w.legacy)),
        }
        .context("no provider for a cbuffer region")
    }

    fn access_chain(
        &mut self,
        b: &mut Builder,
        fr: &mut FunctionRewrite,
        out: &mut Vec<Instruction>,
        inst: &Instruction,
        func: Word,
    ) -> Result<()> {
        let consts = self.types.consts;
        let result = inst.result_id.context("OpAccessChain without a result")?;
        let idx: Vec<Word> = inst.operands[1..]
            .iter()
            .map(|o| o.unwrap_id_ref())
            .collect();
        let n = *consts
            .get(
                idx.first()
                    .context("cbuffer access chain without indices")?,
            )
            .context("dynamic cbuffer member index")? as u32;
        let constant = |ids: &[Word]| -> Option<Vec<u32>> {
            ids.iter()
                .map(|i| consts.get(i).map(|&c| c as u32))
                .collect()
        };
        let below_element = || -> Result<Vec<u32>> {
            ensure!(idx.len() >= 2, "cbuffer array is accessed as a whole");
            constant(&idx[2..]).context("cbuffer array is indexed dynamically below the element")
        };

        let chain = if let Some(e) = self.elems.get(&n) {
            let src = &e.src;
            let rest = below_element()?;
            let (index, base, dynamic) = match e.group {
                None => (idx[1], 0, None),
                Some(k) => {
                    let s = fr
                        .exprs
                        .split(idx[1], k as u64)
                        .context("cbuffer array index is not `x * elements + c (+ d)`")?;
                    let (step, first) = (s.step, s.first);
                    ensure!(
                        step == k as u64,
                        "element provider spans a different number of elements than the shader's stride"
                    );
                    ensure!(first < step, "cbuffer array index reaches past one element");
                    let dynamic = combine_dynamic(b, fr, out, s.dynamic, e.stride)
                        .context("indexing a cbuffer array")?;
                    (s.x, first as u32 * e.stride, dynamic)
                }
            };
            let index_ty = fr.value_type(index)?;
            let arg = fr
                .locals
                .index_arg(b, out, src.func, e.index, index, index_ty);
            let call = emit(out, b, Op::FunctionCall, src.ret, &[src.func, arg]);
            let value = match dynamic {
                None => call,
                Some(_) => {
                    let var = fr.locals.spill(b, src.ret);
                    out.push(store(var, call));
                    var
                }
            };
            Chain {
                value,
                region: Region::Element(n),
                path: rest,
                base,
                dynamic,
            }
        } else {
            let Some(w) = &mut self.whole else {
                bail!("cbuffer member is read but has no provider");
            };
            let wc = w.calls.entry(func).or_insert_with(|| WholeCall {
                call: b.id(),
                var: None,
            });
            match constant(&idx) {
                Some(path) => Chain {
                    value: wc.call,
                    region: Region::Whole,
                    path,
                    base: 0,
                    dynamic: None,
                },
                None => {
                    let rest = below_element()?;
                    let (first, terms) = fr.exprs.sum(idx[1]);
                    let (_, stride) = member_array(self.tables, self.cb.ty, n)?;
                    let dynamic = combine_dynamic(b, fr, out, terms, stride)
                        .context("indexing a cbuffer array")?;
                    let ret = w.src.ret;
                    let var = *wc.var.get_or_insert_with(|| fr.locals.spill(b, ret));
                    Chain {
                        value: var,
                        region: Region::Whole,
                        path: [vec![n, first as u32], rest].concat(),
                        base: 0,
                        dynamic,
                    }
                }
            }
        };
        self.chains.insert(result, chain);
        Ok(())
    }

    fn load(
        &self,
        b: &mut Builder,
        fr: &mut FunctionRewrite,
        out: &mut Vec<Instruction>,
        inst: &Instruction,
    ) -> Result<()> {
        let chain = self
            .chains
            .get(&inst.operands[0].unwrap_id_ref())
            .context("cbuffer is loaded whole")?;
        let (src, legacy) = self.region(chain.region)?;
        let (Some(load_ty), Some(result)) = (inst.result_type, inst.result_id) else {
            bail!("OpLoad without a result")
        };
        let lt = &self.types.defs[&load_ty];
        let scalar_paths: Vec<Vec<u32>> = if lt.class.opcode == Op::TypeVector {
            (0..lt.operands[1].unwrap_literal_bit32())
                .map(|i| chain.path.iter().copied().chain([i]).collect())
                .collect()
        } else {
            vec![chain.path.clone()]
        };
        let mut parts = Vec::with_capacity(scalar_paths.len());
        for (k, sp) in scalar_paths.iter().enumerate() {
            let want = legacy
                .get(sp)
                .context("cbuffer load of a non-scalar/vector")?;
            let off = chain.base + want.offset;
            let Some(leaf) = src.leaves.get(&off) else {
                let unused = scalar_paths.len() > 1
                    && fr
                        .used
                        .get(&result)
                        .is_some_and(|u| !u.contains(&(k as u32)));
                if unused {
                    parts.push(constant(b, want.ty, 0));
                    continue;
                }
                let tail = self
                    .tail
                    .as_ref()
                    .filter(|t| chain.region == Region::Whole && off >= t.base);
                let Some(t) = tail else {
                    bail!("cbuffer read not covered by its provider");
                };
                let mut word = constant(b, t.index.ty, (off - t.base) / 4);
                if let Some(d) = chain.dynamic {
                    ensure!(
                        d.stride % 4 == 0,
                        "dynamic tail index steps are not whole words"
                    );
                    let mut steps = d.index;
                    if d.ty != t.index.ty {
                        steps = emit(out, b, Op::Bitcast, t.index.ty, &[steps]);
                    }
                    if d.stride != 4 {
                        let words_per_step = constant(b, t.index.ty, d.stride / 4);
                        steps = emit(out, b, Op::IMul, t.index.ty, &[steps, words_per_step]);
                    }
                    word = emit(out, b, Op::IAdd, t.index.ty, &[word, steps]);
                }
                let arg = fr
                    .locals
                    .index_arg(b, out, t.func, t.index, word, t.index.ty);
                let value = emit(out, b, Op::FunctionCall, t.ret, &[t.func, arg]);
                parts.push(
                    convert(b, self.types.defs, out, value, t.ret, want.ty)
                        .context("converting a tail word")?,
                );
                continue;
            };
            let value = match chain.dynamic {
                None => extract(chain.value, leaf, out, b),
                Some(d) => self.load_dynamic(b, out, chain, d, leaf)?,
            };
            let v = convert(b, self.types.defs, out, value, leaf.ty, want.ty)
                .context("converting a cbuffer scalar")?;
            parts.push(v);
        }
        let op = if parts.len() == 1 {
            Op::CopyObject
        } else {
            Op::CompositeConstruct
        };
        out.push(Instruction::new(
            op,
            Some(load_ty),
            Some(result),
            parts.into_iter().map(Operand::IdRef).collect(),
        ));
        Ok(())
    }

    fn load_dynamic(
        &self,
        b: &mut Builder,
        out: &mut Vec<Instruction>,
        chain: &Chain,
        d: DynamicIndex,
        leaf: &Leaf,
    ) -> Result<Word> {
        let (src, _) = self.region(chain.region)?;
        let stride = d.stride;
        let at = dynamic_array_step(&self.types, src.ret, &leaf.path, stride, self.matrices)?
            .context("dynamic cbuffer index has no provider array of matching stride")?;
        let int = d.ty;
        let mut indices = vec![chain.value];
        for (depth, &i) in leaf.path.iter().enumerate() {
            let c = constant(b, int, i);
            indices.push(if depth != at {
                c
            } else if i == 0 {
                d.index
            } else {
                emit(out, b, Op::IAdd, int, &[c, d.index])
            });
        }
        let ptr = b.type_pointer(None, StorageClass::Function, leaf.ty);
        let access = emit(out, b, Op::AccessChain, ptr, &indices);
        Ok(emit(out, b, Op::Load, leaf.ty, &[access]))
    }
}

fn dynamic_array_step(
    types: &Types,
    ret: Word,
    path: &[u32],
    stride: u32,
    matrices: MatrixLayout,
) -> Result<Option<usize>> {
    let mut ty = ret;
    let mut found = None;
    for (depth, &i) in path.iter().enumerate() {
        let t = &types.defs[&ty];
        ty = match t.class.opcode {
            Op::TypeStruct => t.operands[i as usize].unwrap_id_ref(),
            Op::TypeArray => {
                let elem = t.operands[0].unwrap_id_ref();
                if types.packed_size(elem, matrices)?.next_multiple_of(16) == stride {
                    found = Some(depth);
                }
                elem
            }
            _ => break,
        };
    }
    Ok(found)
}

fn address_masks(m: &Module, consts: &HashMap<Word, u64>) -> HashSet<Word> {
    let id_at = |i: &Instruction, n: usize| i.operands.get(n).and_then(Operand::id_ref_any);
    m.functions
        .iter()
        .filter(|f| {
            let ([p0, p1], [b]) = (f.parameters.as_slice(), f.blocks.as_slice()) else {
                return false;
            };
            let [div, and, ret] = b.instructions.as_slice() else {
                return false;
            };
            let masked = [id_at(and, 0), id_at(and, 1)];
            div.class.opcode == Op::UDiv
                && id_at(div, 0).and_then(|c| consts.get(&c)) == Some(&(u32::MAX as u64))
                && id_at(div, 1) == p1.result_id
                && and.class.opcode == Op::BitwiseAnd
                && (masked == [p0.result_id, div.result_id]
                    || masked == [div.result_id, p0.result_id])
                && ret.class.opcode == Op::ReturnValue
                && id_at(ret, 0) == and.result_id
        })
        .filter_map(Function::def_id)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Term {
    id: Word,
    scale: u64,
}

struct Split {
    x: Word,
    step: u64,
    first: u64,
    dynamic: Vec<Term>,
}

struct IndexExprs<'a> {
    local: HashMap<Word, Instruction>,
    consts: &'a HashMap<Word, u64>,
    masks: &'a HashSet<Word>,
}

impl IndexExprs<'_> {
    fn binary(&self, id: Word) -> Option<(Op, Word, Word)> {
        let i = self.local.get(&id)?;
        match i.operands.as_slice() {
            [Operand::IdRef(a), Operand::IdRef(b)] => Some((i.class.opcode, *a, *b)),
            _ => None,
        }
    }

    fn unmask(&self, mut id: Word) -> Word {
        while let Some(i) = self.local.get(&id).filter(|i| {
            i.class.opcode == Op::FunctionCall
                && self.masks.contains(&i.operands[0].unwrap_id_ref())
        }) {
            id = i.operands[1].unwrap_id_ref();
        }
        id
    }

    fn scaled(&self, id: Word) -> Option<Term> {
        let (op, a, b) = self.binary(id)?;
        let term = |x: Word, scale: u64| Term {
            id: self.unmask(x),
            scale,
        };
        match op {
            Op::IMul => self
                .consts
                .get(&b)
                .map(|&c| term(a, c))
                .or_else(|| self.consts.get(&a).map(|&c| term(b, c))),
            Op::ShiftLeftLogical => self
                .consts
                .get(&b)
                .filter(|&&c| c < 64)
                .map(|&c| term(a, 1 << c)),
            _ => None,
        }
    }

    fn zero_low_bits(&self, id: Word) -> u32 {
        let id = self.unmask(id);
        if let Some(&c) = self.consts.get(&id) {
            return c.trailing_zeros();
        }
        let Some((op, a, b)) = self.binary(id) else {
            return 0;
        };
        let (za, zb) = (self.zero_low_bits(a), self.zero_low_bits(b));
        match op {
            Op::IMul => (za + zb).min(64),
            Op::ShiftLeftLogical => self
                .consts
                .get(&b)
                .map_or(0, |&c| za.saturating_add(c.min(64) as u32).min(64)),
            Op::IAdd | Op::BitwiseOr => za.min(zb),
            Op::BitwiseAnd => za.max(zb),
            _ => 0,
        }
    }

    fn disjoint(&self, a: Word, b: Word) -> bool {
        let below =
            |c: u64, other: Word| c.checked_shr(self.zero_low_bits(other)).unwrap_or(0) == 0;
        match (self.consts.get(&a), self.consts.get(&b)) {
            (_, Some(&c)) => below(c, a),
            (Some(&c), _) => below(c, b),
            _ => false,
        }
    }

    fn sum(&self, index: Word) -> (u64, Vec<Term>) {
        let (mut constant, mut terms) = (0, Vec::new());
        let mut stack = vec![index];
        while let Some(t) = stack.pop() {
            let t = self.unmask(t);
            if let Some(&c) = self.consts.get(&t) {
                constant += c;
                continue;
            }
            match self.binary(t) {
                Some((Op::IAdd, a, b)) => stack.extend([a, b]),
                Some((Op::BitwiseOr, a, b)) if self.disjoint(a, b) => stack.extend([a, b]),
                _ => terms.push(self.scaled(t).unwrap_or(Term { id: t, scale: 1 })),
            }
        }
        (constant, terms)
    }

    fn split(&self, index: Word, want: u64) -> Option<Split> {
        if want == 1 {
            return Some(Split {
                x: self.unmask(index),
                step: 1,
                first: 0,
                dynamic: vec![],
            });
        }
        let (first, mut terms) = self.sum(index);
        let pick = terms
            .iter()
            .position(|t| t.scale == want)
            .or_else(|| terms.iter().position(|t| t.scale > 1))?;
        let Term { id: x, scale: step } = terms.remove(pick);
        Some(Split {
            x,
            step,
            first,
            dynamic: terms,
        })
    }
}

fn combine_dynamic(
    b: &mut Builder,
    fr: &FunctionRewrite,
    out: &mut Vec<Instruction>,
    terms: Vec<Term>,
    stride: u32,
) -> Result<Option<DynamicIndex>> {
    let Some(scale) = terms.iter().map(|t| t.scale).reduce(gcd) else {
        return Ok(None);
    };
    let mut sum = None;
    for t in terms {
        let ty = fr.value_type(t.id)?;
        let mut v = t.id;
        if t.scale != scale {
            let factor = constant(b, ty, (t.scale / scale) as u32);
            v = emit(out, b, Op::IMul, ty, &[v, factor]);
        }
        sum = Some(match sum {
            None => v,
            Some(s) => emit(out, b, Op::IAdd, fr.value_type(s)?, &[s, v]),
        });
    }
    let stride = u32::try_from(scale)
        .ok()
        .and_then(|s| s.checked_mul(stride))
        .context("dynamic index step overflows")?;
    sum.map(|index| {
        Ok(DynamicIndex {
            index,
            ty: fr.value_type(index)?,
            stride,
        })
    })
    .transpose()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn used_components(
    f: &Function,
    defs: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
) -> HashMap<Word, HashSet<u32>> {
    let width = |v: Word| {
        let t = defs.get(value_types.get(&v)?)?;
        (t.class.opcode == Op::TypeVector).then(|| t.operands[1].unwrap_literal_bit32())
    };
    let mut used: HashMap<Word, HashSet<u32>> = HashMap::new();
    let mut whole: HashSet<Word> = HashSet::new();
    let mut forwards: Vec<(Word, Word)> = Vec::new();
    for i in f.blocks.iter().flat_map(|b| &b.instructions) {
        match (i.class.opcode, i.operands.as_slice(), i.result_id) {
            (Op::Bitcast | Op::CopyObject, &[Operand::IdRef(v)], Some(r))
                if width(v).is_some() && width(v) == width(r) =>
            {
                forwards.push((v, r));
            }
            (Op::CompositeExtract, [Operand::IdRef(v), Operand::LiteralBit32(first), ..], _) => {
                used.entry(*v).or_default().insert(*first);
            }
            (Op::VectorShuffle, [Operand::IdRef(a), Operand::IdRef(b), comps @ ..], _)
                if width(*a).is_some() =>
            {
                let n = width(*a).unwrap_or_default();
                for c in comps.iter().filter_map(|c| match c {
                    Operand::LiteralBit32(c) if *c != u32::MAX => Some(*c),
                    _ => None,
                }) {
                    let (v, c) = if c < n { (*a, c) } else { (*b, c - n) };
                    used.entry(v).or_default().insert(c);
                }
            }
            _ => whole.extend(id_refs(i)),
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &(src, dst) in &forwards {
            if whole.contains(&dst) {
                changed |= whole.insert(src);
            } else if let Some(d) = used.get(&dst).cloned() {
                let s = used.entry(src).or_default();
                let before = s.len();
                s.extend(d);
                changed |= s.len() != before;
            }
        }
    }
    used.retain(|v, _| !whole.contains(v));
    used
}

fn extract(value: Word, leaf: &Leaf, out: &mut Vec<Instruction>, b: &mut Builder) -> Word {
    if leaf.path.is_empty() {
        return value;
    }
    let mut ops = vec![Operand::IdRef(value)];
    ops.extend(leaf.path.iter().map(|&i| Operand::LiteralBit32(i)));
    let id = b.id();
    out.push(Instruction::new(
        Op::CompositeExtract,
        Some(leaf.ty),
        Some(id),
        ops,
    ));
    id
}

fn convert(
    b: &mut Builder,
    defs: &HashMap<Word, Instruction>,
    out: &mut Vec<Instruction>,
    value: Word,
    from: Word,
    to: Word,
) -> Result<Word> {
    if from == to {
        return Ok(value);
    }
    let (Some(fd), Some(td)) = (defs.get(&from), defs.get(&to)) else {
        bail!("scalar type is not defined")
    };
    let (f, t) = (fd.class.opcode, td.class.opcode);
    let id = b.id();
    match (f, t) {
        (Op::TypeFloat | Op::TypeInt, Op::TypeFloat | Op::TypeInt) => {
            ensure!(fd.operands[0] == td.operands[0], "scalar width mismatch");
            out.push(Instruction::new(
                Op::Bitcast,
                Some(to),
                Some(id),
                vec![Operand::IdRef(value)],
            ));
        }
        (Op::TypeBool, Op::TypeFloat | Op::TypeInt) => {
            let one = if t == Op::TypeFloat {
                1.0f32.to_bits()
            } else {
                1
            };
            let (one, zero) = (constant(b, to, one), constant(b, to, 0));
            out.push(Instruction::new(
                Op::Select,
                Some(to),
                Some(id),
                vec![
                    Operand::IdRef(value),
                    Operand::IdRef(one),
                    Operand::IdRef(zero),
                ],
            ));
        }
        _ => bail!("unsupported scalar conversion"),
    }
    Ok(id)
}

fn remove_unreferenced_globals(m: &mut Module) {
    let mut referenced: HashSet<Word> = m
        .entry_points
        .iter()
        .flat_map(|e| e.operands.iter().skip(3))
        .filter_map(Operand::id_ref_any)
        .collect();
    referenced.extend(
        m.functions
            .iter()
            .flat_map(|f| f.blocks.iter().flat_map(|b| &b.instructions))
            .flat_map(id_refs),
    );
    let dead: HashSet<Word> = m
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| i.result_id)
        .filter(|id| !referenced.contains(id))
        .collect();
    m.types_global_values
        .retain(|i| !i.result_id.is_some_and(|id| dead.contains(&id)));
    remove_targeting(m, &dead);
}

fn remove_unreachable_functions(m: &mut Module) {
    let calls: HashMap<Word, Vec<Word>> = m
        .functions
        .iter()
        .filter_map(|f| {
            let callees = f
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .filter(|i| i.class.opcode == Op::FunctionCall)
                .filter_map(|i| i.operands.first().and_then(Operand::id_ref_any))
                .collect();
            Some((f.def_id()?, callees))
        })
        .collect();
    let mut live: HashSet<Word> = m
        .entry_points
        .iter()
        .filter_map(|e| e.operands.get(1).and_then(Operand::id_ref_any))
        .collect();
    let mut stack: Vec<Word> = live.iter().copied().collect();
    while let Some(f) = stack.pop() {
        for &c in calls.get(&f).into_iter().flatten() {
            if live.insert(c) {
                stack.push(c);
            }
        }
    }
    let (keep, dead): (Vec<_>, Vec<_>) = std::mem::take(&mut m.functions)
        .into_iter()
        .partition(|f| f.def_id().is_some_and(|id| live.contains(&id)));
    m.functions = keep;
    let gone: HashSet<Word> = dead
        .iter()
        .flat_map(Function::all_inst_iter)
        .filter_map(|i| i.result_id)
        .collect();
    remove_targeting(m, &gone);
}

fn remove_targeting(m: &mut Module, ids: &HashSet<Word>) {
    let targets =
        |i: &Instruction| matches!(i.operands.first(), Some(Operand::IdRef(t)) if ids.contains(t));
    m.annotations.retain(|i| !targets(i));
    m.debug_names.retain(|i| !targets(i));
}

pub(crate) fn remap_refs(inst: &mut Instruction, mut f: impl FnMut(Word) -> Word) {
    if let Some(t) = inst.result_type.as_mut() {
        *t = f(*t);
    }
    for id in inst.operands.iter_mut().filter_map(Operand::id_ref_any_mut) {
        *id = f(*id);
    }
}

pub(crate) fn linkage_of(a: &Instruction) -> Option<(Word, String, LinkageType)> {
    if a.class.opcode != Op::Decorate
        || a.operands.get(1) != Some(&Operand::Decoration(Decoration::LinkageAttributes))
    {
        return None;
    }
    let Some(Operand::LinkageType(ty)) = a.operands.get(3) else {
        return None;
    };
    Some((
        a.operands[0].unwrap_id_ref(),
        util::string_operand(a, 2)?,
        *ty,
    ))
}
