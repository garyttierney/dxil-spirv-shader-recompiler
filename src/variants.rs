use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::{Builder, Instruction, Module, Operand},
    spirv::{
        AddressingModel, Decoration, FunctionControl, LinkageType, Op, SelectionControl,
        StorageClass, Word,
    },
};

use crate::{
    link::{self, TypeTable, decorations_by_target, linkage_of, remap_refs},
    util::{self, TypeTables, same_type},
};

pub const VARIANT_SPEC_ID: u32 = 0;

pub fn merge(variants: Vec<(u32, Module)>) -> Result<Module> {
    let mut variants = variants.into_iter();
    let (first_case, mut base) = variants.next().context("no variants to merge")?;
    let rest: Vec<(u32, Module)> = variants.collect();
    let mut cases: HashSet<u32> = HashSet::from([first_case]);
    ensure!(
        rest.iter().all(|(c, _)| cases.insert(*c)),
        "two variants have the same case"
    );

    let mut header = base.header.take().context("variant has no header")?;
    for (_, v) in &rest {
        let version = v
            .header
            .as_ref()
            .context("variant has no header")?
            .version();
        if version > header.version() {
            header.set_version(version.0, version.1);
        }
        adopt_addressing_model(&mut base, v)?;
    }

    let base_entry = Entry::of(&base)?;
    let base_interface = interface(&base);
    let mut next_id = header.bound;
    let mut shared = Shared {
        types: TypeTable::new(&base),
        imports: imports(&base),
    };
    let mut calls = vec![(first_case, base_entry.func)];
    for (case, v) in &rest {
        let entry = Entry::of(v)?;
        ensure!(
            entry.model == base_entry.model && same_modes(&base, base_entry.func, v, entry.func),
            "variants have different execution models or modes"
        );
        let f = absorb(
            &mut base,
            &mut next_id,
            &mut shared,
            &base_interface,
            v,
            entry.func,
        )?;
        calls.push((*case, f));
    }

    let all_globals = header.version() >= (1, 4);
    header.bound = next_id;
    base.header = Some(header);
    let mut b = Builder::new_from_module(base);
    let main = add_entry_point(&mut b, base_entry.func, &calls)?;
    let mut merged = b.module();
    retarget_entry_point(&mut merged, base_entry.func, main, all_globals);
    Ok(merged)
}

struct Entry {
    model: Operand,
    func: Word,
}

impl Entry {
    fn of(m: &Module) -> Result<Self> {
        let [ep] = m.entry_points.as_slice() else {
            bail!("a variant must have exactly one entry point");
        };
        Ok(Self {
            model: ep.operands[0].clone(),
            func: ep.operands[1].unwrap_id_ref(),
        })
    }
}

fn adopt_addressing_model(base: &mut Module, v: &Module) -> Result<()> {
    let psb = Operand::AddressingModel(AddressingModel::PhysicalStorageBuffer64);
    let addressing = |m: &Module| m.memory_model.as_ref().map(|i| i.operands[0].clone());
    if addressing(base).as_ref() == Some(&psb) && addressing(v).is_some() {
        return Ok(());
    }
    link::match_addressing_model(base, v)
}

fn same_modes(a: &Module, fa: Word, b: &Module, fb: Word) -> bool {
    let modes = |m: &Module, f: Word| -> Vec<(Op, Vec<Operand>)> {
        m.execution_modes
            .iter()
            .filter(|i| i.operands.first() == Some(&Operand::IdRef(f)))
            .map(|i| (i.class.opcode, i.operands[1..].to_vec()))
            .collect()
    };
    let (x, y) = (modes(a, fa), modes(b, fb));
    x.len() == y.len() && x.iter().all(|e| y.contains(e))
}

#[derive(PartialEq)]
struct Slot {
    class: StorageClass,
    location: Option<u32>,
    component: Option<u32>,
    builtin: Option<Operand>,
}

fn interface(m: &Module) -> Vec<(Word, Slot, Word)> {
    let decos = util::Decorations::new(m);
    let builtin = |var: Word| {
        m.annotations.iter().find_map(|a| {
            (a.class.opcode == Op::Decorate
                && a.operands[0] == Operand::IdRef(var)
                && a.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn)))
            .then(|| a.operands.get(2).cloned())
            .flatten()
        })
    };
    let defs = util::global_defs(m);
    m.types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| {
            let var = i.result_id?;
            let class = match i.operands.first()? {
                Operand::StorageClass(c @ (StorageClass::Input | StorageClass::Output)) => *c,
                _ => return None,
            };
            let pointee = defs.get(&i.result_type?)?.operands.get(1)?.id_ref_any()?;
            let slot = Slot {
                class,
                location: decos.value(var, None, Decoration::Location),
                component: decos.value(var, None, Decoration::Component),
                builtin: builtin(var),
            };
            Some((var, slot, pointee))
        })
        .collect()
}

fn imports(m: &Module) -> HashMap<String, Word> {
    m.annotations
        .iter()
        .filter_map(linkage_of)
        .filter(|(_, _, ty)| *ty == LinkageType::Import)
        .map(|(f, name, _)| (name, f))
        .collect()
}

struct Shared {
    types: TypeTable,
    imports: HashMap<String, Word>,
}

fn absorb(
    base: &mut Module,
    next_id: &mut Word,
    shared: &mut Shared,
    base_interface: &[(Word, Slot, Word)],
    v: &Module,
    v_entry: Word,
) -> Result<Word> {
    let mut map: HashMap<Word, Word> = HashMap::new();
    let mut fresh: HashSet<Word> = HashSet::new();
    let mut translate = |map: &mut HashMap<Word, Word>, id: Word| {
        *map.entry(id).or_insert_with(|| {
            fresh.insert(id);
            *next_id += 1;
            *next_id - 1
        })
    };

    for (list, from) in [
        (&mut base.capabilities, &v.capabilities),
        (&mut base.extensions, &v.extensions),
    ] {
        for i in from {
            if !list.iter().any(|x| x.operands == i.operands) {
                list.push(i.clone());
            }
        }
    }
    for imp in &v.ext_inst_imports {
        let old = imp.result_id.context("OpExtInstImport without a result")?;
        let name = util::string_operand(imp, 0);
        match base
            .ext_inst_imports
            .iter()
            .find(|x| util::string_operand(x, 0) == name)
        {
            Some(existing) => {
                map.insert(
                    old,
                    existing
                        .result_id
                        .context("OpExtInstImport without a result")?,
                );
            }
            None => {
                let mut imp = imp.clone();
                imp.result_id = Some(translate(&mut map, old));
                base.ext_inst_imports.push(imp);
            }
        }
    }

    let (base_types, v_types) = (TypeTables::new(base), TypeTables::new(v));
    let v_interface = interface(v);
    ensure!(
        v_interface.len() == base_interface.len(),
        "variants have different stage interfaces"
    );
    for (var, slot, ty) in &v_interface {
        let (base_var, _, base_ty) = base_interface
            .iter()
            .find(|(_, s, _)| s == slot)
            .context("variants have different stage interfaces")?;
        ensure!(
            same_type(&base_types, *base_ty, &v_types, *ty, false),
            "variants have different stage interfaces"
        );
        map.insert(*var, *base_var);
    }

    let v_decos = decorations_by_target(v);
    for inst in &v.types_global_values {
        let old = inst.result_id.context("global without a result")?;
        if map.contains_key(&old) {
            continue;
        }
        let mut i = inst.clone();
        remap_refs(&mut i, |id| translate(&mut map, id));
        let decos = v_decos.get(&old).cloned().unwrap_or_default();
        if let Some(existing) = shared.types.find(&i, &decos) {
            map.insert(old, existing);
            continue;
        }
        i.result_id = Some(translate(&mut map, old));
        shared.types.insert(&i, decos);
        base.types_global_values.push(i);
    }

    let mut new_imports = Vec::new();
    let mut shared_imports = HashSet::new();
    for (name, f) in imports(v) {
        match shared.imports.get(&name) {
            Some(&existing) => {
                map.insert(f, existing);
                shared_imports.insert(f);
            }
            None => new_imports.push((name, f)),
        }
    }
    for f in &v.functions {
        let id = f
            .def
            .as_ref()
            .and_then(|d| d.result_id)
            .context("function without an id")?;
        if shared_imports.contains(&id) {
            continue;
        }
        let mut f = f.clone();
        let mut each = |i: &mut Instruction| {
            remap_refs(i, |id| translate(&mut map, id));
            if let Some(r) = i.result_id.as_mut() {
                *r = translate(&mut map, *r);
            }
        };
        f.def.iter_mut().for_each(&mut each);
        f.parameters.iter_mut().for_each(&mut each);
        for b in &mut f.blocks {
            b.label.iter_mut().for_each(&mut each);
            b.instructions.iter_mut().for_each(&mut each);
        }
        f.end.iter_mut().for_each(&mut each);
        base.functions.push(f);
    }
    for (name, f) in new_imports {
        shared.imports.insert(name, map[&f]);
    }

    for a in v.annotations.iter().chain(&v.debug_names) {
        let Some(&Operand::IdRef(target)) = a.operands.first() else {
            continue;
        };
        if !fresh.contains(&target) {
            continue;
        }
        let mut a = a.clone();
        remap_refs(&mut a, |id| map.get(&id).copied().unwrap_or(id));
        match a.class.opcode {
            Op::Name | Op::MemberName => base.debug_names.push(a),
            _ => base.annotations.push(a),
        }
    }
    map.get(&v_entry)
        .copied()
        .context("variant entry function missing")
}

fn add_entry_point(b: &mut Builder, entry: Word, calls: &[(u32, Word)]) -> Result<Word> {
    let def = b
        .module_ref()
        .functions
        .iter()
        .find_map(|f| f.def.as_ref().filter(|d| d.result_id == Some(entry)))
        .context("entry function missing")?;
    let void = def.result_type.context("entry function without a type")?;
    let fn_ty = def
        .operands
        .get(1)
        .and_then(Operand::id_ref_any)
        .context("entry function without a type")?;

    let uint = b.type_int(32, 0);
    let spec = b.spec_constant_bit32(uint, 0);
    b.decorate(
        spec,
        Decoration::SpecId,
        [Operand::LiteralBit32(VARIANT_SPEC_ID)],
    );

    let main = b.begin_function(void, None, FunctionControl::NONE, fn_ty)?;
    b.begin_block(None)?;
    let (default, merge) = (b.id(), b.id());
    let cases: Vec<(u32, Word, Word)> = calls.iter().map(|&(case, f)| (case, f, b.id())).collect();
    b.selection_merge(merge, SelectionControl::NONE)?;
    b.switch(
        spec,
        default,
        cases
            .iter()
            .map(|&(case, _, block)| (Operand::LiteralBit32(case), block)),
    )?;
    for &(_, f, block) in &cases {
        b.begin_block(Some(block))?;
        b.function_call(void, None, f, [])?;
        b.branch(merge)?;
    }
    b.begin_block(Some(default))?;
    b.branch(merge)?;
    b.begin_block(Some(merge))?;
    b.ret()?;
    b.end_function()?;
    Ok(main)
}

fn retarget_entry_point(base: &mut Module, old: Word, main: Word, all_globals: bool) {
    let interface: Vec<Operand> = base
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter(|i| match i.operands.first() {
            Some(Operand::StorageClass(StorageClass::Input | StorageClass::Output)) => true,
            Some(Operand::StorageClass(StorageClass::Function)) => false,
            _ => all_globals,
        })
        .filter_map(|i| i.result_id.map(Operand::IdRef))
        .collect();
    for ep in &mut base.entry_points {
        if ep.operands.get(1) == Some(&Operand::IdRef(old)) {
            ep.operands[1] = Operand::IdRef(main);
            ep.operands.truncate(3);
            ep.operands.extend(interface.iter().cloned());
        }
    }
    for mode in &mut base.execution_modes {
        if mode.operands.first() == Some(&Operand::IdRef(old)) {
            mode.operands[0] = Operand::IdRef(main);
        }
    }
}
