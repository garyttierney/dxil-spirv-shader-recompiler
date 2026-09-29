use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::Module,
    spirv::{Decoration, Op, StorageClass, Word},
};

use crate::util::{self, TypeTables, same_type};

struct CBufferUse<'a> {
    tables: &'a TypeTables,
    views: Vec<Word>,
    set: Option<u32>,
    binding: Option<u32>,
}

pub fn check_cbuffers<'a>(stages: impl IntoIterator<Item = &'a Module>) -> Result<()> {
    let stages: Vec<(&Module, TypeTables)> = stages
        .into_iter()
        .map(|m| (m, TypeTables::new(m)))
        .collect();
    let mut seen: BTreeMap<String, Vec<CBufferUse>> = BTreeMap::new();
    for (m, tables) in &stages {
        for named in util::named_cbuffers(m)? {
            let first = named.first();
            let u = CBufferUse {
                tables,
                views: named.views.iter().map(|v| v.ty).collect(),
                set: first.set,
                binding: first.binding,
            };
            seen.entry(named.name).or_default().push(u);
        }
    }
    for uses in seen.values() {
        let first = &uses[0];
        for u in &uses[1..] {
            ensure!(
                (u.set, u.binding) == (first.set, first.binding),
                "a cbuffer has different bindings in two stages"
            );
            for &v in &u.views {
                let counterpart = first
                    .views
                    .iter()
                    .find(|&&f| same_view_kind(first.tables, f, u.tables, v));
                if let Some(&f) = counterpart {
                    ensure!(
                        same_type(first.tables, f, u.tables, v, true),
                        "a cbuffer has different layouts in two stages"
                    );
                }
            }
        }
    }
    Ok(())
}

fn same_view_kind(a: &TypeTables, va: Word, b: &TypeTables, vb: Word) -> bool {
    match (flat_element(a, va), flat_element(b, vb)) {
        (Some(ea), Some(eb)) => same_type(a, ea, b, eb, true),
        (None, None) => true,
        _ => false,
    }
}

fn flat_element(t: &TypeTables, block: Word) -> Option<Word> {
    let [member] = t.defs.get(&block)?.operands.as_slice() else {
        return None;
    };
    let array = t.defs.get(&member.unwrap_id_ref())?;
    (array.class.opcode == Op::TypeArray).then(|| array.operands[0].unwrap_id_ref())
}

struct Slot {
    location: u32,
    component: u32,
    count: u32,
    scalar: Word,
}

fn interface(m: &Module, t: &TypeTables, class: StorageClass) -> Result<Vec<Slot>> {
    let mut out = Vec::new();
    for (id, mut ty) in util::variables(m, &t.defs, class) {
        let Some(location) = t.decos.value(id, None, Decoration::Location) else {
            continue;
        };
        let component = t.decos.value(id, None, Decoration::Component).unwrap_or(0);
        let mut rows = 1;
        if t.defs[&ty].class.opcode == Op::TypeArray {
            let len = t.defs[&ty].operands[1].unwrap_id_ref();
            rows = *t
                .consts
                .get(&len)
                .context("interface array length is not a constant")? as u32;
            ty = t.defs[&ty].operands[0].unwrap_id_ref();
        }
        let (count, scalar) = match t.defs[&ty].class.opcode {
            Op::TypeVector => (
                t.defs[&ty].operands[1].unwrap_literal_bit32(),
                t.defs[&ty].operands[0].unwrap_id_ref(),
            ),
            _ => (1, ty),
        };
        out.extend((0..rows).map(|r| Slot {
            location: location + r,
            component,
            count,
            scalar,
        }));
    }
    Ok(out)
}

/// The first location after those `m`'s `class` variables use.
pub(crate) fn next_location(m: &Module, class: StorageClass) -> Result<u32> {
    let t = TypeTables::new(m);
    Ok(interface(m, &t, class)?
        .iter()
        .map(|s| s.location + 1)
        .max()
        .unwrap_or(0))
}

/// The first location neither `vs`'s outputs nor `fs`'s inputs use: where the stub's hooks pass
/// values between the stages, for [`crate::link::link`].
pub fn varying_location(vs: &Module, fs: Option<&Module>) -> Result<u32> {
    let mut location = next_location(vs, StorageClass::Output)?;
    if let Some(fs) = fs {
        location = location.max(next_location(fs, StorageClass::Input)?);
    }
    Ok(location)
}

pub fn check_io(vs: &Module, fs: &Module) -> Result<()> {
    let (vt, ft) = (TypeTables::new(vs), TypeTables::new(fs));
    let outputs = interface(vs, &vt, StorageClass::Output)?;
    for i in interface(fs, &ft, StorageClass::Input)? {
        let at = |o: &&Slot| o.location == i.location && o.component == i.component;
        match outputs.iter().find(at) {
            None => bail!("a fragment input has no matching vertex output"),
            Some(o) if !same_type(&vt, o.scalar, &ft, i.scalar, false) || o.count < i.count => {
                bail!("a fragment input is fed by a vertex output of another type")
            }
            Some(_) => {}
        }
    }
    Ok(())
}
