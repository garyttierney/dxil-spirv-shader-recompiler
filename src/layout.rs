use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, bail, ensure};
use rspirv::{
    dr::Instruction,
    spirv::{Decoration, Op, Word},
};

use crate::util::Decorations;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone)]
pub struct Leaf {
    pub path: Vec<u32>,
    pub ty: Word,
}

pub struct Types<'a> {
    pub defs: &'a HashMap<Word, Instruction>,
    pub consts: &'a HashMap<Word, u64>,
    pub decos: &'a Decorations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Packing {
    Cbuffer,
    Structured,
}

#[derive(Default)]
struct Collector {
    path: Vec<u32>,
    out: BTreeMap<u32, Leaf>,
}

impl Collector {
    fn leaf(&mut self, offset: u32, ty: Word) {
        self.out.insert(
            offset,
            Leaf {
                path: self.path.clone(),
                ty,
            },
        );
    }

    fn nested<R>(&mut self, indices: &[u32], f: impl FnOnce(&mut Self) -> R) -> R {
        let len = self.path.len();
        self.path.extend(indices);
        let r = f(self);
        self.path.truncate(len);
        r
    }
}

impl Types<'_> {
    fn lit(&self, ty: Word, i: usize) -> u32 {
        self.defs[&ty].operands[i].unwrap_literal_bit32()
    }

    pub fn array_len(&self, ty: Word) -> Result<u32> {
        let len = self.defs[&ty].operands[1].unwrap_id_ref();
        let len = self
            .consts
            .get(&len)
            .context("array length is not a constant")?;
        u32::try_from(*len).context("array too long")
    }

    pub fn scalar_size(&self, ty: Word) -> Result<u32> {
        match self.defs[&ty].class.opcode {
            Op::TypeFloat | Op::TypeInt => {
                let bits = self.lit(ty, 0);
                ensure!(bits == 32, "only 32-bit scalars are supported");
                Ok(4)
            }
            Op::TypeBool => Ok(4),
            _ => bail!("not a scalar type"),
        }
    }

    fn vector(&self, ty: Word, offset: u32, c: &mut Collector) -> Result<u32> {
        let comp = self.defs[&ty].operands[0].unwrap_id_ref();
        let size = self.scalar_size(comp)?;
        let n = self.lit(ty, 1);
        for i in 0..n {
            c.nested(&[i], |c| c.leaf(offset + i * size, comp));
        }
        Ok(n * size)
    }

    pub fn leaves(&self, logical: Word, laid_out: Word) -> Result<BTreeMap<u32, Leaf>> {
        let mut c = Collector::default();
        self.walk(logical, laid_out, 0, None, &mut c)?;
        Ok(c.out)
    }

    pub fn member_element_leaves(&self, st: Word, n: u32) -> Result<BTreeMap<u32, Leaf>> {
        let arr = self.defs[&st].operands[n as usize].unwrap_id_ref();
        let elem = self.defs[&arr].operands[0].unwrap_id_ref();
        let mut c = Collector::default();
        self.walk(elem, elem, 0, self.matrix_decorations(st, n), &mut c)?;
        Ok(c.out)
    }

    pub fn packed_leaves(&self, ty: Word, matrices: MatrixLayout) -> Result<BTreeMap<u32, Leaf>> {
        let mut c = Collector::default();
        self.pack(ty, 0, Packing::Cbuffer, matrices, &mut c)?;
        Ok(c.out)
    }

    pub fn structured_leaves(
        &self,
        ty: Word,
        matrices: MatrixLayout,
    ) -> Result<(BTreeMap<u32, Leaf>, u32)> {
        let mut c = Collector::default();
        let size = self.pack(ty, 0, Packing::Structured, matrices, &mut c)?;
        Ok((c.out, size))
    }

    pub fn packed_size(&self, ty: Word, matrices: MatrixLayout) -> Result<u32> {
        self.pack(ty, 0, Packing::Cbuffer, matrices, &mut Collector::default())
    }

    fn pack(
        &self,
        ty: Word,
        offset: u32,
        packing: Packing,
        matrices: MatrixLayout,
        c: &mut Collector,
    ) -> Result<u32> {
        let t = &self.defs[&ty];
        let align = |size: u32| match packing {
            Packing::Cbuffer => size.next_multiple_of(16),
            Packing::Structured => size,
        };
        Ok(match t.class.opcode {
            Op::TypeFloat | Op::TypeInt | Op::TypeBool => {
                c.leaf(offset, ty);
                self.scalar_size(ty)?
            }
            Op::TypeVector => self.vector(ty, offset, c)?,
            Op::TypeMatrix => {
                let row = t.operands[0].unwrap_id_ref();
                let comp = self.defs[&row].operands[0].unwrap_id_ref();
                let size = self.scalar_size(comp)?;
                let (rows, cols) = (self.lit(ty, 1), self.lit(row, 1));
                let (major, minor) = match matrices {
                    MatrixLayout::RowMajor => (rows, cols),
                    MatrixLayout::ColumnMajor => (cols, rows),
                };
                let stride = align(minor * size);
                for r in 0..rows {
                    for col in 0..cols {
                        let (i, j) = match matrices {
                            MatrixLayout::RowMajor => (r, col),
                            MatrixLayout::ColumnMajor => (col, r),
                        };
                        c.nested(&[r, col], |c| c.leaf(offset + i * stride + j * size, comp));
                    }
                }
                (major - 1) * stride + minor * size
            }
            Op::TypeArray => {
                let elem = t.operands[0].unwrap_id_ref();
                let len = self.array_len(ty)?;
                let mut elem_size = 0u32;
                for k in 0..len {
                    let at = offset + k * align(elem_size);
                    elem_size = c.nested(&[k], |c| self.pack(elem, at, packing, matrices, c))?;
                }
                align(elem_size) * (len - 1) + elem_size
            }
            Op::TypeStruct => {
                let mut cur = 0u32;
                for (i, m) in t.operands.iter().enumerate() {
                    let m = m.unwrap_id_ref();
                    if packing == Packing::Cbuffer {
                        let starts_register = matches!(
                            self.defs[&m].class.opcode,
                            Op::TypeStruct | Op::TypeArray | Op::TypeMatrix
                        );
                        if starts_register || cur % 16 + self.simple_size(m)? > 16 {
                            cur = cur.next_multiple_of(16);
                        }
                    }
                    cur += c.nested(&[i as u32], |c| {
                        self.pack(m, offset + cur, packing, matrices, c)
                    })?;
                }
                cur
            }
            _ => match packing {
                Packing::Cbuffer => bail!("unsupported type in a provider return type"),
                Packing::Structured => bail!("unsupported type in a structured buffer element"),
            },
        })
    }

    fn simple_size(&self, ty: Word) -> Result<u32> {
        let t = &self.defs[&ty];
        Ok(match t.class.opcode {
            Op::TypeVector => self.lit(ty, 1) * self.scalar_size(t.operands[0].unwrap_id_ref())?,
            _ => self.scalar_size(ty)?,
        })
    }

    fn matrix_decorations(&self, st: Word, i: u32) -> Option<(u32, bool)> {
        let m = Some(i);
        let stride = self.decos.value(st, m, Decoration::MatrixStride)?;
        Some((stride, self.decos.has(st, m, Decoration::RowMajor)))
    }

    fn walk(
        &self,
        l: Word,
        d: Word,
        offset: u32,
        matrix: Option<(u32, bool)>,
        c: &mut Collector,
    ) -> Result<()> {
        let (lt, dt) = (&self.defs[&l], &self.defs[&d]);
        ensure!(
            lt.class.opcode == dt.class.opcode || lt.class.opcode == Op::TypeBool,
            "layout twin mismatch"
        );
        match lt.class.opcode {
            Op::TypeFloat | Op::TypeInt | Op::TypeBool => c.leaf(offset, l),
            Op::TypeVector => {
                self.vector(l, offset, c)?;
            }
            Op::TypeMatrix => {
                let (stride, row_major) = matrix.context("matrix without MatrixStride")?;
                let col = lt.operands[0].unwrap_id_ref();
                let comp = self.defs[&col].operands[0].unwrap_id_ref();
                let size = self.scalar_size(comp)?;
                for j in 0..self.lit(l, 1) {
                    for i in 0..self.lit(col, 1) {
                        let off = if row_major {
                            i * stride + j * size
                        } else {
                            j * stride + i * size
                        };
                        c.nested(&[j, i], |c| c.leaf(offset + off, comp));
                    }
                }
            }
            Op::TypeArray => {
                let stride = self
                    .decos
                    .value(d, None, Decoration::ArrayStride)
                    .context("array without ArrayStride")?;
                let len = self.array_len(l)?;
                let (le, de) = (
                    lt.operands[0].unwrap_id_ref(),
                    dt.operands[0].unwrap_id_ref(),
                );
                for k in 0..len {
                    c.nested(&[k], |c| self.walk(le, de, offset + k * stride, matrix, c))?;
                }
            }
            Op::TypeStruct => {
                ensure!(
                    lt.operands.len() == dt.operands.len(),
                    "layout twin mismatch"
                );
                for (i, (lm, dm)) in lt.operands.iter().zip(&dt.operands).enumerate() {
                    let off = self
                        .decos
                        .value(d, Some(i as u32), Decoration::Offset)
                        .context("struct member has no Offset")?;
                    let mat = self.matrix_decorations(d, i as u32);
                    c.nested(&[i as u32], |c| {
                        self.walk(lm.unwrap_id_ref(), dm.unwrap_id_ref(), offset + off, mat, c)
                    })?;
                }
            }
            _ => bail!("unsupported type in a cbuffer"),
        }
        Ok(())
    }
}
