#pragma once

#include <cstddef>
#include <cstdint>

#include "rust/cxx.h"

namespace dxil_spirv_shader_recompiler {

struct Converted;

// Converts a DXIL shader in a DXBC container to SPIR-V with dxil-spirv, binding each resource
// at its register and space, and each vertex input at its register.
Converted convert_dxil(rust::Slice<const uint8_t> dxbc);

// Links SPIR-V modules via spirv-link
rust::Vec<uint32_t> link(rust::Slice<const uint32_t> words, rust::Slice<const size_t> lengths);

// Runs spirv-opt's passes, given as its command-line flags, on a module.
rust::Vec<uint32_t> optimize(rust::Slice<const uint32_t> words,
                             rust::Slice<const rust::String> flags);

} // namespace dxil_spirv_shader_recompiler
