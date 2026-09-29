#include "dxil-spirv-shader-recompiler-sys/include/bridge.h"
#include "dxil-spirv-shader-recompiler-sys/src/lib.rs.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dxil_spirv_c.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"

namespace dxil_spirv_shader_recompiler {

namespace {

constexpr spv_target_env TARGET_ENV = SPV_ENV_UNIVERSAL_1_6;

dxil_spv_vulkan_binding unbound() { return {}; }

dxil_spv_vulkan_binding at_register(const dxil_spv_d3d_binding &d3d) {
  dxil_spv_vulkan_binding binding = unbound();
  binding.set = d3d.register_space;
  binding.binding = d3d.register_index;
  return binding;
}

dxil_spv_bool srv(void *, const dxil_spv_d3d_binding *d3d,
                  dxil_spv_srv_vulkan_binding *vk) {
  vk->buffer_binding = at_register(*d3d);
  vk->offset_binding = unbound();
  return DXIL_SPV_TRUE;
}

dxil_spv_bool sampler(void *, const dxil_spv_d3d_binding *d3d,
                      dxil_spv_vulkan_binding *vk) {
  *vk = at_register(*d3d);
  return DXIL_SPV_TRUE;
}

dxil_spv_bool cbv(void *, const dxil_spv_d3d_binding *d3d,
                  dxil_spv_cbv_vulkan_binding *vk) {
  vk->vulkan.uniform_binding = at_register(*d3d);
  vk->push_constant = DXIL_SPV_FALSE;
  return DXIL_SPV_TRUE;
}

dxil_spv_bool uav(void *, const dxil_spv_uav_d3d_binding *d3d,
                  dxil_spv_uav_vulkan_binding *vk) {
  vk->buffer_binding = at_register(d3d->d3d_binding);
  vk->counter_binding = unbound();
  vk->offset_binding = unbound();
  return DXIL_SPV_TRUE;
}

// Reads each vertex input from its Direct3D register, recording its semantic.
dxil_spv_bool vertex_input(void *inputs, const dxil_spv_d3d_vertex_input *d3d,
                           dxil_spv_vulkan_vertex_input *vk) {
  vk->location = d3d->start_row;
  static_cast<rust::Vec<VertexInput> *>(inputs)->push_back(VertexInput{
      rust::String(d3d->semantic), d3d->semantic_index, d3d->start_row});
  return DXIL_SPV_TRUE;
}

void check(dxil_spv_result result, const char *what) {
  if (result != DXIL_SPV_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed");
  }
}

struct BlobFree {
  void operator()(dxil_spv_parsed_blob blob) const {
    dxil_spv_parsed_blob_free(blob);
  }
};
using Blob =
    std::unique_ptr<std::remove_pointer_t<dxil_spv_parsed_blob>, BlobFree>;

struct ConverterFree {
  void operator()(dxil_spv_converter converter) const {
    dxil_spv_converter_free(converter);
  }
};
using Converter =
    std::unique_ptr<std::remove_pointer_t<dxil_spv_converter>, ConverterFree>;

struct Messages {
  std::string text;

  spvtools::MessageConsumer consumer() {
    return [this](spv_message_level_t, const char *,
                  const spv_position_t &position, const char *message) {
      text += "word " + std::to_string(position.index) + ": " + message + "\n";
    };
  }

  [[noreturn]] void fail(const char *what) const {
    throw std::runtime_error(text.empty() ? std::string(what)
                                          : std::string(what) + ":\n" + text);
  }
};

rust::Vec<uint32_t> to_rust(const std::vector<uint32_t> &words) {
  rust::Vec<uint32_t> out;
  out.reserve(words.size());
  for (uint32_t word : words) {
    out.push_back(word);
  }
  return out;
}

} // namespace

Converted convert_dxil(rust::Slice<const uint8_t> dxbc) {
  dxil_spv_parsed_blob parsed = nullptr;
  check(dxil_spv_parse_dxil_blob(dxbc.data(), dxbc.size(), &parsed),
        "parsing the DXIL");
  Blob blob(parsed);

  dxil_spv_parsed_blob reflection = nullptr;
  dxil_spv_result result = dxil_spv_parse_reflection_dxil_blob(
      dxbc.data(), dxbc.size(), &reflection);
  if (result != DXIL_SPV_SUCCESS && result != DXIL_SPV_ERROR_NO_DATA) {
    check(result, "parsing the DXIL's reflection");
  }
  Blob reflected(result == DXIL_SPV_SUCCESS ? reflection : nullptr);

  dxil_spv_converter created = nullptr;
  check(dxil_spv_create_converter_with_reflection(blob.get(), reflected.get(),
                                                  &created),
        "creating the converter");
  Converter converter(created);
  dxil_spv_converter_set_srv_remapper(converter.get(), srv, nullptr);
  dxil_spv_converter_set_sampler_remapper(converter.get(), sampler, nullptr);
  dxil_spv_converter_set_cbv_remapper(converter.get(), cbv, nullptr);
  dxil_spv_converter_set_uav_remapper(converter.get(), uav, nullptr);
  rust::Vec<VertexInput> vertex_inputs;
  dxil_spv_converter_set_vertex_input_remapper(converter.get(), vertex_input,
                                               &vertex_inputs);
  check(dxil_spv_converter_run(converter.get()), "converting the DXIL");

  dxil_spv_compiled_spirv spirv = {};
  check(dxil_spv_converter_get_compiled_spirv(converter.get(), &spirv),
        "reading the converted SPIR-V");
  if (spirv.data == nullptr || spirv.size == 0) {
    throw std::runtime_error("dxil-spirv produced no SPIR-V");
  }
  Converted out;
  out.spirv.reserve(spirv.size);
  const auto *bytes = static_cast<const uint8_t *>(spirv.data);
  for (size_t i = 0; i < spirv.size; i++) {
    out.spirv.push_back(bytes[i]);
  }
  out.vertex_inputs = std::move(vertex_inputs);
  return out;
}

rust::Vec<uint32_t> link(rust::Slice<const uint32_t> words,
                         rust::Slice<const size_t> lengths) {
  std::vector<std::vector<uint32_t>> modules;
  size_t at = 0;
  for (size_t length : lengths) {
    if (at + length > words.size()) {
      throw std::runtime_error("module lengths exceed the words given");
    }
    modules.emplace_back(words.data() + at, words.data() + at + length);
    at += length;
  }

  Messages messages;
  spvtools::Context context(TARGET_ENV);
  context.SetMessageConsumer(messages.consumer());
  std::vector<uint32_t> linked;
  if (spvtools::Link(context, modules, &linked) != SPV_SUCCESS) {
    messages.fail("linking failed");
  }
  return to_rust(linked);
}

rust::Vec<uint32_t> optimize(rust::Slice<const uint32_t> words,
                             rust::Slice<const rust::String> flags) {
  Messages messages;
  spvtools::Optimizer optimizer(TARGET_ENV);
  optimizer.SetMessageConsumer(messages.consumer());
  std::vector<std::string> passes;
  for (const rust::String &flag : flags) {
    passes.emplace_back(flag);
  }
  if (!optimizer.RegisterPassesFromFlags(passes)) {
    messages.fail("unknown spirv-opt flags");
  }
  std::vector<uint32_t> optimized;
  if (!optimizer.Run(words.data(), words.size(), &optimized)) {
    messages.fail("optimizing failed");
  }
  return to_rust(optimized);
}

} // namespace dxil_spirv_shader_recompiler
