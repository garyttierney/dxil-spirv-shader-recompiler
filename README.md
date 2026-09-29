# dxil-spirv-shader-recompiler

Converts DXIL shaders (`.vpo`/`.ppo` DXBC containers) to SPIR-V with upstream
[dxil-spirv](https://github.com/HansKristian-Work/dxil-spirv), then rewrites them so every
resource they read comes from a function in SPIR-V library instead of a binding. The rewritten
shader declares those functions as imports, and we relink it to produce a shader program targeting a different binding table.
