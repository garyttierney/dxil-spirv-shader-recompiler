use std::{
    env,
    path::{Path, PathBuf},
};

/// The static libraries to link, each before those it depends on.
const LIBRARIES: &[&str] = &[
    "dxil-spirv-c-static",
    "dxil-converter",
    "spirv-module",
    "glslang-spirv-builder",
    "llvm-bc",
    "bc-decoder",
    "dxbc-spirv",
    "dxil-debug",
    "dxil-utils",
    "SPIRV-Tools-link",
    "SPIRV-Tools-opt",
    // The SPIRV-Tools-static target.
    "SPIRV-Tools",
];

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let dxil_spirv = manifest.join("dxil-spirv");
    assert!(
        dxil_spirv.join("CMakeLists.txt").exists(),
        "the dxil-spirv submodule is missing: run `git submodule update --init --recursive`"
    );
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=dxil-spirv");
    println!("cargo:rerun-if-changed=src/bridge.cc");
    println!("cargo:rerun-if-changed=include/bridge.h");

    let dst = cmake::Config::new(&manifest)
        .profile("Release")
        .build_target("dxil-spirv-shader-recompiler-native")
        .build();

    cxx_build::bridge("src/lib.rs")
        .file("src/bridge.cc")
        .include(&dxil_spirv)
        .include(dxil_spirv.join("third_party/SPIRV-Tools/include"))
        .std("c++17")
        .compile("dxil-spirv-shader-recompiler-bridge");

    let build = dst.join("build");
    for library in LIBRARIES {
        let file = format!("lib{library}.a");
        let dir = find(&build, &file)
            .unwrap_or_else(|| panic!("{file} wasn't built under {}", build.display()));
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={library}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn find(dir: &Path, file: &str) -> Option<PathBuf> {
    if dir.join(file).is_file() {
        return Some(dir.to_path_buf());
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|t| t.is_dir()))
        .find_map(|entry| find(&entry.path(), file))
}
