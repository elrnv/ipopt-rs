//   Copyright 2018 Egor Larionov
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

use curl::easy::Easy;
use flate2::read::GzDecoder;
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};

/**
 * # Goals
 *
 * 1. Make this library build without external dependencies on all platforms. This allows us to run
 *    this library through a third party CI easily.
 * 2. Use all available local dependencies to build this library if any to make the build process
 *    as fast as possible. This also allows developers with optimized/personalized blas
 *    environments to use their configuration without needing to customize this build process.
 *
 *
 * This build file is responsible for:
 *
 * 1. Finding or building Ipopt.
 * 2. Building the custom ipopt_cnlp interface linking to the found/buld Ipopt in step 1.
 * 3. Building this Rust library.
 *
 *
 * # Build Process
 *
 * 1. To satisfy goal 2. first use pkg-config to find any preinstalled ipopt libs.
 * 2. If nothing is found in pkg-config, we try to build ipopt from source on supoorted platforms
 *    with MKL (currently only macOS) to maintain the most optimal build.
 * 3. To satisfy goal 1. if the steps above fail, we will download the prebuild libraries from
 *    https://github.com/JuliaOpt/IpoptBuilder/releases which is already referenced on the official
 *    Ipopt webpage as a source of prebuilt binaries.
 */
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};
use tar::Archive;

const LIBRARY: &str = "ipopt";
const SOURCE_URL: &str = "https://www.coin-or.org/download/source/Ipopt";
const VERSION: &str = "3.12.10";
const MIN_VERSION: &str = "3.11.9";
const BINARY_DL_URL: &str = "https://github.com/JuliaOpt/IpoptBuilder/releases/download/";
const SOURCE_MD5: &str = "ee250ece251a82dc2580efa51f79d758";
const SOURCE_SHA1: &str = "5eb1aefb2f9acfd8b1b5838370528ac1d73787d6";

#[cfg(target_os = "macos")]
mod platform {
    // For some reason I couldn't build and link to Ipopt as a static lib on macos, so this is here.
    pub static BUILD_FLAGS: [&str; 1] = ["--disable-shared"];
    pub static LIB_EXT: &str = "a";
    pub static DYNAMIC_LIB_EXT: &str = "dylib";
    pub static BINARY_SUFFIX: &str = "x86_64-apple-darwin14.tar.gz";
    pub static BINARY_MD5: &str = "59825a6b7e40929ff2c88fb23dc82b7c";
    pub static BINARY_SHA1: &str = "a24f1def1ce9fc33393779b574cea9bfb4765c4f";
}

#[cfg(target_os = "linux")]
mod platform {
    pub static BUILD_FLAGS: [&str; 2] = ["--disable-shared", "--with-pic"];
    pub static LIB_EXT: &str = "a";
    pub static DYNAMIC_LIB_EXT: &str = "so";
    pub static BINARY_SUFFIX: &str = "x86_64-linux-gnu-gcc8.tar.gz";
    pub static BINARY_MD5: &str = "9c406cb1b54918b56945548e64b8e9ca";
    pub static BINARY_SHA1: &str = "a940b1f70021ddbd057643a056b61228d68f26e6";
}

#[cfg(target_family = "unix")]
mod family {
    pub static LIB_MAJ_VER: &str = "1";
    pub static LIB_MIN_VER: &str = "10.10";
}

#[cfg(target_os = "windows")]
mod platform {
    pub static BUILD_FLAGS: [&str;1] = [""];
    pub static LIB_EXT: &str = "dll";
    pub static DYNAMIC_LIB_EXT: &str = "dll";
    pub static BINARY_SUFFIX: &str = "x86_64-w64-mingw32-gcc8.tar.gz";
}

#[cfg(target_os = "windows")]
mod family {
}

use crate::platform::*;
use crate::family::*;

lazy_static! {
    static ref BINARY_NAME: String = format!(
        "IpoptBuilder.v{ver}.{suffix}",
        ver = VERSION,
        suffix = BINARY_SUFFIX
    );
    static ref BINARY_URL: String = format!(
        "{dl}v{ver}-1-static/{name}",
        dl = BINARY_DL_URL,
        ver = VERSION,
        name = BINARY_NAME.as_str()
    );
}

fn main() {
    let mut msg = String::from("\n\n");

    // Try to find Ipopt preinstalled.
    match try_pkg_config() {
        Ok(link_info) => {
            link(build_cnlp(&link_info.include_paths), link_info)
                .expect("Failed to create bindings for Ipopt library.");
            return;
        }
        Err(err) => {
            msg.push_str(&format!("Failed to find Ipopt using pkg-config: {:?}\n\n", err));
        }
    }

    // Check if Ipopt has been installed as a local system lib, but for some reason pkg-config is
    // missing.
    match try_system_install() {
        Ok(link_info) => {
            link(build_cnlp(&link_info.include_paths), link_info)
                .expect("Failed to create bindings for Ipopt library.");
            return;
        }
        Err(err) => {
            msg.push_str(&format!("Failed to find Ipopt installed on the system: {:?}\n\n", err));
        }
    }

    match build_and_install_ipopt() {
        Ok(link_info) => {
            link(build_cnlp(&link_info.include_paths), link_info)
                .expect("Failed to create bindings for Ipopt library.");
            return;
        }
        Err(err) => {
            msg.push_str(&format!("Failed to build Ipopt from source: {:?}\n\n", err));
        }
    }

    match download_and_install_prebuilt_binary() {
        Ok(link_info) => {
            link(build_cnlp(&link_info.include_paths), link_info)
                .expect("Failed to create bindings for Ipopt library.");
            return;
        }
        Err(err) => {
            msg.push_str(&format!("Failed to download and install Ipopt binaries: {:?}\n\n", err));
        }
    }

    panic!(msg);
}

#[derive(Clone, Debug, PartialEq)]
enum Error {
    SystemLibNotFound,
    PkgConfigNotFound,
    MKLInstallNotFound,
    DownloadFailure { response_code: u32, url: String },
    UrlFailure,
    UnsupportedPlatform,
    IOError,
    HashMismatch,
}

impl From<std::io::Error> for Error {
    fn from(_: std::io::Error) -> Error {
        Error::IOError
    }
}

impl From<curl::Error> for Error {
    fn from(_: curl::Error) -> Error {
        Error::UrlFailure
    }
}


// The following convenience functions produce the correct library filename for the corresponding
// platform when downloading the binaries. We always download dynamic libs.

fn library_name() -> String {
    format!("lib{}.{}", LIBRARY, DYNAMIC_LIB_EXT)
}

#[cfg(target_family = "windows")]
fn versioned_library_name() -> String {
    // No versioning in filenames on Windows.
    format!("lib{}.{}", LIBRARY, LIB_EXT)
}

#[cfg(target_family = "unix")]
fn versioned_library_name() -> String {
    if cfg!(target_os = "macos") {
        format!("lib{}.{}.{}.{}", LIBRARY, LIB_MAJ_VER, LIB_MIN_VER, DYNAMIC_LIB_EXT)
    } else {
        format!("lib{}.{}.{}.{}", LIBRARY, DYNAMIC_LIB_EXT, LIB_MAJ_VER, LIB_MIN_VER)
    }
}

#[cfg(target_family = "unix")]
fn major_versioned_library_name() -> String {
    if cfg!(target_os = "macos") {
        format!("lib{}.{}.{}", LIBRARY, LIB_MAJ_VER, DYNAMIC_LIB_EXT)
    } else {
        format!("lib{}.{}.{}", LIBRARY, DYNAMIC_LIB_EXT, LIB_MAJ_VER)
    }
}

// Try to find ipopt install path from pkg_config.
fn try_pkg_config() -> Result<LinkInfo, Error> {
    match pkg_config::Config::new()
        .atleast_version(MIN_VERSION)
        .cargo_metadata(false) // We are linking to cnlp, not to the rust lib
        .probe(LIBRARY)
    {
        Ok(lib) => {
            let lib_type = check_pkg_config_lib_type(LIBRARY, &lib);
            let link_info = LinkInfo {
                libs: lib.libs.iter().cloned().map(|lib| (lib_type, lib)).collect(),
                search_paths: lib.link_paths.clone(),
                include_paths: lib.include_paths.clone(),
            };

            save_link_info(&link_info)?;

            Ok(link_info)
        },
        Err(_) => Err(Error::PkgConfigNotFound),
    }
}

// A vector of system lib/include path pairs to search for libraries in.
fn system_install_paths() -> Vec<(PathBuf, PathBuf)> {
    vec![
        ("/usr/lib", "/usr/include"),
        ("/usr/local/lib", "/usr/local/include"),
        ("/usr/lib/x86_64-linux-gnu", "/usr/include/x86_64-linux-gnu"),
    ]
        .into_iter()
        .map(|(l,i)| (PathBuf::from(l), PathBuf::from(i)))
        .collect()
}

// Just check system libs. There may be something there.
fn try_system_install() -> Result<LinkInfo, Error> {
    // Check standard prefixes
    for (lib, include) in system_install_paths().into_iter() {
        // Try to find a Dynamic lib. We don't try to find static libs here, because we don't know
        // how they should be linked without something like pkg-config.
        
        let lib_ipopt = lib.join(major_versioned_library_name());
        let include_ipopt = include.join("coin").join("IpIpoptApplication.hpp");

        if lib_ipopt.exists() && include_ipopt.exists() {
            let link_info = LinkInfo {
                libs: vec![(LibType::Dynamic, "ipopt".to_string())],
                search_paths: vec![lib],
                include_paths: vec![include],
            };
            save_link_info(&link_info)?;
            return Ok(link_info);
        }
    }
    Err(Error::SystemLibNotFound)
}

/// Download the ipopt prebuilt binary from JuliaOpt and install it.
fn download_and_install_prebuilt_binary() -> Result<LinkInfo, Error> {
    let file_name = BINARY_NAME.clone();

    // Extract the filename from the URL
    let mut base_name = file_name.clone();
    remove_suffix(&mut base_name, ".tar.gz");
    dbg!(&base_name);

    // Create download directory if it doesn't yet exist
    let crate_dir = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = crate_dir.join("target");
    dbg!(&target_dir);
    if !target_dir.exists() {
        fs::create_dir(&target_dir).unwrap();
    }

    let download_dir = target_dir.join(format!("ipopt-{}-binaries", VERSION));
    dbg!(&download_dir);
    if !download_dir.exists() {
        fs::create_dir(&download_dir).unwrap();
    }

    // Download and extract the tarball if the library isn't there.
    let unpacked_dir = download_dir.join(base_name);
    let output = PathBuf::from(&env::var("OUT_DIR").unwrap());
    let install_dir = output.clone();
    let library_file = versioned_library_name();
    let lib_dir = install_dir.join("lib");
    let library_path = lib_dir.join(&library_file);
    dbg!(&library_path);

    if library_path.exists() {
        // Nothing to be done, library is already installed
        return Ok(load_link_info()?);
    }

    // On unix make sure all artifacts are removed to cleanup the environment
    if cfg!(unix) {
        fs::remove_file(lib_dir.join(major_versioned_library_name())).ok();
        fs::remove_file(lib_dir.join(library_name())).ok();
    }

    // Build destination path
    let tarball_path = download_dir.join(file_name);
    dbg!(&tarball_path);

    if !tarball_path.exists() {
        download_tarball(&tarball_path, &BINARY_URL, BINARY_MD5, BINARY_SHA1)?;
    }

    // Remove previously extracted files if any
    dbg!(&unpacked_dir);
    fs::remove_dir_all(&unpacked_dir).ok();
    
    extract_tarball(tarball_path, &unpacked_dir);

    // Copy lib
    if !lib_dir.exists() {
        fs::create_dir(&lib_dir)?;
    }

    let downloaded_lib_path = unpacked_dir.join("lib").join(&library_file);

    // Make links (on unix only)
    if cfg!(unix) {
        use std::os::unix::fs::symlink;
        eprintln!("Creating symlinks for dynamic libraries...");
        symlink(&library_path, lib_dir.join(major_versioned_library_name()))?;
        symlink(&library_path, lib_dir.join(library_name()))?;
    }

    // Copy headers
    let install_include_dir = install_dir.join("include").join("coin");
    if !install_include_dir.exists() {
        fs::create_dir_all(&install_include_dir)?;
    }

    let include_dir = unpacked_dir.join("include").join("coin");
    for entry in fs::read_dir(include_dir)? {
        let file = entry?;
        fs::copy(file.path(), install_include_dir.join(file.file_name())).unwrap();
    }

    // Copy the actual library last because we use its existence to check if everything above has
    // already been done correctly.
    eprintln!("Copying {} to {}", downloaded_lib_path.display(), library_path.display());
    fs::copy(
        downloaded_lib_path,
        &library_path,
    )?;

    let link_info = LinkInfo {
        libs: vec![(LibType::Dynamic, "ipopt".to_string())],
        search_paths: vec![lib_dir],
        include_paths: vec![install_dir.join("include")]
    };

    save_link_info(&link_info)?;

    Ok(link_info)
}

fn link_info_path() -> PathBuf {
    let output = PathBuf::from(&env::var("OUT_DIR").unwrap());
    output.join("ipopt_config.json")
}

// Create Pkg-Config-like file to remember how the ipopt library was installed so we don't have to
// build it every time.
fn save_link_info(info: &LinkInfo) -> Result<(), Error> {
    let json = serde_json::to_string(info).expect("Failed to serialize link info.");
    let mut file = fs::File::create(link_info_path())?;
    write!(&mut file, "{}", json)?;
    Ok(())
}

// Read the Pkg-Config-like file to check how the ipopt library was installed so we don't have to
// build it every time.
fn load_link_info() -> Result<LinkInfo, Error> {
    use std::io::Read;
    let mut file = fs::File::open(link_info_path())?;
    let mut info = String::new();
    file.read_to_string(&mut info)?;
    Ok(serde_json::from_str(&info).expect("Failed to deserialize link info."))
}


fn check_tarball_hashes(tarball_path: &Path, md5: &str, sha1: &str) -> Result<(), Error> {
    use std::io::Read;
    use crypto::digest::Digest;

    {
        let mut f = File::open(tarball_path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let mut hasher = crypto::md5::Md5::new();
        hasher.input(&buffer);
        let dl_hex = hasher.result_str();
        if md5 != dl_hex {
            return Err(Error::HashMismatch);
        }
    }
    {
        let mut f = File::open(tarball_path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let mut hasher = crypto::sha1::Sha1::new();
        hasher.input(&buffer);
        let dl_hex = hasher.result_str();
        if sha1 != dl_hex {
            return Err(Error::HashMismatch);
        }
    }

    Ok(())
}

/// Build the CNLP interface.
fn build_cnlp(ipopt_include_paths: &[PathBuf]) -> PathBuf {
    let mut ipopt_include_dirs = String::new();
    for path in ipopt_include_paths.iter() {
        ipopt_include_dirs.push_str(path.to_str().unwrap());
        ipopt_include_dirs.push(' ');
    }
    cmake::Config::new("cnlp")
        .define("Ipopt_INCLUDE_DIRS:STRING", ipopt_include_dirs)
        .build()
}

/// Link ipopt-sys to our cnlp api. If ipopt is provided as a dynamic lib, we need to link it here.
/// The `dynamic` flags specifies if ipopt is being linked dynamically.
fn link(cnlp_install_path: PathBuf, link_info: LinkInfo) -> Result<(), Error> {
    // Link to cnlp
    println!(
        "cargo:rustc-link-search=native={}",
        cnlp_install_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ipopt_cnlp");

    // Add metadata about where this dependency is being installed.
    println!("cargo:root={}", cnlp_install_path.display());
    println!("cargo:libdir={}", cnlp_install_path.join("lib").display());
    println!("cargo:include={}", cnlp_install_path.join("include").display());

    // Order is important here. The most core libs should appear last.
    for path in link_info.search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for (dep_type, lib) in link_info.libs {
        let lib_type_str = match dep_type {
            LibType::Dynamic => "dylib",
            LibType::Static => "static",
        };
        println!("cargo:rustc-link-lib={}={}", lib_type_str, lib);
    }

    // Add the C++ standard lib for linking against CNLP.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Generate raw bindings to CNLP interface
    let c_api_header = cnlp_install_path.join("include").join("c_api.h");

    let bindings = bindgen::builder()
        .header(c_api_header.to_str().unwrap())
        .generate()
        .expect("Unable to generate bindings!");

    let output = PathBuf::from(&env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(output.join("ipopt_cnlp.rs"))
        .expect("Couldn't write bindings!");

    Ok(())
}

/// Download a tarball if it doesn't already exist.
fn download_tarball(tarball_path: &Path, binary_url: &str, md5: &str, sha1: &str) -> Result<(), Error> {
    if !tarball_path.exists() {
        eprintln!("Tarball doesn't exist, downloading...");
        let f = File::create(tarball_path).unwrap();
        let mut writer = BufWriter::new(f);
        let mut easy = Easy::new();
        easy.follow_location(true)?;
        easy.url(binary_url).unwrap();
        easy.write_function(move |data| Ok(writer.write(data).unwrap()))
            .unwrap();
        easy.perform().unwrap();

        let response_code = easy.response_code().unwrap();
        if response_code != 200 {
            return Err(Error::DownloadFailure {
                response_code,
                url: binary_url.to_string(),
            });
        } else {
            eprintln!("Download successful!");
        }
    }

    check_tarball_hashes(&tarball_path, md5, sha1)?;

    Ok(())
}

/// Build Ipopt statically linked to MKL if possible. Return the path to the ipopt library.
fn build_and_install_ipopt() -> Result<LinkInfo, Error> {
    // Compile ipopt from source
    // Build URL to download from
    let binary_url = format!("{}/Ipopt-{}.tgz", SOURCE_URL, VERSION);
    dbg!(&binary_url);

    // Extract the filename from the URL
    let file_name = binary_url.split("/").last().unwrap();
    let mut base_name = file_name.to_string();
    remove_suffix(&mut base_name, ".tgz");
    dbg!(&base_name);

    // Create download directory if it doesn't yet exist
    let crate_dir = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = crate_dir.join("target");
    dbg!(&target_dir);
    if !target_dir.exists() {
        fs::create_dir(&target_dir).unwrap();
    }

    let download_dir = target_dir.join(format!("ipopt-{}-source", VERSION));
    dbg!(&download_dir);
    if !download_dir.exists() {
        fs::create_dir(&download_dir).unwrap();
    }

    // Download, extract and compile the tarball if the library isn't there.
    let unpacked_dir = download_dir.join(base_name);
    let output = PathBuf::from(&env::var("OUT_DIR").unwrap());
    let install_dir = output.clone();
    let library_file = format!("lib{}.{}", LIBRARY, LIB_EXT);
    let library_path = install_dir.join("lib").join(&library_file);
    if library_path.exists() {
        // Library is already installed, retrieve link info and return.
        return Ok(load_link_info()?);
    }

    // Build destination path
    let tarball_path = download_dir.join(file_name);
    dbg!(&tarball_path);

    download_tarball(&tarball_path, &binary_url, SOURCE_MD5, SOURCE_SHA1)?;

    // Remove previously extracted files if any
    dbg!(&unpacked_dir);
    fs::remove_dir_all(&unpacked_dir).ok();

    extract_tarball(tarball_path, &download_dir);

    // Configure and compile
    // We shall compile ipopt in the same mode we build the sys library. This will allow users
    // to debug the internals of ipopt more easily.
    let debug: bool = env::var("DEBUG").unwrap().parse().unwrap();
    dbg!(debug);

    let build_dir = unpacked_dir
        .join("build")
        .join(if debug { "debug" } else { "release" });

    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).unwrap();
    }

    // Remember project root directory
    let proj_root_dir = env::current_dir().unwrap();
    env::set_current_dir(build_dir).unwrap();
    
    // Build a static lib for ipopt.
    let res = build_with_mkl(&install_dir, debug).or_else(|_|
        build_with_default_blas(&install_dir, debug));

    // Restore current directory
    env::set_current_dir(proj_root_dir).unwrap();

    let libs_info = res?; // Propagate any errors after we have restored the current dir.

    save_link_info(&libs_info)?;

    Ok(libs_info)
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
enum LibType {
    Dynamic,
    Static,
}

#[derive(Serialize, Deserialize, Debug)]
struct LinkInfo {
    /// Libs to link in rustc.
    libs: Vec<(LibType, String)>,
    /// Search paths for the specified libs.
    search_paths: Vec<PathBuf>,
    /// Include directories.
    include_paths: Vec<PathBuf>,
}

// Build Ipopt static lib with MKL in the current directory.
fn build_with_mkl(install_dir: &Path, debug: bool) -> Result<LinkInfo, Error> {
    let mkl_libs = ["mkl_intel_lp64", "mkl_tbb_thread", "mkl_core"];

    // Look for intel MKL and link to its libraries if found.
    let mkl_root = env::var("MKLROOT");
    dbg!(&mkl_root);
    let mkl_libs_path = if let Ok(mkl_root) = mkl_root {
        PathBuf::from(mkl_root).join("lib")
    } else {
        let opt_path = PathBuf::from("/opt/intel/mkl/lib");
        if opt_path.exists() { // directory exists
            opt_path
        } else {
            let usr_lib_path = PathBuf::from("/usr/lib/x86_64-linux-gnu");
            if mkl_libs.iter().all(|lib| usr_lib_path.join(format!("lib{}.a", lib)).exists()) {
                usr_lib_path
            } else {
                PathBuf::from("NOTFOUND")
            }
        }
    };

    dbg!(&mkl_libs_path);

    let mut link_libs = vec![(LibType::Static, "ipopt".to_string())];

    let blas = {
        if !mkl_libs_path.exists() {
            return Err(Error::MKLInstallNotFound);
        } else {
            let lib_prefix = format!("{}/lib", mkl_libs_path.display());
            let aux_libs = format!("-L{mkl} -ltbb -lpthread -lm -ldl", mkl = mkl_libs_path.display());
            let mkl_libs_str = format!("{lib}{l1}.a {lib}{l2}.a {lib}{l3}.a", 
                    lib = lib_prefix,
                    l1 = mkl_libs[0],
                    l2 = mkl_libs[1],
                    l3 = mkl_libs[2])
                ;
            if cfg!(target_os = "macos") {
                format!( "--with-blas={mkl} {aux} -lc++", mkl=mkl_libs_str, aux=aux_libs)
            } else if cfg!(target_os = "linux") {
                // Only forward the libs to cnlp on linux because we build ipopt statically here.
                let mkl_group = format!("-Wl,--start-group {} -Wl,--end-group", mkl_libs_str);
                let all_libs = format!("{mkl} {aux} -lstdc++", mkl=mkl_group, aux=aux_libs);
                format!("--with-blas={}", all_libs)
            } else {
                // Currently only support building Ipopt with MKL on macOS.
                return Err(Error::UnsupportedPlatform);
            }
        }
    };

    run(env::current_dir()?.parent().unwrap().parent().unwrap()
        .join("configure").to_str().unwrap(), |cmd| {
        let cmd = cmd
            .arg(format!("--prefix={}", install_dir.display()))
            .args(&BUILD_FLAGS)
            .arg(blas.clone());

        if debug {
            cmd.arg(format!("--enable-debug-ipopt"))
        } else {
            cmd
        }
    });

    run("make", |cmd| cmd.arg(format!("-j{}", num_cpus::get())));
    //run("make", |cmd| cmd.arg("test")); // Ensure everything is working
    run("make", |cmd| cmd.arg("install")); // Install to install_dir

    if cfg!(unix) {
        // Strip redundant modules from the archive. This is an Ipopt artifact.
        run("ar", |cmd| cmd
            .arg("-d")
            .arg(format!("{}/lib/libipopt.a", install_dir.display()))
            .arg("libmkl_intel_lp64.a")
            .arg("libmkl_tbb_thread.a")
            .arg("libmkl_core.a")
            .arg("libmkl_intel_lp64.a")
            .arg("libmkl_tbb_thread.a")
            .arg("libmkl_core.a")
            .arg("libmkl_intel_lp64.a")
            .arg("libmkl_tbb_thread.a")
            .arg("libmkl_core.a")
            .arg("lt1-libmkl_intel_lp64.a")
            .arg("lt2-libmkl_tbb_thread.a")
            .arg("lt3-libmkl_core.a"));
    }

    link_libs.push((LibType::Static, mkl_libs[0].to_string()));
    link_libs.push((LibType::Static, mkl_libs[1].to_string()));
    link_libs.push((LibType::Static, mkl_libs[2].to_string()));
    link_libs.push((LibType::Dynamic, "tbb".to_string()));

    Ok(LinkInfo {
        libs: link_libs,
        search_paths: vec![mkl_libs_path, install_dir.join("lib")],
        include_paths: vec![install_dir.join("include")],
    })
}

fn check_pkg_config_lib_type(lib_name: &str, lib: &pkg_config::Library) -> LibType {
    let mut lib_type = LibType::Dynamic;

    if cfg!(target_os = "linux") {
        // Check if there is a static library, in which case link to that. Otherwise fallback
        // to dynamic linking.
        let static_lib = format!("lib{}.a", lib_name);
        for path in lib.link_paths.iter() {
            if path.join(&static_lib).exists() {
                lib_type = LibType::Static;
            }
        }
    }
    lib_type
}

// TODO: This should be handled with an external *-sys crate.
// library is the name of the library to search for and header is an associated header to determine
// that the include path is also valid.
fn find_linux_lib(library: &str, header: &str) -> Result<LinkInfo, Error> {
    // Try with pkg-config
    if let Ok(lib) = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe(library)
    {
        dbg!(&lib);

        let lib_type = check_pkg_config_lib_type(library, &lib);

        let link_info = LinkInfo {
            libs: lib.libs.iter().cloned().map(|lib| (lib_type, lib)).collect(),
            search_paths: lib.link_paths.clone(),
            include_paths: lib.include_paths.clone(),
        };

        return Ok(link_info);
    }

    eprintln!("Couldn't find {} with pkg-config", library);

    // Try searching in system paths
    for (lib, include) in system_install_paths().into_iter() {
        // Try to find a Dynamic lib. We don't try to find static libs here, because we don't know
        // how they should be linked without something like pkg-config.
        
        let lib_path = lib.join(format!("lib{}.so", library));
        let include_path = include.join(header);
        eprintln!("Checking existence of {} and {}", lib_path.to_str().unwrap(), include_path.to_str().unwrap());
        dbg!(lib_path.exists());
        dbg!(include_path.exists());

        if lib_path.exists() && include_path.exists() {
            let link_info = LinkInfo {
                libs: vec![(LibType::Dynamic, library.to_string())],
                search_paths: vec![lib],
                include_paths: vec![include],
            };
            return Ok(link_info);
        }
    }
    Err(Error::SystemLibNotFound)
}

// Build Ipopt static lib with Default libs.
fn build_with_default_blas(install_dir: &Path, debug: bool) -> Result<LinkInfo, Error> {
    let build_dir = env::current_dir().unwrap();
    let root_dir = build_dir.parent().unwrap().parent().unwrap();
    let third_party = root_dir.join("ThirdParty");
    let metis_dir = third_party.join("Metis");
    let mumps_dir = third_party.join("Mumps");

    let set_wget_cmd = "s/wgetcmd=ftp/wgetcmd=\"curl -L -O\"/g";

    env::set_current_dir(metis_dir).unwrap();
    run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Metis"));
    run(env::current_dir()?.join("get.Metis").to_str().unwrap(), |cmd| cmd);

    env::set_current_dir(mumps_dir).unwrap();
    run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Mumps"));
    run(env::current_dir()?.join("get.Mumps").to_str().unwrap(), |cmd| cmd);

    let mut link_libs = vec![
        (LibType::Static, "ipopt".to_string()),
        (LibType::Static, "coinmumps".to_string()),
        (LibType::Static, "coinmetis".to_string()),
    ];

    let mut search_paths = vec![install_dir.join("lib")];
    let mut include_paths = vec![install_dir.join("include")];

    if cfg!(target_os = "linux") {
        if let Ok(mut openblas_lib) = find_linux_lib("openblas", "cblas.h") {
            link_libs.append(&mut openblas_lib.libs);
            search_paths.append(&mut openblas_lib.search_paths);
            include_paths.append(&mut openblas_lib.include_paths);
        } else {
            // Couldn't find system installed openblas. Build the blas library included with Ipopt.
            let blas_dir = third_party.join("Blas");
            env::set_current_dir(blas_dir).unwrap();
            run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Blas"));
            run(env::current_dir()?.join("get.Blas").to_str().unwrap(), |cmd| cmd);
            let lapack_dir = third_party.join("Lapack");
            env::set_current_dir(lapack_dir).unwrap();
            run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Lapack"));
            run(env::current_dir()?.join("get.Lapack").to_str().unwrap(), |cmd| cmd);
            link_libs.push((LibType::Static, "coinblas".to_string()));
            link_libs.push((LibType::Static, "coinlapack".to_string()));
        }
        // This is a prerequisite on linux systems
        link_libs.push((LibType::Dynamic, "gfortran".to_string()));
    } // macOS ships with the Accelerate framework.

    env::set_current_dir(&build_dir).unwrap();

    run(root_dir.join("configure").to_str().unwrap(), |cmd| {
        let cmd = cmd
            .arg(format!("--prefix={}", install_dir.display()))
            .args(&BUILD_FLAGS);

        if debug {
            cmd.arg(format!("--enable-debug-ipopt"))
        } else {
            cmd
        }
    });

    run("make", |cmd| cmd.arg(format!("-j{}", num_cpus::get())));
    //run("make", |cmd| cmd.arg("test")); // Ensure everything is working
    run("make", |cmd| cmd.arg("install")); // Install to install_dir

    Ok(LinkInfo { libs: link_libs, search_paths, include_paths })
}

fn remove_suffix(value: &mut String, suffix: &str) {
    if value.ends_with(suffix) {
        let n = value.len();
        value.truncate(n - suffix.len());
    }
}

fn extract_tarball<P: AsRef<Path> + std::fmt::Debug, P2: AsRef<Path> + std::fmt::Debug>(archive_path: P, extract_to: P2) {
    eprintln!("Extracting tarball {:?} to {:?}", &archive_path, &extract_to);

    let file = File::open(archive_path).unwrap();
    let mut a = Archive::new(GzDecoder::new(file));
    a.unpack(extract_to).unwrap();
}

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    eprintln!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    eprintln!("Command {:?} finished successfully", configured);
}
