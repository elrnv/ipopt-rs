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
use std::path::{Path, PathBuf, Component};
use std::ffi::OsStr;
use std::process::Command;
use std::{env, fs};
use tar::Archive;

const LIBRARY: &str = "ipopt";
const SOURCE_URL: &str = "https://www.coin-or.org/download/source/Ipopt";
const VERSION: &str = "3.12.10";
const MIN_VERSION: &str = "3.12.8";
const BINARY_DL_URL: &str = "https://github.com/JuliaOpt/IpoptBuilder/releases/download/";
const SOURCE_MD5: &str = "ee250ece251a82dc2580efa51f79d758";
const SOURCE_SHA1: &str = "5eb1aefb2f9acfd8b1b5838370528ac1d73787d6";

#[cfg(target_os = "macos")]
mod platform {
    // For some reason I couldn't build and link to Ipopt as a static lib on macos, so this is here.
    pub static BUILD_FLAGS: [&str; 2] = ["--enable-shared", "--disable-static"];
    pub static LIB_EXT: &str = "dylib";
    pub static BINARY_SUFFIX: &str = "x86_64-apple-darwin14.tar.gz";
    pub static BINARY_MD5: &str = "59825a6b7e40929ff2c88fb23dc82b7c";
    pub static BINARY_SHA1: &str = "a24f1def1ce9fc33393779b574cea9bfb4765c4f";
}

#[cfg(target_os = "linux")]
mod platform {
    pub static BUILD_FLAGS: [&str; 3] = ["--disable-shared", "--enable-static", "--with-pic"];
    pub static LIB_EXT: &str = "a";
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
        Ok(ipopt_install_path) => {
            link(build_cnlp(ipopt_install_path))
                .expect("Failed to create bindings for Ipopt library.");
        }
        Err(err) => {
            msg.push_str(&format!("Failed to find Ipopt using pkg-config: {:?}\n\n", err));
        }
    }

    match build_and_install_ipopt() {
        Ok(ipopt_install_path) => {
            link(build_cnlp(ipopt_install_path))
                .expect("Failed to create bindings for Ipopt library.");
            return;
        }
        Err(err) => {
            msg.push_str(&format!("Failed to build Ipopt from source: {:?}\n\n", err));
        }
    }

    match download_and_install_prebuilt_binary() {
        Ok(ipopt_install_path) => {
            link(build_cnlp(ipopt_install_path)).expect("Failed to create bindings for Ipopt library.");
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
    PkgConfigNotFound,
    PkgConfigInvalidInstallPath,
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

fn library_name() -> String {
    format!("lib{}.{}", LIBRARY, LIB_EXT)
}

#[cfg(target_family = "windows")]
fn versioned_library_name() -> String {
    // No versioning in filenames on Windows.
    format!("lib{}.{}", LIBRARY, LIB_EXT)
}

#[cfg(target_family = "unix")]
fn versioned_library_name() -> String {
    if cfg!(target_os = "macos") {
        format!("lib{}.{}.{}.{}", LIBRARY, LIB_MAJ_VER, LIB_MIN_VER, LIB_EXT)
    } else {
        format!("lib{}.{}", LIBRARY, LIB_EXT)
    }
}

#[cfg(target_family = "unix")]
fn major_versioned_library_name() -> String {
    if cfg!(target_os = "macos") {
        format!("lib{}.{}.{}", LIBRARY, LIB_MAJ_VER, LIB_EXT)
    } else {
        format!("lib{}.{}", LIBRARY, LIB_EXT)
    }
}

// Try to find ipopt install path from pkg_config.
fn try_pkg_config() -> Result<PathBuf, Error> {
    match pkg_config::Config::new()
        .atleast_version(MIN_VERSION)
        .probe(LIBRARY)
    {
        Ok(lib) => {
            dbg!(&lib);
            let ipopt_lib_name = library_name();
            let mut lib_path = Err(Error::PkgConfigInvalidInstallPath);
            for path in lib.link_paths.iter() {
                let candidate_lib = path.join(&ipopt_lib_name);
                if candidate_lib.exists() {
                    lib_path = Ok(candidate_lib);
                    break;
                }
            }

            let lib_path = lib_path?;

            let mut include_path = Err(Error::PkgConfigInvalidInstallPath);
            for path in lib.include_paths.iter() {
                let candidate_include = path.join("coin");
                if candidate_include.exists() {
                    include_path = Ok(candidate_include);
                    break;
                }
            }

            let include_path = include_path?;

            // Extract the common install path.
            let mut install_path = PathBuf::new();
            let mut comp_iter = lib_path.components().zip(include_path.components());
            for (l,i) in &mut comp_iter {
                if l == i {
                    install_path.push(l);
                } else {
                    break;
                }
            }

            // Make sure that the next element gives the include and lib dirs.
            if comp_iter.next() !=
                Some((Component::Normal(OsStr::new("lib")), Component::Normal(OsStr::new("include")))) {
                    Err(Error::PkgConfigInvalidInstallPath)
            } else {
                Ok(install_path)
            }
        },
        Err(_) => Err(Error::PkgConfigNotFound),
    }
}

/// Download the ipopt prebuilt binary from JuliaOpt and install it.
fn download_and_install_prebuilt_binary() -> Result<PathBuf, Error> {
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
    if library_path.exists() {
        // Nothing to be done, library is already installed
        return Ok(install_dir);
    }

    // On unix make sure all artifacts are removed to cleanup the environment
    if cfg!(target_family = "unix") {
        fs::remove_file(lib_dir.join(major_versioned_library_name())).ok();
        fs::remove_file(lib_dir.join(library_name())).ok();
    }

    // Build destination path
    let tarball_path = download_dir.join(file_name);
    dbg!(&tarball_path);

    if !unpacked_dir.exists() {
        download_tarball(&tarball_path, &BINARY_URL, BINARY_MD5, BINARY_SHA1)?;
        extract_tarball(tarball_path, &unpacked_dir);
    }

    // Copy lib
    if !lib_dir.exists() {
        fs::create_dir(&lib_dir)?;
    }
    fs::copy(
        unpacked_dir.join("lib").join(&library_file),
        &library_path,
    )
    .unwrap();

    // Make links (on unix only)
    if cfg!(target_family = "unix") {
        use std::os::unix::fs::symlink;
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

    Ok(install_dir)
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

fn build_cnlp(ipopt_install_dir: PathBuf) -> PathBuf {
    cmake::Config::new("cnlp")
        .define("Ipopt_DIR:STRING", ipopt_install_dir.to_str().unwrap())
        .build()
}

fn link(cnlp_install_path: PathBuf) -> Result<(), Error> {
    // Link to cnlp
    println!(
        "cargo:rustc-link-search=native={}",
        cnlp_install_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=ipopt_cnlp");

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
        }
    }

    check_tarball_hashes(&tarball_path, md5, sha1)?;

    Ok(())
}

/// Build Ipopt statically linked to MKL if possible. Return the path to the ipopt library.
fn build_and_install_ipopt() -> Result<PathBuf, Error> {
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
        // Nothing to be done, library is already installed.
        return Ok(install_dir);
    }

    // Build destination path
    let tarball_path = download_dir.join(file_name);
    dbg!(&tarball_path);

    download_tarball(&tarball_path, &binary_url, SOURCE_MD5, SOURCE_SHA1)?;
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
    
    let res = build_with_mkl(&install_dir, debug).or_else(|_|
        build_with_default_blas(&install_dir, debug));

    // Restore current directory
    env::set_current_dir(proj_root_dir).unwrap();

    let link_libs = res?; // Propagate any errors after we have restored the current dir.

    // Generate an additional CMake script for cnlp to link against any additional libs.
    // This, for instance, depends on whether we link to libipopt statically or dynamically.
    let mut link_libs_cmake_script = fs::File::create(install_dir.join("link_libs.cmake"))?;
    write!(&mut link_libs_cmake_script, "cmake_minimum_required(VERSION 3.6)\n\n")?;
    write!(&mut link_libs_cmake_script, "set(LINK_LIBS \"{}\")", link_libs)?;

    Ok(install_dir)
}


// Build Ipopt with MKL in the current directory.
fn build_with_mkl(install_dir: &Path, debug: bool) -> Result<String, Error> {
    let mkl_libs = ["intel_lp64", "tbb_thread", "core"];

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
            if mkl_libs.iter().all(|lib| usr_lib_path.join(format!("libmkl_{}.a", lib)).exists()) {
                usr_lib_path
            } else {
                PathBuf::from("NOTFOUND")
            }
        }
    };

    dbg!(&mkl_libs_path);

    let mut forward_libs = String::new();

    let blas = {
        if !mkl_libs_path.exists() {
            return Err(Error::MKLInstallNotFound);
        } else {
            let mkl_prefix = format!("{}/libmkl_", mkl_libs_path.display());
            let aux_libs = format!("-L{mkl} -ltbb -lpthread -lm -ldl", mkl = mkl_libs_path.display());
            let mkl_libs = format!("{mkl}{l1}.a {mkl}{l2}.a {mkl}{l3}.a", 
                    mkl = mkl_prefix,
                    l1 = mkl_libs[0],
                    l2 = mkl_libs[1],
                    l3 = mkl_libs[2])
                ;
            if cfg!(target_os = "macos") {
                format!( "--with-blas={mkl} {aux} -lc++", mkl=mkl_libs, aux=aux_libs)
            } else if cfg!(target_os = "linux") {
                // Only forward the libs to cnlp on linux because we build ipopt statically here.
                let mkl_group = format!("-Wl,--start-group {} -Wl,--end-group", mkl_libs);
                let all_libs = format!("{mkl} {aux} -lstdc++", mkl=mkl_group, aux=aux_libs);
                forward_libs.push_str(&all_libs);
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

    Ok(forward_libs)
}

// Build Ipopt with Default libs.
fn build_with_default_blas(install_dir: &Path, debug: bool) -> Result<String, Error> {
    let build_dir = env::current_dir().unwrap();
    let root_dir = build_dir.parent().unwrap().parent().unwrap();
    let third_party = root_dir.join("ThirdParty");
    let asl_dir = third_party.join("ASL");
    let metis_dir = third_party.join("Metis");
    let mumps_dir = third_party.join("Mumps");

    env::set_current_dir(asl_dir).unwrap();
    run(env::current_dir()?.join("get.ASL").to_str().unwrap(), |cmd| cmd);

    let set_wget_cmd = "s/wgetcmd=ftp/wgetcmd=\"curl -L -O\"/g";

    env::set_current_dir(metis_dir).unwrap();
    run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Metis"));
    run(env::current_dir()?.join("get.Metis").to_str().unwrap(), |cmd| cmd);

    env::set_current_dir(mumps_dir).unwrap();
    run("sed", |cmd| cmd.arg("-i~").arg(set_wget_cmd).arg("get.Mumps"));
    run(env::current_dir()?.join("get.Mumps").to_str().unwrap(), |cmd| cmd);

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

    let blas_lapack_libs = if cfg!(target_os = "linux") { "-lblas -llapack" } else { "" };

    let mut link_libs = 
    format!("{link} {inst}/libcoinmumps.a {inst}/libcoinasl.a {inst}/libcoinmetis.a",
            link=blas_lapack_libs, inst=install_dir.join("lib").to_str().unwrap());

    Ok(link_libs)
}

fn remove_suffix(value: &mut String, suffix: &str) {
    if value.ends_with(suffix) {
        let n = value.len();
        value.truncate(n - suffix.len());
    }
}

fn extract_tarball<P: AsRef<Path> + std::fmt::Debug, P2: AsRef<Path> + std::fmt::Debug>(archive_path: P, extract_to: P2) {
    println!("Extracting tarball {:?} to {:?}", &archive_path, &extract_to);
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
    println!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    println!("Command {:?} finished successfully", configured);
}
