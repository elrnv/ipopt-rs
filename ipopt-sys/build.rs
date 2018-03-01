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
extern crate bindgen;
extern crate curl;
extern crate flate2;
extern crate tar;

use std::fs::File;
use std::{env, fs};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use curl::easy::Easy;
use flate2::read::GzDecoder;
use tar::Archive;

const LIBRARY: &'static str = "ipopt";
const SOURCE_URL: &'static str = "https://www.coin-or.org/download/source/Ipopt";
const VERSION: &'static str = "3.12.8";
#[cfg(target_os = "macos")]
static LIB_EXT: &'static str = "dylib";
#[cfg(target_os = "linux")]
static LIB_EXT: &'static str = "so.1";

macro_rules! log {
    ($fmt:expr) => (println!(concat!("ipopt-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("ipopt-sys/build.rs:{}: ", $fmt),
    line!(), $($arg)*));
}

macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    // Compile ipopt from source
    // TODO: Implement building on Windows.
    
    // Build URL to download from
    let binary_url = format!("{}/Ipopt-{}.tgz", SOURCE_URL, VERSION);
    log_var!(binary_url);

    // Extract the filename from the URL
    let file_name = binary_url.split("/").last().unwrap();
    let mut base_name = file_name.to_string();
    remove_suffix(&mut base_name, ".tgz");
    log_var!(base_name);

    // Create download directory if it doesn't yet exist
    let target_dir = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("target");
    log_var!(target_dir);
    if !target_dir.exists() {
        fs::create_dir(&target_dir).unwrap();
    }

    let download_dir = target_dir.join(format!("ipopt-{}", VERSION));
    log_var!(download_dir);
    if !download_dir.exists() {
        fs::create_dir(&download_dir).unwrap();
    }

    // Download, extract and compile the tarball if the library isn't there.
    let unpacked_dir = download_dir.join(base_name);
    let install_dir = download_dir.clone();
    let lib_dir = install_dir.join("lib");
    let library_file = format!("lib{}.{}", LIBRARY, LIB_EXT);
    let library_path = lib_dir.join(&library_file);
    if !library_path.exists() {

        // Build destination path
        let tarball_path = download_dir.join(file_name);
        log_var!(tarball_path);

        // Download the tarball.
        if !tarball_path.exists() {
            let f = File::create(&tarball_path).unwrap();
            let mut writer = BufWriter::new(f);
            let mut easy = Easy::new();
            easy.url(&binary_url).unwrap();
            easy.write_function(move |data| {
                Ok(writer.write(data).unwrap())
            }).unwrap();
            easy.perform().unwrap();

            let response_code = easy.response_code().unwrap();
            if response_code != 200 {
                panic!("Unexpected response code {} for {}", response_code, binary_url);
            }
        }

        extract(tarball_path, &download_dir);

        // Configure and compile
        let build_dir = unpacked_dir.join("build");
        if !build_dir.exists() {
            fs::create_dir(&build_dir).unwrap();
        }

        // Remember project root directory
        let proj_root_dir = env::current_dir().unwrap();

        // Look for intel MKL and link to its libraries if found.

        // TODO: Find intel MKL programmatically
        let mkl_dir = PathBuf::from("/opt/intel/mkl");

        env::set_current_dir(build_dir).unwrap();

        run(unpacked_dir.join("configure").to_str().unwrap(), |cmd| {
            let blas = 
                if !mkl_dir.exists() {
                    String::new()
                } else {
                    let link_libs = "-ltbb -lstdc++ -lpthread -lm -ldl";
                    let mkl_prefix = format!("{}/lib/libmkl_", mkl_dir.display());
                    format!("--with-blas={mkl}intel_lp64.a {mkl}tbb_thread.a {mkl}core.a {}",
                            link_libs, mkl=mkl_prefix)
                };
            cmd.arg(format!("--prefix={}", install_dir.display())).arg(blas)
        });

        run("make", |cmd| cmd.arg("-j8")); // TODO: Get CPU count programmatically.
        run("make", |cmd| cmd.arg("test")); // Ensure everything is working
        run("make", |cmd| cmd.arg("install")); // Install to install_dir

        // Restore current directory
        env::set_current_dir(proj_root_dir).unwrap();
    }

    // Link to library
    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    let output = PathBuf::from(&env::var("OUT_DIR").unwrap());
    let new_library_path = output.join(&library_file);
    if new_library_path.exists() {
        log!("File {} already exists, deleting.", new_library_path.display());
        fs::remove_file(&new_library_path).unwrap();
    }

    // Copy new lib to the location from which we link
    log!("Copying {} to {}...", library_path.display(), new_library_path.display());
    fs::copy(&library_path, &new_library_path).unwrap();
    println!("cargo:rustc-link-search={}", output.display());

    // Generate raw bindings to ipopt C interface
    let capi_path = install_dir
        .join("include")
        .join("coin")
        .join("IpStdCInterface.h");
    let bindings = bindgen::builder()
            .header(capi_path.to_str().unwrap())
            .link(LIBRARY)
            .generate()
            .expect("Unable to generate bindings!");
    bindings
        .write_to_file(output.join("IpStdCInterface.rs"))
        .expect("Couldn't write bindings!");
}

fn remove_suffix(value: &mut String, suffix: &str) {
    if value.ends_with(suffix) {
        let n = value.len();
        value.truncate(n - suffix.len());
    }
}

fn extract<P: AsRef<Path>, P2: AsRef<Path>>(archive_path: P, extract_to: P2) {
    let file = File::open(archive_path).unwrap();
    let mut a = Archive::new(GzDecoder::new(file));
    a.unpack(extract_to).unwrap();
}

fn run<F>(name: &str, mut configure: F)
        where F: FnMut(&mut Command) -> &mut Command
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}
