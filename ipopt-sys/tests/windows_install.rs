#[cfg(windows)]
#[test]
fn ipopt_dir_contains_an_msvc_layout() {
    let Some(root) = std::env::var_os("IPOPT_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let headers = [
        root.join("include")
            .join("coin")
            .join("IpIpoptApplication.hpp"),
        root.join("include")
            .join("coin-or")
            .join("IpIpoptApplication.hpp"),
        root.join("include").join("IpIpoptApplication.hpp"),
    ];
    assert!(headers.iter().any(|path| path.exists()));
    assert!(
        root.join("lib").join("ipopt.dll.lib").exists()
            || root.join("lib").join("ipopt.lib").exists()
    );
    assert!(
        root.join("bin").join("ipopt.dll").exists()
            || root.join("bin").join("ipopt-3.dll").exists()
    );
}
