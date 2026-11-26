// Gives cherrycolallm-gui.exe its cherry icon in Explorer and the taskbar.
fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "CherryColaLLM");
        res.set("FileDescription", "CherryColaLLM");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the icon: {e}");
        }
    }
}
