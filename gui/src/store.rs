//! Saved chats: one JSON file per chat in the CherryCola data folder.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum Who {
    Me,
    Ai,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Msg {
    pub who: Who,
    pub text: String,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Chat {
    pub id: u64,
    #[serde(default)]
    pub title: String,
    pub msgs: Vec<Msg>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl Chat {
    pub fn new() -> Self {
        Chat { id: now_ms(), title: String::new(), msgs: vec![] }
    }

    /// Earlier turns for `ask --history` (the last, unanswered pair excluded).
    pub fn history(&self) -> Vec<(String, String)> {
        self.msgs
            .iter()
            .filter(|m| !m.text.is_empty())
            .map(|m| {
                let role = if m.who == Who::Me { "user" } else { "assistant" };
                let mut text = m.text.clone();
                if !m.files.is_empty() {
                    let names: Vec<String> = m.files.iter().map(|f| f.rsplit(['/', '\\']).next().unwrap_or(f).to_string()).collect();
                    text = format!("[attached: {}]\n{}", names.join(", "), text);
                }
                (role.to_string(), text)
            })
            .collect()
    }
}

pub fn dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            if cfg!(windows) {
                std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("CherryCola"))
            } else {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share"))
            }
        })
        .unwrap_or_else(|| PathBuf::from("."));
    let d = base.join("cherrycola").join("gui-chats");
    let _ = std::fs::create_dir_all(&d);
    d
}

pub fn load_all() -> Vec<Chat> {
    let mut chats: Vec<Chat> = std::fs::read_dir(dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .filter_map(|t| serde_json::from_str::<Chat>(&t).ok())
        .filter(|c| !c.msgs.is_empty())
        .collect();
    chats.sort_by(|a, b| b.id.cmp(&a.id));
    chats
}

pub fn save(c: &Chat) {
    if c.msgs.is_empty() {
        return;
    }
    if let Ok(t) = serde_json::to_string_pretty(c) {
        let _ = std::fs::write(dir().join(format!("{}.json", c.id)), t);
    }
}

pub fn remove(c: &Chat) {
    let _ = std::fs::remove_file(dir().join(format!("{}.json", c.id)));
}
