// SPDX-License-Identifier: GPL-3.0-or-later
//! Continuous export of the current conversation to a JSON file, in the same
//! format as an individual conversation object from a ChatGPT
//! "conversations.json" export:
//!
//! ```json
//! {
//!   "conversation_id": "<uuid>", "create_time": <unix secs>,
//!   "current_node": "<uuid of last message>", "id": "<uuid>",
//!   "is_starred": false, "mapping": { "<node-id>": {id, message, parent}, … },
//!   "title": "…", "update_time": <unix secs>
//! }
//! ```
//!
//! The file is rewritten from scratch whenever [`ChatTranscript::sync`] sees
//! new turns, so it always mirrors `client.history`. User and assistant turns
//! become text messages; the internal system/tool turns are skipped (the
//! ChatGPT format has no representation for them). One file per session: a new
//! session overwrites the previous one. Privacy note: the transcript is
//! written in cleartext, unlike the rest of lumo-cli's state.

use std::path::PathBuf;

use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use serde::Serialize;
use uuid::Uuid;

use crate::client::Turn;
use crate::protocol::Role;

/// Node id used for the (message-less) tree root, mirroring ChatGPT exports.
const ROOT_ID: &str = "client-created-root";

#[derive(Serialize)]
struct Author {
    name: Option<&'static str>,
    role: &'static str,
}

#[derive(Serialize)]
struct Content<'a> {
    content_type: &'static str,
    parts: Vec<&'a str>,
}

#[derive(Serialize)]
struct Metadata<'a> {
    parent_id: Option<&'a str>,
    status: &'static str,
}

#[derive(Serialize)]
struct Message<'a> {
    author: Author,
    content: Content<'a>,
    create_time: f64,
    id: &'a str,
    metadata: Metadata<'a>,
}

#[derive(Serialize)]
struct Node<'a> {
    id: &'a str,
    message: Option<Message<'a>>,
    parent: Option<&'a str>,
}

#[derive(Serialize)]
struct Conversation<'a> {
    conversation_id: &'a str,
    create_time: u64,
    current_node: &'a str,
    id: &'a str,
    is_starred: bool,
    mapping: serde_json::Map<String, serde_json::Value>,
    title: &'a str,
    update_time: u64,
}

/// One recorded message.
struct Record {
    id: String,
    role: &'static str,
    text: String,
    create_time: f64,
    parent: String,
}

/// Rewrite the whole document whenever new turns arrive: conversations are
/// small compared to the network round-trips the CLI performs anyway. One file
/// per session; `/clear` (and the agent↔chat switch) start a fresh one.
pub struct ChatTranscript {
    path: PathBuf,
    conversation_id: String,
    create_time: u64,
    records: Vec<Record>,
    /// Turns of `client.history` already mirrored (system and tool turns
    /// included, so the index stays aligned with the history vec).
    consumed: usize,
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_secs_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn export_role(role: Role) -> Option<&'static str> {
    match role {
        Role::User => Some("user"),
        Role::Assistant => Some("assistant"),
        // The ChatGPT conversation format has no slots for the system prompt
        // or tool traffic; those turns stay in the encrypted request history.
        Role::System | Role::ToolCall | Role::ToolResult => None,
    }
}

impl ChatTranscript {
    /// Create the transcript file at `path`, writing an (empty) document right
    /// away so a bad path surfaces at startup, not after an hour of chatting.
    pub fn open(path: PathBuf) -> Result<Self> {
        let t = Self {
            path,
            conversation_id: Uuid::new_v4().to_string(),
            create_time: unix_secs(),
            records: Vec::new(),
            consumed: 0,
        };
        t.write()?;
        Ok(t)
    }

    /// Mirror any not-yet-recorded user/assistant turns of `history`.
    /// Errors are reported but keep the transcript alive: a failed export
    /// (full disk, …) must not kill the chat; the next sync retries.
    pub fn sync(&mut self, history: &[Turn]) {
        let start = self.consumed.min(history.len());
        let Some(new_turns) = history.get(start..) else {
            return;
        };
        for turn in new_turns {
            self.consumed = self.consumed.saturating_add(1);
            let Some(role) = export_role(turn.role) else {
                continue;
            };
            let parent = self
                .records
                .last()
                .map(|r| r.id.clone())
                .unwrap_or_else(|| ROOT_ID.to_string());
            self.records.push(Record {
                id: Uuid::new_v4().to_string(),
                role,
                text: turn.content.clone(),
                create_time: unix_secs_f64(),
                parent,
            });
        }
        if let Err(e) = self.write() {
            eprintln!("{} conversation export failed: {e:#}", "warning:".yellow());
        }
    }

    /// Start over with a fresh conversation (after /clear or a mode switch).
    pub fn reset(&mut self) {
        self.conversation_id = Uuid::new_v4().to_string();
        self.create_time = unix_secs();
        self.records.clear();
        self.consumed = 0;
        if let Err(e) = self.write() {
            eprintln!("{} conversation export failed: {e:#}", "warning:".yellow());
        }
    }

    fn write(&self) -> Result<()> {
        let mut mapping = serde_json::Map::new();
        mapping.insert(
            ROOT_ID.into(),
            serde_json::to_value(Node {
                id: ROOT_ID,
                message: None,
                parent: None,
            })?,
        );
        let mut last_id = ROOT_ID;
        for r in &self.records {
            let node = Node {
                id: &r.id,
                message: Some(Message {
                    author: Author {
                        name: None,
                        role: r.role,
                    },
                    content: Content {
                        content_type: "text",
                        parts: vec![r.text.as_str()],
                    },
                    create_time: r.create_time,
                    id: &r.id,
                    metadata: Metadata {
                        parent_id: Some(&r.parent),
                        status: "succeeded",
                    },
                }),
                parent: Some(&r.parent),
            };
            mapping.insert(r.id.clone(), serde_json::to_value(node)?);
            last_id = &r.id;
        }

        let title = self
            .records
            .iter()
            .find(|r| r.role == "user")
            .map(|r| title_from(&r.text))
            .unwrap_or_else(|| "lumo-cli session".to_string());

        let doc = Conversation {
            conversation_id: &self.conversation_id,
            create_time: self.create_time,
            current_node: last_id,
            id: &self.conversation_id,
            is_starred: false,
            mapping,
            title: &title,
            update_time: unix_secs(),
        };

        // Write-then-rename, so a reader never sees a half-written file.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("cannot write {}", self.path.display()))?;
        Ok(())
    }
}

/// Truncate to the first line, like a chat title.
fn title_from(text: &str) -> String {
    let first_line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let short: String = first_line.chars().take(60).collect();
    if short.chars().count() < first_line.chars().count() {
        format!("{}…", short.trim_end())
    } else {
        short
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, text: &str) -> Turn {
        Turn {
            role: role,
            content: text.into(),
        }
    }

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lumo-transcript-{tag}-{}", std::process::id()))
    }

    fn cleanup(path: &PathBuf) {
        std::fs::remove_file(path).ok();
        std::fs::remove_file(path.with_extension("json.tmp")).ok();
    }

    #[test]
    fn format_matches_chatgpt_shape() {
        let path = tmp_path("shape");
        let mut t = ChatTranscript::open(path.clone()).unwrap();
        t.sync(&[turn(Role::User, "Hello there"), turn(Role::Assistant, "Hi!")]);

        let raw = std::fs::read_to_string(&path).unwrap();
        let d: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            d["conversation_id"]
                .as_str()
                .and_then(|s| uuid::Uuid::parse_str(s).ok()),
            d["id"].as_str().and_then(|s| uuid::Uuid::parse_str(s).ok())
        );
        assert_eq!(
            d["mapping"]["client-created-root"],
            serde_json::json!({"id": "client-created-root", "message": null, "parent": null})
        );
        // current_node must be the last message's node
        let current = d["current_node"].as_str().unwrap();
        let msg = &d["mapping"][current]["message"];
        assert_eq!(msg["author"]["role"], "assistant");
        assert_eq!(msg["author"]["name"], serde_json::Value::Null);
        assert_eq!(msg["content"]["content_type"], "text");
        assert_eq!(msg["content"]["parts"][0], "Hi!");
        assert_eq!(msg["metadata"]["status"], "succeeded");
        // parent chain ends at the root
        let parent = msg["metadata"]["parent_id"].as_str().unwrap();
        assert_eq!(d["mapping"][parent]["message"]["author"]["role"], "user");
        assert_eq!(d["title"], "Hello there");
        cleanup(&path);
    }

    #[test]
    fn skips_system_and_tool_turns() {
        let path = tmp_path("skip");
        let mut t = ChatTranscript::open(path.clone()).unwrap();
        t.sync(&[
            turn(Role::System, "sys"),
            turn(Role::User, "hi"),
            turn(Role::ToolCall, "{}"),
            turn(Role::ToolResult, "out"),
            turn(Role::Assistant, "done"),
        ]);
        let d: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // mapping is keyed by fresh UUIDs and serde_json sorts map keys, so
        // iteration order is not insertion order: compare as a set.
        let mut roles: Vec<&str> = d["mapping"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(|n| n["message"]["author"]["role"].as_str())
            .collect();
        roles.sort_unstable();
        assert_eq!(roles, vec!["assistant", "user"]);
        cleanup(&path);
    }

    #[test]
    fn incremental_sync_does_not_duplicate() {
        let path = tmp_path("incr");
        let mut t = ChatTranscript::open(path.clone()).unwrap();
        let history = vec![turn(Role::User, "q"), turn(Role::Assistant, "a")];
        t.sync(&history);
        t.sync(&history); // no new turns: file unchanged
        let d: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            d["mapping"]
                .as_object()
                .unwrap()
                .values()
                .filter(|n| n["message"].is_object())
                .count(),
            2
        );
        cleanup(&path);
    }

    #[test]
    fn reset_starts_fresh_conversation() {
        let path = tmp_path("reset");
        let mut t = ChatTranscript::open(path.clone()).unwrap();
        t.sync(&[turn(Role::User, "before")]);
        t.reset();
        t.sync(&[turn(Role::User, "after")]);
        let d: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(d["title"], "after");
        assert_eq!(
            d["mapping"]
                .as_object()
                .unwrap()
                .values()
                .filter(|n| n["message"].is_object())
                .count(),
            1
        );
        cleanup(&path);
    }
}
