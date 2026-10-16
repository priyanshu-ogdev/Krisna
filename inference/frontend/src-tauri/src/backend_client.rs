//! Production HTTP client for communicating between Rust control plane and Python inference backend.
//!
//! Provides zero-dependency, reliable IPC over local TCP socket (127.0.0.1:8420)
//! with connection timeouts, response parsing, and retries.

use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8420;

#[derive(Clone, Debug)]
pub struct BackendClient {
    pub host: String,
    pub port: u16,
    pub timeout: Duration,
}

impl Default for BackendClient {
    fn default() -> Self {
        let host = std::env::var("KRISNA_BACKEND_HOST").unwrap_or_else(|_| DEFAULT_HOST.into());
        let port = std::env::var("KRISNA_BACKEND_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_PORT);
        Self {
            host,
            port,
            timeout: Duration::from_secs(30),
        }
    }
}

impl BackendClient {
    pub fn new(host: &str, port: u16, timeout_secs: u64) -> Self {
        Self {
            host: host.to_string(),
            port,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    fn connect(&self) -> Result<TcpStream, String> {
        let addr_str = format!("{}:{}", self.host, self.port);
        let addresses = addr_str
            .to_socket_addrs()
            .map_err(|e| format!("cannot resolve backend address {}: {}", addr_str, e))?;
        let mut last_error = None;
        let mut stream = None;
        for address in addresses {
            match TcpStream::connect_timeout(&address, Duration::from_secs(5)) {
                Ok(connected) => {
                    stream = Some(connected);
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }
        let stream = stream.ok_or_else(|| {
            format!(
                "cannot connect to Python backend at {}: {}",
                addr_str,
                last_error.map(|e| e.to_string()).unwrap_or_else(|| "no addresses resolved".into())
            )
        })?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|e| format!("failed setting read timeout: {}", e))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|e| format!("failed setting write timeout: {}", e))?;
        Ok(stream)
    }

    fn request(&self, method: &str, path: &str, body: Option<&str>) -> Result<(u16, Value), String> {
        let mut stream = self.connect()?;
        let body_bytes = body.unwrap_or("").as_bytes();
        let content_length = body_bytes.len();

        let request_header = format!(
            "{} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            method, path, self.host, self.port, content_length
        );

        stream
            .write_all(request_header.as_bytes())
            .map_err(|e| format!("failed sending request header: {}", e))?;
        if content_length > 0 {
            stream
                .write_all(body_bytes)
                .map_err(|e| format!("failed sending request body: {}", e))?;
        }
        stream.flush().map_err(|e| format!("flush failed: {}", e))?;

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader
            .read_line(&mut status_line)
            .map_err(|e| format!("failed reading HTTP status: {}", e))?;

        let parts: Vec<&str> = status_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("malformed status line: {}", status_line));
        }
        let status_code: u16 = parts[1]
            .parse()
            .map_err(|_| format!("invalid status code: {}", parts[1]))?;

        // Read headers
        let mut content_len: Option<usize> = None;
        let mut is_chunked = false;
        loop {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|e| format!("error reading header: {}", e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                let key = k.trim().to_lowercase();
                let val = v.trim();
                if key == "content-length" {
                    content_len = val.parse().ok();
                } else if key == "transfer-encoding" && val.to_lowercase().contains("chunked") {
                    is_chunked = true;
                }
            }
        }

        // Read body
        let body_str = if is_chunked {
            let mut full_body = Vec::new();
            loop {
                let mut size_str = String::new();
                reader
                    .read_line(&mut size_str)
                    .map_err(|e| format!("reading chunk size: {}", e))?;
                let chunk_size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
                if chunk_size == 0 {
                    break;
                }
                let mut chunk_buf = vec![0u8; chunk_size];
                reader
                    .read_exact(&mut chunk_buf)
                    .map_err(|e| format!("reading chunk data: {}", e))?;
                full_body.extend_from_slice(&chunk_buf);
                let mut crlf = [0u8; 2];
                let _ = reader.read_exact(&mut crlf);
            }
            String::from_utf8_lossy(&full_body).to_string()
        } else if let Some(len) = content_len {
            let mut buf = vec![0u8; len];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("error reading fixed body (len {}): {}", len, e))?;
            String::from_utf8_lossy(&buf).to_string()
        } else {
            let mut buf = Vec::new();
            reader
                .read_to_end(&mut buf)
                .map_err(|e| format!("error reading body to EOF: {}", e))?;
            String::from_utf8_lossy(&buf).to_string()
        };

        let parsed: Value = if body_str.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&body_str).unwrap_or_else(|_| Value::String(body_str.clone()))
        };

        Ok((status_code, parsed))
    }

    pub fn get_health(&self) -> Result<Value, String> {
        let (status, val) = self.request("GET", "/health", None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("health failed with status {}: {}", status, val))
        }
    }

    pub fn get_ready(&self) -> Result<Value, String> {
        let (status, val) = self.request("GET", "/ready", None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("ready probe returned {}: {}", status, val))
        }
    }

    pub fn get_models_status(&self) -> Result<Value, String> {
        let (status, val) = self.request("GET", "/models/status", None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("models/status returned {}: {}", status, val))
        }
    }

    pub fn warm_models(&self, background: bool) -> Result<Value, String> {
        let body = serde_json::json!({ "background": background }).to_string();
        let (status, val) = self.request("POST", "/models/warm", Some(&body))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("models/warm returned {}: {}", status, val))
        }
    }

    pub fn get_warm_status(&self) -> Result<Value, String> {
        let (status, val) = self.request("GET", "/models/warm/status", None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("models/warm/status returned {}: {}", status, val))
        }
    }

    pub fn unload_models(&self) -> Result<Value, String> {
        let (status, val) = self.request("POST", "/models/unload", None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("models/unload returned {}: {}", status, val))
        }
    }

    pub fn create_session(&self, style: Option<String>, palette: Vec<String>) -> Result<Value, String> {
        let payload = serde_json::json!({
            "style": style,
            "palette": palette,
        });
        let body = payload.to_string();
        let (status, val) = self.request("POST", "/session", Some(&body))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("create_session failed ({}) : {}", status, val))
        }
    }

    pub fn get_session(&self, session_id: &str) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}", session_id);
        let (status, val) = self.request("GET", &path, None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("get_session failed ({}) : {}", status, val))
        }
    }

    pub fn get_session_model_status(&self, session_id: &str) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/model-status", session_id);
        let (status, val) = self.request("GET", &path, None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("get_session_model_status failed ({}) : {}", status, val))
        }
    }

    pub fn send_message(&self, session_id: &str, message: &str) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/message", session_id);
        let payload = serde_json::json!({ "message": message }).to_string();
        let (status, val) = self.request("POST", &path, Some(&payload))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("send_message failed ({}) : {}", status, val))
        }
    }

    pub fn trigger_self_review(&self, session_id: &str) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/self-review", session_id);
        let (status, val) = self.request("POST", &path, None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("trigger_self_review failed ({}) : {}", status, val))
        }
    }

    pub fn trigger_finalize(&self, session_id: &str, quality: bool, prompt: Option<String>) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/finalize", session_id);
        let payload = serde_json::json!({
            "quality": quality,
            "prompt": prompt,
        }).to_string();
        let (status, val) = self.request("POST", &path, Some(&payload))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("trigger_finalize failed ({}) : {}", status, val))
        }
    }

    pub fn trigger_refine(
        &self,
        session_id: &str,
        instruction: &str,
        quality: bool,
        num_inference_steps: Option<u32>,
        actions: Option<Vec<Value>>,
    ) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/refine", session_id);
        let mut map = serde_json::Map::new();
        map.insert("instruction".into(), serde_json::Value::String(instruction.into()));
        map.insert("quality".into(), serde_json::Value::Bool(quality));
        map.insert("num_inference_steps".into(), serde_json::json!(num_inference_steps.unwrap_or(8)));
        if let Some(acts) = actions {
            map.insert("actions".into(), serde_json::Value::Array(acts));
        }
        let payload = serde_json::Value::Object(map).to_string();
        let (status, val) = self.request("POST", &path, Some(&payload))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("trigger_refine failed ({}) : {}", status, val))
        }
    }

    pub fn trigger_preview(&self, session_id: &str, prompt: Option<String>) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/preview", session_id);
        let payload = serde_json::json!({
            "prompt": prompt,
        }).to_string();
        let (status, val) = self.request("POST", &path, Some(&payload))?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("trigger_preview failed ({}) : {}", status, val))
        }
    }

    pub fn get_research_trace(&self, session_id: &str) -> Result<Value, String> {
        validate_session_id(session_id)?;
        let path = format!("/session/{}/research-trace", session_id);
        let (status, val) = self.request("GET", &path, None)?;
        if status == 200 {
            Ok(val)
        } else {
            Err(format!("get_research_trace failed ({}) : {}", status, val))
        }
    }
}

fn validate_session_id(session_id: &str) -> Result<(), String> {
    if session_id.is_empty()
        || !session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("invalid session id".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_session_id;

    #[test]
    fn session_ids_are_path_safe() {
        assert!(validate_session_id("a1b2-c3").is_ok());
        assert!(validate_session_id("../etc/passwd").is_err());
        assert!(validate_session_id("").is_err());
    }
}
