//! Safe ownership wrapper for reqwest's blocking client.
//!
//! Dropping the last blocking client inside an async Tokio task can panic while
//! reqwest shuts down its private runtime. Provider objects are often owned by
//! async servers, so release the client on a small plain thread instead.

pub(crate) struct SafeBlockingClient {
    inner: Option<reqwest::blocking::Client>,
}

impl From<reqwest::blocking::Client> for SafeBlockingClient {
    fn from(client: reqwest::blocking::Client) -> Self {
        Self {
            inner: Some(client),
        }
    }
}

impl std::ops::Deref for SafeBlockingClient {
    type Target = reqwest::blocking::Client;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .expect("blocking HTTP client is available before drop")
    }
}

impl Drop for SafeBlockingClient {
    fn drop(&mut self) {
        let Some(client) = self.inner.take() else {
            return;
        };
        if let Ok(handle) = std::thread::Builder::new()
            .name("memme-http-client-drop".to_string())
            .spawn(move || drop(client))
        {
            let _ = handle.join();
        }
    }
}
