/// A database-agnostic SQL parameter value.
///
/// Backend-independent SQL parameter type, allowing the filter layer
/// to remain database-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlParam {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl SqlParam {
    /// Convert a JSON value to a `SqlParam`.
    pub fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => SqlParam::Null,
            serde_json::Value::Bool(b) => SqlParam::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    SqlParam::Int(i)
                } else if let Some(f) = n.as_f64() {
                    SqlParam::Float(f)
                } else {
                    SqlParam::Text(n.to_string())
                }
            }
            serde_json::Value::String(s) => SqlParam::Text(s.clone()),
            _ => SqlParam::Text(value.to_string()),
        }
    }
}
