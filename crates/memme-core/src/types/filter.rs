use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::sql_param::SqlParam;

// ============================================================
// Advanced Filter Expressions
// ============================================================

/// A filter expression supporting advanced operators like in, contains, gte, lt, AND, OR.
///
/// # Examples
/// ```
/// use memme_core::types::FilterExpression;
///
/// // Simple equality
/// let f = FilterExpression::eq("user_id", "alice");
///
/// // Compound
/// let f = FilterExpression::and(vec![
///     FilterExpression::eq("user_id", "alice"),
///     FilterExpression::contains("categories", "finance"),
///     FilterExpression::gte("importance", 0.5),
/// ]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterExpression {
    /// Field-level condition: { field, operator, value }
    Condition {
        field: String,
        op: FilterOp,
        value: serde_json::Value,
    },
    /// Logical AND: all sub-expressions must match
    And(Vec<FilterExpression>),
    /// Logical OR: any sub-expression must match
    Or(Vec<FilterExpression>),
}

/// Supported filter operators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterOp {
    /// Exact equality
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// Value is in a list
    In,
    /// Field contains substring (case-sensitive)
    Contains,
    /// Field contains substring (case-insensitive)
    IContains,
}

impl FilterExpression {
    /// Create an equality condition (`field = value`).
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Eq,
            value: value.into(),
        }
    }

    /// Create a not-equal condition (`field != value`).
    pub fn ne(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Ne,
            value: value.into(),
        }
    }

    /// Create a greater-than condition (`field > value`).
    pub fn gt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Gt,
            value: value.into(),
        }
    }

    /// Create a greater-than-or-equal condition (`field >= value`).
    pub fn gte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Gte,
            value: value.into(),
        }
    }

    /// Create a less-than condition (`field < value`).
    pub fn lt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Lt,
            value: value.into(),
        }
    }

    /// Create a less-than-or-equal condition (`field <= value`).
    pub fn lte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Lte,
            value: value.into(),
        }
    }

    /// Create an "in list" condition (`field IN (values...)`).
    pub fn is_in(field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::In,
            value: serde_json::Value::Array(values),
        }
    }

    /// Create a case-sensitive substring containment condition.
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::Contains,
            value: serde_json::Value::String(value.into()),
        }
    }

    /// Create a case-insensitive substring containment condition.
    pub fn icontains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Condition {
            field: field.into(),
            op: FilterOp::IContains,
            value: serde_json::Value::String(value.into()),
        }
    }

    /// Combine multiple expressions with logical AND.
    pub fn and(exprs: Vec<FilterExpression>) -> Self {
        Self::And(exprs)
    }

    /// Combine multiple expressions with logical OR.
    pub fn or(exprs: Vec<FilterExpression>) -> Self {
        Self::Or(exprs)
    }

    /// Convert from legacy simple HashMap filter to FilterExpression.
    pub fn from_simple_map(map: HashMap<String, serde_json::Value>) -> Self {
        let conditions: Vec<FilterExpression> = map
            .into_iter()
            .map(|(k, v)| FilterExpression::eq(k, v))
            .collect();
        if conditions.len() == 1 {
            conditions.into_iter().next().unwrap()
        } else {
            FilterExpression::And(conditions)
        }
    }

    /// Compile the filter expression into a SQL WHERE clause fragment and parameters.
    /// Returns (sql_fragment, params) where params are database-agnostic `SqlParam` values.
    pub fn to_sql(&self, param_offset: &mut usize) -> (String, Vec<SqlParam>) {
        match self {
            FilterExpression::Condition { field, op, value } => {
                // Validate field name to prevent SQL injection
                let safe_field = sanitize_field_name(field);
                let (sql, params) = condition_to_sql(&safe_field, op, value, param_offset);
                (sql, params)
            }
            FilterExpression::And(exprs) => {
                if exprs.is_empty() {
                    return ("1=1".to_string(), vec![]);
                }
                let mut parts = Vec::new();
                let mut all_params = Vec::new();
                for expr in exprs {
                    let (sql, params) = expr.to_sql(param_offset);
                    parts.push(format!("({})", sql));
                    all_params.extend(params);
                }
                (parts.join(" AND "), all_params)
            }
            FilterExpression::Or(exprs) => {
                if exprs.is_empty() {
                    return ("1=0".to_string(), vec![]);
                }
                let mut parts = Vec::new();
                let mut all_params = Vec::new();
                for expr in exprs {
                    let (sql, params) = expr.to_sql(param_offset);
                    parts.push(format!("({})", sql));
                    all_params.extend(params);
                }
                (format!("({})", parts.join(" OR ")), all_params)
            }
        }
    }
}

/// Allowed column names that can be used in filter expressions.
const ALLOWED_FILTER_FIELDS: &[&str] = &[
    "user_id",
    "agent_id",
    "app_id",
    "run_id",
    "actor_id",
    "importance",
    "access_count",
    "immutable",
    "categories",
    "created_at",
    "updated_at",
    "expiration_date",
    "content",
    "metadata",
    "memory_type",
    "privacy",
    "event_time",
];

fn sanitize_field_name(field: &str) -> String {
    if ALLOWED_FILTER_FIELDS.contains(&field) {
        field.to_string()
    } else {
        // Treat as metadata JSON path — validate field name to prevent injection.
        // Only allow alphanumeric, underscore, and dot (for nested paths).
        if !field
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
        {
            // Return a safe expression that always evaluates to NULL
            return "NULL".to_string();
        }
        format!("json_extract(metadata, '$.{}')", field)
    }
}

fn condition_to_sql(
    field: &str,
    op: &FilterOp,
    value: &serde_json::Value,
    offset: &mut usize,
) -> (String, Vec<SqlParam>) {
    match op {
        FilterOp::Eq => {
            *offset += 1;
            let sql = format!("{} = ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::Ne => {
            *offset += 1;
            let sql = format!("{} != ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::Gt => {
            *offset += 1;
            let sql = format!("{} > ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::Gte => {
            *offset += 1;
            let sql = format!("{} >= ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::Lt => {
            *offset += 1;
            let sql = format!("{} < ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::Lte => {
            *offset += 1;
            let sql = format!("{} <= ${}", field, *offset);
            (sql, vec![SqlParam::from_json(value)])
        }
        FilterOp::In => {
            if let serde_json::Value::Array(arr) = value {
                let placeholders: Vec<String> = arr
                    .iter()
                    .map(|_| {
                        *offset += 1;
                        format!("${}", *offset)
                    })
                    .collect();
                let sql = format!("{} IN ({})", field, placeholders.join(", "));
                let params: Vec<SqlParam> = arr.iter().map(SqlParam::from_json).collect();
                (sql, params)
            } else {
                *offset += 1;
                let sql = format!("{} = ${}", field, *offset);
                (sql, vec![SqlParam::from_json(value)])
            }
        }
        FilterOp::Contains => {
            *offset += 1;
            // For categories field (JSON array), use json_each; for strings, use LIKE
            if field == "categories" {
                let sql = format!(
                    "EXISTS (SELECT 1 FROM json_each({}) WHERE value = ${})",
                    field, *offset
                );
                (sql, vec![SqlParam::from_json(value)])
            } else {
                let sql = format!("{} LIKE '%' || ${} || '%'", field, *offset);
                (sql, vec![SqlParam::from_json(value)])
            }
        }
        FilterOp::IContains => {
            *offset += 1;
            if field == "categories" {
                let sql = format!(
                    "EXISTS (SELECT 1 FROM json_each({}) WHERE LOWER(value) LIKE '%' || LOWER(${}) || '%')",
                    field, *offset
                );
                (sql, vec![SqlParam::from_json(value)])
            } else {
                let sql = format!("LOWER({}) LIKE '%' || LOWER(${}) || '%'", field, *offset);
                (sql, vec![SqlParam::from_json(value)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_expression_eq() {
        let f = FilterExpression::eq("user_id", "alice");
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "user_id = $1");
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_filter_expression_and() {
        let f = FilterExpression::and(vec![
            FilterExpression::eq("user_id", "alice"),
            FilterExpression::gte("importance", serde_json::json!(0.5)),
        ]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "(user_id = $1) AND (importance >= $2)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_filter_expression_or() {
        let f = FilterExpression::or(vec![
            FilterExpression::eq("agent_id", "a1"),
            FilterExpression::eq("agent_id", "a2"),
        ]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "((agent_id = $1) OR (agent_id = $2))");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_filter_expression_in() {
        let f = FilterExpression::is_in("run_id", vec!["r1".into(), "r2".into(), "r3".into()]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "run_id IN ($1, $2, $3)");
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_filter_expression_contains_categories() {
        let f = FilterExpression::contains("categories", "finance");
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(
            sql,
            "EXISTS (SELECT 1 FROM json_each(categories) WHERE value = $1)"
        );
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_filter_expression_contains_string() {
        let f = FilterExpression::contains("content", "coffee");
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "content LIKE '%' || $1 || '%'");
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_filter_expression_nested() {
        let f = FilterExpression::and(vec![
            FilterExpression::eq("user_id", "alice"),
            FilterExpression::or(vec![
                FilterExpression::contains("categories", "work"),
                FilterExpression::gte("importance", serde_json::json!(0.8)),
            ]),
            FilterExpression::lt("created_at", "2026-06-01"),
        ]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert!(sql.contains("user_id = $1"));
        assert!(sql.contains("OR"));
        assert!(sql.contains("created_at < $4"));
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_filter_from_simple_map() {
        let mut map = HashMap::new();
        map.insert("user_id".into(), serde_json::json!("alice"));
        let f = FilterExpression::from_simple_map(map);
        let mut offset = 0;
        let (sql, _) = f.to_sql(&mut offset);
        assert!(sql.contains("user_id = $1"));
    }

    #[test]
    fn test_filter_metadata_json_path() {
        let f = FilterExpression::eq("source", "web");
        let mut offset = 0;
        let (sql, _) = f.to_sql(&mut offset);
        // "source" is not in ALLOWED_FILTER_FIELDS, so it becomes json_extract_string
        assert!(sql.contains("json_extract(metadata, '$.source')"));
    }

    #[test]
    fn test_filter_empty_and_or() {
        let f = FilterExpression::and(vec![]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "1=1");
        assert!(params.is_empty());

        let f = FilterExpression::or(vec![]);
        let mut offset = 0;
        let (sql, params) = f.to_sql(&mut offset);
        assert_eq!(sql, "1=0");
        assert!(params.is_empty());
    }
}
