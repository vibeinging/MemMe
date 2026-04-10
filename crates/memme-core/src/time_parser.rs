//! Parse natural-language time references (English + Chinese) into date ranges.
//!
//! Used by the temporal search channel to convert queries like "last week" or
//! "three days ago" into concrete `(start, end)` ISO date pairs that can be
//! used in SQL `BETWEEN` clauses.

use chrono::{Datelike, NaiveDate, Utc};

/// A concrete date range (inclusive on both ends).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimeRange {
    /// ISO-8601 date string (YYYY-MM-DD)
    pub start: String,
    /// ISO-8601 date string (YYYY-MM-DD)
    pub end: String,
}

/// Extract all recognisable time references from `query` and return the
/// corresponding date ranges.  An empty `Vec` means no temporal references
/// were detected.
pub(crate) fn parse_time_references(query: &str) -> Vec<TimeRange> {
    let today = Utc::now().date_naive();
    parse_time_references_with_anchor(query, today)
}

/// Testable version that accepts an explicit "today" anchor.
fn parse_time_references_with_anchor(query: &str, today: NaiveDate) -> Vec<TimeRange> {
    let q = query.to_lowercase();
    let mut ranges: Vec<TimeRange> = Vec::new();

    // ── Exact ISO dates (YYYY-MM-DD) ──
    // Scan for patterns like 2026-04-10
    let bytes = q.as_bytes();
    let mut i = 0;
    while i + 9 < bytes.len() {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4] == b'-'
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6].is_ascii_digit()
            && bytes[i + 7] == b'-'
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
        {
            let date_str = &q[i..i + 10];
            if NaiveDate::parse_from_str(date_str, "%Y-%m-%d").is_ok() {
                ranges.push(TimeRange {
                    start: date_str.to_string(),
                    end: date_str.to_string(),
                });
            }
            i += 10;
            continue;
        }
        i += 1;
    }

    // ── "yesterday" / "昨天" ──
    if q.contains("yesterday") || q.contains("昨天") {
        if let Some(d) = today.pred_opt() {
            let s = d.format("%Y-%m-%d").to_string();
            ranges.push(TimeRange {
                start: s.clone(),
                end: s,
            });
        }
    }

    // ── "today" / "今天" ──
    if q.contains("today") || q.contains("今天") {
        let s = today.format("%Y-%m-%d").to_string();
        ranges.push(TimeRange {
            start: s.clone(),
            end: s,
        });
    }

    // ── "last week" / "上周" / "上个星期" ──
    if q.contains("last week") || q.contains("上周") || q.contains("上个星期") {
        let start = today - chrono::Duration::days(7);
        ranges.push(TimeRange {
            start: start.format("%Y-%m-%d").to_string(),
            end: today.format("%Y-%m-%d").to_string(),
        });
    }

    // ── "last month" / "上个月" / "上月" ──
    if q.contains("last month") || q.contains("上个月") || q.contains("上月") {
        let (y, m) = if today.month() == 1 {
            (today.year() - 1, 12)
        } else {
            (today.year(), today.month() - 1)
        };
        let start = NaiveDate::from_ymd_opt(y, m, 1).unwrap_or(today);
        let end = month_last_day(y, m);
        ranges.push(TimeRange {
            start: start.format("%Y-%m-%d").to_string(),
            end: end.format("%Y-%m-%d").to_string(),
        });
    }

    // ── "last year" / "去年" ──
    if q.contains("last year") || q.contains("去年") {
        let y = today.year() - 1;
        ranges.push(TimeRange {
            start: format!("{y}-01-01"),
            end: format!("{y}-12-31"),
        });
    }

    // ── "N days ago" / "N天前" ──
    if let Some(n) = parse_n_unit_ago(&q, "day", "天") {
        let d = today - chrono::Duration::days(n as i64);
        let s = d.format("%Y-%m-%d").to_string();
        ranges.push(TimeRange {
            start: s.clone(),
            end: s,
        });
    }

    // ── "N weeks ago" / "N周前" / "N个星期前" ──
    if let Some(n) = parse_n_unit_ago(&q, "week", "周") {
        let d = today - chrono::Duration::weeks(n as i64);
        let s = d.format("%Y-%m-%d").to_string();
        ranges.push(TimeRange {
            start: s,
            end: today.format("%Y-%m-%d").to_string(),
        });
    }

    // ── "N months ago" / "N个月前" ──
    if let Some(n) = parse_n_unit_ago(&q, "month", "月") {
        let (y, m) = subtract_months(today.year(), today.month(), n);
        let start = NaiveDate::from_ymd_opt(y, m, 1).unwrap_or(today);
        let end = month_last_day(y, m);
        ranges.push(TimeRange {
            start: start.format("%Y-%m-%d").to_string(),
            end: end.format("%Y-%m-%d").to_string(),
        });
    }

    // ── "in March", "in March 2026" / English month names ──
    let en_months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];
    for (name, month_num) in &en_months {
        if q.contains(name) {
            // Try to find an associated year (e.g. "march 2026" or "2026 march")
            let year = extract_year_near_month(&q, name).unwrap_or(today.year());
            let start = NaiveDate::from_ymd_opt(year, *month_num, 1).unwrap_or(today);
            let end = month_last_day(year, *month_num);
            ranges.push(TimeRange {
                start: start.format("%Y-%m-%d").to_string(),
                end: end.format("%Y-%m-%d").to_string(),
            });
        }
    }

    // ── Chinese month names: "一月"..."十二月" or "1月"..."12月" or "三月" ──
    let cn_months: &[(&str, u32)] = &[
        ("一月", 1),
        ("二月", 2),
        ("三月", 3),
        ("四月", 4),
        ("五月", 5),
        ("六月", 6),
        ("七月", 7),
        ("八月", 8),
        ("九月", 9),
        ("十月", 10),
        ("十一月", 11),
        ("十二月", 12),
    ];
    for (name, month_num) in cn_months {
        if q.contains(name) {
            let year = today.year(); // Chinese month names rarely include year inline
            let start = NaiveDate::from_ymd_opt(year, *month_num, 1).unwrap_or(today);
            let end = month_last_day(year, *month_num);
            ranges.push(TimeRange {
                start: start.format("%Y-%m-%d").to_string(),
                end: end.format("%Y-%m-%d").to_string(),
            });
        }
    }

    // ── "N月" pattern (digit + 月) — only if not already matched by Chinese month names ──
    // e.g. "3月", "12月"
    if ranges.is_empty() || !cn_months.iter().any(|(n, _)| q.contains(n)) {
        for cap in find_digit_month_chinese(&q) {
            let year = today.year();
            let start = NaiveDate::from_ymd_opt(year, cap, 1).unwrap_or(today);
            let end = month_last_day(year, cap);
            ranges.push(TimeRange {
                start: start.format("%Y-%m-%d").to_string(),
                end: end.format("%Y-%m-%d").to_string(),
            });
        }
    }

    // ── "this week" / "这周" / "本周" ──
    if q.contains("this week") || q.contains("这周") || q.contains("本周") {
        // ISO week starts on Monday
        let weekday = today.weekday().num_days_from_monday();
        let monday = today - chrono::Duration::days(weekday as i64);
        let sunday = monday + chrono::Duration::days(6);
        ranges.push(TimeRange {
            start: monday.format("%Y-%m-%d").to_string(),
            end: sunday.format("%Y-%m-%d").to_string(),
        });
    }

    // ── "this month" / "这个月" / "本月" ──
    if q.contains("this month") || q.contains("这个月") || q.contains("本月") {
        let start = NaiveDate::from_ymd_opt(today.year(), today.month(), 1).unwrap_or(today);
        let end = month_last_day(today.year(), today.month());
        ranges.push(TimeRange {
            start: start.format("%Y-%m-%d").to_string(),
            end: end.format("%Y-%m-%d").to_string(),
        });
    }

    ranges
}

// ── Helpers ──

/// Get the last day of a given year/month.
fn month_last_day(year: i32, month: u32) -> NaiveDate {
    if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .and_then(|d| d.pred_opt())
    .unwrap_or_else(|| NaiveDate::from_ymd_opt(year, month, 28).unwrap())
}

/// Subtract `n` months from `(year, month)`, handling underflow.
fn subtract_months(year: i32, month: u32, n: u32) -> (i32, u32) {
    let total = (year * 12 + month as i32 - 1) - n as i32;
    let y = total.div_euclid(12);
    let m = total.rem_euclid(12) as u32 + 1;
    (y, m)
}

/// Parse "N days ago" / "N天前" style patterns.
/// Returns `Some(N)` if found.
fn parse_n_unit_ago(q: &str, en_unit: &str, cn_unit: &str) -> Option<u32> {
    // English: "<N> <unit>(s) ago"
    let en_pattern = format!("{en_unit}s ago");
    let en_pattern_singular = format!("{en_unit} ago");
    for pat in [&en_pattern, &en_pattern_singular] {
        if let Some(pos) = q.find(pat.as_str()) {
            let before = &q[..pos].trim_end();
            if let Some(n) = extract_trailing_number(before) {
                return Some(n);
            }
        }
    }

    // Chinese: "<N><cn_unit>前"  or  "<N>个<cn_unit>前"
    let cn_pattern = format!("{cn_unit}前");
    if let Some(pos) = q.find(&cn_pattern) {
        let before = &q[..pos];
        // strip optional "个" before the unit
        let before = before.strip_suffix('个').unwrap_or(before);
        if let Some(n) = extract_trailing_number(before) {
            return Some(n);
        }
    }

    None
}

/// Extract a trailing integer from a string (the last contiguous digit sequence).
fn extract_trailing_number(s: &str) -> Option<u32> {
    let trimmed = s.trim_end();
    // Find the last non-digit char and skip past it (respecting multi-byte boundaries).
    let digit_start = if let Some((byte_pos, ch)) = trimmed
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_ascii_digit())
    {
        byte_pos + ch.len_utf8()
    } else {
        0
    };
    let digits = &trimmed[digit_start..];
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u32>().ok().filter(|&n| n > 0)
}

/// Try to find a 4-digit year near a month name in the query.
/// E.g. "march 2026" or "2026 march".
fn extract_year_near_month(q: &str, month_name: &str) -> Option<i32> {
    let pos = q.find(month_name)?;

    // Look after the month name
    let after = &q[pos + month_name.len()..];
    let after = after.trim_start();
    if after.len() >= 4 {
        let candidate = &after[..4];
        if let Ok(y) = candidate.parse::<i32>() {
            if (1900..=2100).contains(&y) {
                return Some(y);
            }
        }
    }

    // Look before the month name
    if pos >= 4 {
        let before = q[..pos].trim_end();
        if before.len() >= 4 {
            let candidate = &before[before.len() - 4..];
            if let Ok(y) = candidate.parse::<i32>() {
                if (1900..=2100).contains(&y) {
                    return Some(y);
                }
            }
        }
    }

    None
}

/// Find patterns like "3月" or "12月" (digit + 月 without a Chinese numeral prefix).
fn find_digit_month_chinese(q: &str) -> Vec<u32> {
    let mut results = Vec::new();
    let month_char = '月';
    for (idx, _) in q.match_indices(month_char) {
        // Check if preceded by "前" — that would be "N个月前" already handled
        if idx == 0 {
            continue;
        }
        // Extract digits before 月
        let before = &q[..idx];
        if let Some(n) = extract_trailing_number(before) {
            if (1..=12).contains(&n) {
                results.push(n);
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn anchor() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 4, 8).unwrap()
    }

    #[test]
    fn test_yesterday_en() {
        let ranges = parse_time_references_with_anchor("what happened yesterday", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-07");
        assert_eq!(ranges[0].end, "2026-04-07");
    }

    #[test]
    fn test_yesterday_cn() {
        let ranges = parse_time_references_with_anchor("昨天发生了什么", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-07");
    }

    #[test]
    fn test_last_week_en() {
        let ranges = parse_time_references_with_anchor("what did I do last week", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-01");
        assert_eq!(ranges[0].end, "2026-04-08");
    }

    #[test]
    fn test_last_week_cn() {
        let ranges = parse_time_references_with_anchor("上周的事情", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-01");
    }

    #[test]
    fn test_last_month_en() {
        let ranges = parse_time_references_with_anchor("tell me about last month", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-03-01");
        assert_eq!(ranges[0].end, "2026-03-31");
    }

    #[test]
    fn test_last_month_cn() {
        let ranges = parse_time_references_with_anchor("上个月做了什么", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-03-01");
        assert_eq!(ranges[0].end, "2026-03-31");
    }

    #[test]
    fn test_in_march() {
        let ranges = parse_time_references_with_anchor("what happened in march", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-03-01");
        assert_eq!(ranges[0].end, "2026-03-31");
    }

    #[test]
    fn test_in_march_2025() {
        let ranges = parse_time_references_with_anchor("what happened in march 2025", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2025-03-01");
        assert_eq!(ranges[0].end, "2025-03-31");
    }

    #[test]
    fn test_exact_date() {
        let ranges = parse_time_references_with_anchor("what happened on 2026-04-10", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-10");
        assert_eq!(ranges[0].end, "2026-04-10");
    }

    #[test]
    fn test_3_days_ago_en() {
        let ranges = parse_time_references_with_anchor("3 days ago", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-05");
        assert_eq!(ranges[0].end, "2026-04-05");
    }

    #[test]
    fn test_3_days_ago_cn() {
        let ranges = parse_time_references_with_anchor("3天前发生了什么", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-05");
    }

    #[test]
    fn test_chinese_month_name() {
        let ranges = parse_time_references_with_anchor("三月的记录", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-03-01");
        assert_eq!(ranges[0].end, "2026-03-31");
    }

    #[test]
    fn test_last_year_en() {
        let ranges = parse_time_references_with_anchor("events from last year", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2025-01-01");
        assert_eq!(ranges[0].end, "2025-12-31");
    }

    #[test]
    fn test_last_year_cn() {
        let ranges = parse_time_references_with_anchor("去年的事", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2025-01-01");
        assert_eq!(ranges[0].end, "2025-12-31");
    }

    #[test]
    fn test_no_temporal_reference() {
        let ranges = parse_time_references_with_anchor("tell me about cats", anchor());
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_today_en() {
        let ranges = parse_time_references_with_anchor("what happened today", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-08");
        assert_eq!(ranges[0].end, "2026-04-08");
    }

    #[test]
    fn test_today_cn() {
        let ranges = parse_time_references_with_anchor("今天发生了什么", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-08");
    }

    #[test]
    fn test_2_weeks_ago() {
        let ranges = parse_time_references_with_anchor("2 weeks ago", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-03-25");
        assert_eq!(ranges[0].end, "2026-04-08");
    }

    #[test]
    fn test_this_week() {
        // 2026-04-08 is a Wednesday; Monday = 2026-04-06, Sunday = 2026-04-12
        let ranges = parse_time_references_with_anchor("this week", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-06");
        assert_eq!(ranges[0].end, "2026-04-12");
    }

    #[test]
    fn test_this_month() {
        let ranges = parse_time_references_with_anchor("this month", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-04-01");
        assert_eq!(ranges[0].end, "2026-04-30");
    }

    #[test]
    fn test_january_edge_last_month() {
        // Test last month when current month is January
        let jan = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let ranges = parse_time_references_with_anchor("last month", jan);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2025-12-01");
        assert_eq!(ranges[0].end, "2025-12-31");
    }

    #[test]
    fn test_2_months_ago() {
        let ranges = parse_time_references_with_anchor("2 months ago", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-02-01");
        assert_eq!(ranges[0].end, "2026-02-28");
    }

    #[test]
    fn test_cn_2_months_ago() {
        let ranges = parse_time_references_with_anchor("2个月前的事", anchor());
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "2026-02-01");
        assert_eq!(ranges[0].end, "2026-02-28");
    }
}
