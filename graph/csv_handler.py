"""
graph/csv_handler.py
────────────────────
Handles CSV upload parsing, storage, and querying for AWB-level shipment data.

The user uploads a ClickPost CSV export. This module:
1. Parses it into a pandas DataFrame
2. Normalizes column names and date fields
3. Provides query functions that mirror Control Tower widget logic
   (using status codes from Section 9 of data.txt)

The DataFrame lives in Chainlit's user_session for the duration of the chat.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone, timedelta

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Status code mapping (from data.txt Section 9)
# ─────────────────────────────────────────────────────────────────────────────

STATUS_CODES = {
    0: "EXCEPTION",
    1: "ORDER_PLACED",
    2: "PICKUP_PENDING",
    3: "PICKUP_FAILED",
    4: "PICKED_UP",
    5: "INTRANSIT",
    6: "OUT_FOR_DELIVERY",
    7: "NOT_SERVICEABLE",
    8: "DELIVERED",
    9: "FAILED_DELIVERY",
    10: "CANCELLED_ORDER",
    11: "RTO_REQUESTED",
    12: "RTO",
    13: "RTO_OUT_FOR_DELIVERY",
    14: "RTO_DELIVERED",
    15: "RTO_FAILED",
    16: "LOST",
    17: "DAMAGED",
    18: "SHIPMENT_DELAYED",
    19: "CONTACT_CUSTOMER_CARE",
    20: "SHIPMENT_HELD",
    21: "RTO_INTRANSIT",
    22: "TRACKING_NOT_AVAILABLE",
    23: "EXPIRED",
    24: "AGED",
    25: "OUT_FOR_PICKUP",
    26: "RTO_CONTACT_CUSTOMER_CARE",
    27: "RTO_SHIPMENT_DELAY",
    28: "AWB_REGISTERED",
}

# ─────────────────────────────────────────────────────────────────────────────
# Widget filter definitions
# Maps widget names to filter logic using column names from ClickPost exports
# ─────────────────────────────────────────────────────────────────────────────

WIDGET_FILTERS = {
    "pending_pickups": {
        "description": "Orders awaiting pickup — no pickup date assigned",
        "status_codes": [1, 2, 3, 25, 28],
        "conditions": {"pickup_date": "null"},
        "time_column": "created_at",
        "time_buckets_hours": [24, 36, 48],
    },
    "stuck_at_destination_hub": {
        "description": "Shipments at destination hub but not out for delivery",
        "status_codes": [4, 5, 18, 1004, 1005, 1006],
        "conditions": {
            "destination_hub_inscan_checkpoint_timestamp": "not_null",
            "ofd_count": 0,
            "ndr_count": 0,
        },
        "time_column": "destination_hub_inscan_checkpoint_timestamp",
        "time_buckets_hours": [12, 24, 36],
    },
    "stuck_at_source_hub": {
        "description": "Shipments at origin hub without forward movement",
        "status_codes": [4, 5, 18, 1004, 1005, 1006],
        "conditions": {
            "origin_hub_inscan_checkpoint_timestamp": "not_null",
            "ofd_count": 0,
        },
        "time_column": "origin_hub_inscan_checkpoint_timestamp",
        "time_buckets_hours": [12, 24, 36],
    },
    "stuck_in_transit": {
        "description": "Shipments with no tracking updates and no delivery attempts",
        "status_codes": [4, 5, 18, 20, 19, 6, 9, 16, 17, 23, 24, 1004, 1005, 1006],
        "conditions": {"ofd_count_lt": 1},
        "time_column": "tracking_last_updated_at",
        "time_buckets_hours": [12, 24, 36],
    },
    "failed_deliveries": {
        "description": "Shipments that attempted delivery but failed, not yet RTO",
        "status_codes_exclude": [8],
        "conditions": {
            "ofd1_latest_timestamp": "not_null",
            "ndr_count_gte": 1,
            "rto_mark_date": "null",
        },
        "time_column": "tracking_last_updated_at",
        "time_buckets_days": [1, 2, 3],
    },
    "rto_without_attempt": {
        "description": "Shipments returned without any delivery attempt",
        "status_codes": [11, 12, 13, 21, 26, 27, 14, 45, 1002, 51, 52, 53],
        "conditions": {
            "ofd_count": 0,
            "rto_mark_date": "not_null",
        },
    },
    "rto_marked": {
        "description": "Shipments officially marked for RTO",
        "status_codes": [11, 12],
        "conditions": {"rto_mark_date": "not_null"},
        "group_by": "ofd_count",
    },
    "rto_potential": {
        "description": "Active shipments with prolonged tracking silence — RTO risk",
        "status_codes": [4, 5, 18, 6, 9, 1004, 1005, 1006],
        "time_column": "tracking_last_updated_at",
        "time_buckets_hours": [48, 72, 96],
    },
    "delivered": {
        "description": "Successfully delivered shipments",
        "status_codes": [8],
    },
    "exceptions": {
        "description": "Shipments with exceptions (Lost, Damaged, Expired, Aged)",
        "status_codes": [0, 19, 16, 17, 23, 24],
        "conditions": {"pickup_date": "not_null"},
    },
}

# Aliases — common ways users refer to widgets
WIDGET_ALIASES = {
    "pending pickup": "pending_pickups",
    "pending pickups": "pending_pickups",
    "not picked": "pending_pickups",
    "stuck destination": "stuck_at_destination_hub",
    "stuck at destination hub": "stuck_at_destination_hub",
    "stuck at destination": "stuck_at_destination_hub",
    "destination hub": "stuck_at_destination_hub",
    "stuck source": "stuck_at_source_hub",
    "stuck at source hub": "stuck_at_source_hub",
    "stuck at source": "stuck_at_source_hub",
    "source hub": "stuck_at_source_hub",
    "stuck in transit": "stuck_in_transit",
    "stuck transit": "stuck_in_transit",
    "in transit stuck": "stuck_in_transit",
    "failed delivery": "failed_deliveries",
    "failed deliveries": "failed_deliveries",
    "delivery failed": "failed_deliveries",
    "rto without attempt": "rto_without_attempt",
    "rto no attempt": "rto_without_attempt",
    "rto marked": "rto_marked",
    "marked rto": "rto_marked",
    "rto potential": "rto_potential",
    "potential rto": "rto_potential",
    "delivered": "delivered",
    "exception": "exceptions",
    "exceptions": "exceptions",
    "lost": "exceptions",
    "damaged": "exceptions",
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV Parser
# ─────────────────────────────────────────────────────────────────────────────

# Date columns to auto-parse
DATE_COLUMNS = [
    "created_at", "pickup_date", "order_date", "latest_timestamp",
    "rto_mark_date", "tracking_last_updated_at",
    "ofd1_latest_timestamp", "ofd2_latest_timestamp", "ofd3_latest_timestamp",
    "ofd4_latest_timestamp", "ofd5_latest_timestamp",
    "ofp1_latest_timestamp", "ofp2_latest_timestamp", "ofp3_latest_timestamp",
    "npr1_latest_timestamp",
    "origin_hub_inscan_checkpoint_timestamp",
    "origin_hub_outscan_checkpoint_timestamp",
    "destination_hub_inscan_checkpoint_timestamp",
    "promised_delivery_date", "estimated_delivery_date",
    "delivery_date", "rto_delivered_timestamp",
    "shipment_stuck_latest_timestamp",
    "rto_intransit_mark_timestamp",
]


def parse_csv(content: bytes | str) -> tuple[pd.DataFrame, dict]:
    """Parse a ClickPost CSV export into a DataFrame.

    Args:
        content: Raw CSV content as bytes or string.

    Returns:
        Tuple of (DataFrame, summary_dict).
        summary_dict contains: row_count, column_count, column_names,
        status_breakdown, carrier_breakdown.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    # Auto-detect delimiter: try tab first (ClickPost default), fall back to comma
    if "\t" in content.split("\n")[0]:
        sep = "\t"
    else:
        sep = ","

    print(f"[CSV] Detected delimiter: {'tab' if sep == '\t' else 'comma'}")

    df = pd.read_csv(io.StringIO(content), sep=sep, low_memory=False)

    # Normalize column names: strip whitespace, lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # Replace dash "-" with NaN (ClickPost uses "-" for null)
    df = df.replace("-", pd.NA)
    df = df.replace("null", pd.NA)
    df = df.replace("", pd.NA)

    # Parse date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Ensure numeric columns
    for col in ["uber_status_code", "ofd_count", "ndr_count", "npr_count",
                "invoice_value", "cod_amount", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map status codes to names
    if "uber_status_code" in df.columns:
        df["status_name"] = df["uber_status_code"].map(STATUS_CODES).fillna("UNKNOWN")

    # Build summary
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    if "uber_status_code" in df.columns:
        status_counts = df["status_name"].value_counts().head(10).to_dict()
        summary["status_breakdown"] = status_counts

    if "courier_partner_name" in df.columns:
        carrier_counts = df["courier_partner_name"].value_counts().head(10).to_dict()
        summary["carrier_breakdown"] = carrier_counts

    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# Widget Query Engine
# ─────────────────────────────────────────────────────────────────────────────

def resolve_widget_name(query: str) -> str | None:
    """Try to match a user query to a widget filter key."""
    query_lower = query.lower().strip()

    # Direct alias match
    for alias, widget_key in WIDGET_ALIASES.items():
        if alias in query_lower:
            return widget_key

    return None


def query_widget(df: pd.DataFrame, widget_key: str) -> dict:
    """Apply a widget's filter logic to the uploaded DataFrame.

    Args:
        df: Parsed shipment DataFrame.
        widget_key: Key from WIDGET_FILTERS dict.

    Returns:
        Dict with: widget_name, description, total_count, time_buckets (if applicable),
        carrier_breakdown, sample_awbs, shipment_table.
    """
    if widget_key not in WIDGET_FILTERS:
        return {"error": f"Unknown widget: {widget_key}"}

    config = WIDGET_FILTERS[widget_key]
    filtered = df.copy()

    # Filter by status codes
    if "status_codes" in config and "uber_status_code" in filtered.columns:
        filtered = filtered[filtered["uber_status_code"].isin(config["status_codes"])]
        print(f"  [filter] After status codes {config['status_codes']}: {len(filtered)} rows")
    elif "status_codes_exclude" in config and "uber_status_code" in filtered.columns:
        filtered = filtered[~filtered["uber_status_code"].isin(config["status_codes_exclude"])]
        print(f"  [filter] After excluding status codes: {len(filtered)} rows")

    # Apply conditions
    conditions = config.get("conditions", {})
    for col, condition in conditions.items():
        # Handle special condition suffixes
        if col.endswith("_lt"):
            real_col = col[:-3]
            if real_col in filtered.columns:
                filtered = filtered[pd.to_numeric(filtered[real_col], errors="coerce").fillna(0) < condition]
                print(f"  [filter] After {real_col} < {condition}: {len(filtered)} rows")
            continue
        if col.endswith("_gte"):
            real_col = col[:-4]
            if real_col in filtered.columns:
                filtered = filtered[pd.to_numeric(filtered[real_col], errors="coerce").fillna(0) >= condition]
                print(f"  [filter] After {real_col} >= {condition}: {len(filtered)} rows")
            continue

        if col not in filtered.columns:
            print(f"  [filter] ⚠️ Column '{col}' not found in data, skipping")
            continue

        if condition == "null":
            filtered = filtered[filtered[col].isna() | filtered[col].isnull()]
        elif condition == "not_null":
            filtered = filtered[filtered[col].notna() & filtered[col].notnull()]
        elif isinstance(condition, (int, float)):
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce").fillna(-999) == condition]

        print(f"  [filter] After {col} == {condition}: {len(filtered)} rows")

    result = {
        "widget_name": widget_key.replace("_", " ").title(),
        "description": config["description"],
        "total_count": len(filtered),
    }

    # Time bucket breakdown
    now = datetime.now(timezone.utc)

    if "time_buckets_hours" in config and config.get("time_column") in filtered.columns:
        time_col = config["time_column"]
        buckets = config["time_buckets_hours"]
        bucket_counts = {}

        for i, threshold in enumerate(buckets):
            cutoff = now - timedelta(hours=threshold)
            if i == 0:
                bucket_df = filtered[filtered[time_col] >= cutoff]
                label = f"0-{threshold}h"
            else:
                prev_cutoff = now - timedelta(hours=buckets[i - 1])
                bucket_df = filtered[
                    (filtered[time_col] < prev_cutoff) & (filtered[time_col] >= cutoff)
                ]
                label = f"{buckets[i-1]}-{threshold}h"
            bucket_counts[label] = len(bucket_df)

        # Last bucket: beyond the largest threshold
        last_cutoff = now - timedelta(hours=buckets[-1])
        beyond = filtered[filtered[time_col] < last_cutoff]
        bucket_counts[f"{buckets[-1]}h+"] = len(beyond)

        result["time_buckets"] = bucket_counts

    if "time_buckets_days" in config and config.get("time_column") in filtered.columns:
        time_col = config["time_column"]
        buckets = config["time_buckets_days"]
        bucket_counts = {}

        for i, threshold in enumerate(buckets):
            cutoff = now - timedelta(days=threshold)
            if i == 0:
                bucket_df = filtered[filtered[time_col] >= cutoff]
                label = f"0-{threshold}d"
            else:
                prev_cutoff = now - timedelta(days=buckets[i - 1])
                bucket_df = filtered[
                    (filtered[time_col] < prev_cutoff) & (filtered[time_col] >= cutoff)
                ]
                label = f"{buckets[i-1]}-{threshold}d"
            bucket_counts[label] = len(bucket_df)

        last_cutoff = now - timedelta(days=buckets[-1])
        beyond = filtered[filtered[time_col] < last_cutoff]
        bucket_counts[f"{buckets[-1]}d+"] = len(beyond)

        result["time_buckets"] = bucket_counts

    # Carrier breakdown
    if "courier_partner_name" in filtered.columns:
        carrier = filtered["courier_partner_name"].value_counts().head(5).to_dict()
        result["carrier_breakdown"] = carrier

    # Sample AWBs
    if "awb" in filtered.columns and len(filtered) > 0:
        result["sample_awbs"] = filtered["awb"].head(5).tolist()

    # Group by (e.g., RTO marked by attempt count)
    if "group_by" in config and config["group_by"] in filtered.columns:
        group_col = config["group_by"]
        grouped = filtered[group_col].fillna(0).astype(int).value_counts().sort_index().to_dict()
        result["grouped_by"] = {
            "column": group_col,
            "breakdown": {f"{k} attempts": v for k, v in grouped.items()},
        }

    # Shipment table — key columns for display
    display_cols = [
        c for c in [
            "awb", "status_name", "courier_partner_name", "created_at",
            "pickup_date", "drop_city", "drop_pin_code", "invoice_value",
            "ofd_count", "ndr_count", "rto_mark_date",
            "tracking_last_updated_at",
        ]
        if c in filtered.columns
    ]

    if display_cols and len(filtered) > 0:
        table_df = filtered[display_cols].copy()

        # Format dates to readable strings (drop timezone, keep date + time)
        for col in table_df.columns:
            if hasattr(table_df[col], "dt"):
                try:
                    table_df[col] = table_df[col].dt.strftime("%Y-%m-%d %H:%M").fillna("-")
                except Exception:
                    pass

        table_df = table_df.fillna("-")

        # Cap at 50 rows for display — full data stays in memory
        if len(table_df) > 50:
            result["shipment_table"] = table_df.head(50).to_dict(orient="records")
            result["table_truncated"] = True
            result["table_total"] = len(table_df)
        else:
            result["shipment_table"] = table_df.to_dict(orient="records")
            result["table_truncated"] = False

    return result


def get_data_summary(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of the uploaded data."""
    lines = [f"Loaded {len(df)} shipments."]

    if "status_name" in df.columns:
        top_statuses = df["status_name"].value_counts().head(5)
        lines.append("Top statuses:")
        for status, count in top_statuses.items():
            lines.append(f"  - {status}: {count}")

    if "courier_partner_name" in df.columns:
        top_carriers = df["courier_partner_name"].value_counts().head(5)
        lines.append("Top carriers:")
        for carrier, count in top_carriers.items():
            lines.append(f"  - {carrier}: {count}")

    return "\n".join(lines)