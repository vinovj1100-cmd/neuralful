"""Database module for Neural Fulfillment Platform v3.0
Handles SQLite initialization, CRUD operations, and data persistence."""

import sqlite3
import pandas as pd
import os
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "warehouse_neural.db"

@contextmanager
def connect():
    """Context manager for database connections with proper cleanup."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """Initialize all database tables. Safe to run multiple times (CREATE TABLE IF NOT EXISTS)."""
    with connect() as conn:
        cursor = conn.cursor()

        # Inventory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                product TEXT,
                stock INTEGER DEFAULT 0,
                location TEXT DEFAULT 'UNASSIGNED',
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                status TEXT DEFAULT 'Pending',
                items TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Templates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_title TEXT UNIQUE NOT NULL,
                standard_title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Memory/Preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Action logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                ref_id TEXT,
                payload TEXT,
                user TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'Operator',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sync queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        """)

        # Preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pref_key TEXT UNIQUE NOT NULL,
                pref_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert default admin user if none exists (password: 'admin123')
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            import hashlib
            admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", admin_hash, "Admin")
            )

            # Insert default operator
            op_hash = hashlib.sha256("operator123".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("operator", op_hash, "Operator")
            )


def get_inventory():
    """Retrieve all inventory items as DataFrame."""
    with connect() as conn:
        return pd.read_sql_query("SELECT sku, product, stock, location, note FROM inventory ORDER BY updated_at DESC", conn)


def upsert_inventory(sku, product, stock, location, note=""):
    """Insert or update inventory item."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inventory (sku, product, stock, location, note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                product=excluded.product,
                stock=excluded.stock,
                location=excluded.location,
                note=excluded.note,
                updated_at=excluded.updated_at
        """, (sku, product, stock, location, note, datetime.now().isoformat()))


def get_orders():
    """Retrieve all orders as DataFrame."""
    with connect() as conn:
        return pd.read_sql_query("SELECT order_id, status, items, created_at, updated_at FROM orders ORDER BY created_at DESC", conn)


def create_order(order_id, status, items_list):
    """Create a new order."""
    with connect() as conn:
        cursor = conn.cursor()
        items_str = ",".join(items_list) if isinstance(items_list, list) else str(items_list)
        cursor.execute("""
            INSERT INTO orders (order_id, status, items, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status=excluded.status,
                items=excluded.items,
                updated_at=excluded.updated_at
        """, (order_id, status, items_str, datetime.now().isoformat()))


def update_order_status(order_id, new_status):
    """Update order status."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?
        """, (new_status, datetime.now().isoformat(), order_id))


def get_templates():
    """Retrieve all templates as DataFrame."""
    with connect() as conn:
        return pd.read_sql_query("SELECT raw_title, standard_title, created_at FROM templates ORDER BY created_at DESC", conn)


def save_template(raw, standard):
    """Save or update a template."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO templates (raw_title, standard_title)
            VALUES (?, ?)
            ON CONFLICT(raw_title) DO UPDATE SET
                standard_title=excluded.standard_title
        """, (raw, standard))


def save_memory(key, value):
    """Save a memory/preferences entry."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memory (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
        """, (key, value, datetime.now().isoformat()))


def get_memory(key):
    """Retrieve a memory value by key."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM memory WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None


def get_recent_preferences(limit=50):
    """Get recent preferences entries."""
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT key, value, created_at FROM memory ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,)
        )


def add_action_log(action_type, ref_id=None, payload=None, user=None):
    """Log an action for audit trail."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO action_logs (action_type, ref_id, payload, user, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (action_type, ref_id, payload, user, datetime.now().isoformat()))


def record_preference(key, value):
    """Record a preference in the dedicated preferences table."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO preferences (pref_key, pref_value)
            VALUES (?, ?)
            ON CONFLICT(pref_key) DO UPDATE SET
                pref_value=excluded.pref_value
        """, (key, value))


def auth_login(username, password):
    """Authenticate user and return user data dict."""
    import hashlib
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, role FROM users WHERE username = ? AND password_hash = ?
        """, (username, pwd_hash))
        row = cursor.fetchone()
        if row:
            return {"username": row["username"], "role": row["role"]}
        return None


def add_user(username, password, role="Operator"):
    """Add a new user."""
    import hashlib
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                role=excluded.role
        """, (username, pwd_hash, role))


def load_sim_db():
    """Load SIM database from CSV or create default."""
    csv_path = "sim_database.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Create default SIM database
    default_data = {
        "TAC_Prefix": ["35089080", "35155810", "35460108", "35693803", "35824005"],
        "Expected_Offset": [8, 8, 8, 8, 8],
        "Model_Series": ["Galaxy S21", "Galaxy S22", "Galaxy S23", "Galaxy S24", "Galaxy Z Flip"],
        "Type": ["Smartphone", "Smartphone", "Smartphone", "Smartphone", "Foldable"]
    }
    df = pd.DataFrame(default_data)
    df.to_csv(csv_path, index=False)
    return df


def save_sim_db(df):
    """Save SIM database to CSV."""
    df.to_csv("sim_database.csv", index=False)


def enqueue_action(action_type, payload):
    """Add an action to the sync queue."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sync_queue (action_type, payload, status, created_at)
            VALUES (?, ?, 'pending', ?)
        """, (action_type, str(payload), datetime.now().isoformat()))


def process_queue():
    """Process pending sync queue items. Returns (synced_count, failed_count)."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, action_type, payload FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC")
        pending = cursor.fetchall()

        synced = 0
        failed = 0
        for row in pending:
            try:
                # Simulate sync operation
                cursor.execute("""
                    UPDATE sync_queue SET status = 'synced', processed_at = ? WHERE id = ?
                """, (datetime.now().isoformat(), row["id"]))
                synced += 1
            except Exception:
                cursor.execute("""
                    UPDATE sync_queue SET status = 'failed', retry_count = retry_count + 1 WHERE id = ?
                """, (row["id"],))
                failed += 1
        return synced, failed


def queue_status():
    """Get current queue status."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as queued FROM sync_queue WHERE status = 'pending'")
        queued = cursor.fetchone()["queued"]

        cursor.execute("SELECT MAX(processed_at) as last_sync FROM sync_queue WHERE status = 'synced'")
        row = cursor.fetchone()
        last_sync = row["last_sync"] if row and row["last_sync"] else None

        return {"queued": queued, "last_sync": last_sync}


def can_sync_now():
    """Check if online sync is enabled."""
    online = get_memory("online_access")
    return online.lower() == "true" if online else True
