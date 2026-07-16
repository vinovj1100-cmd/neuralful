"""Guardian module - System health monitoring and anomaly detection."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import deque


@dataclass
class GuardianConfig:
    """Configuration for the Guardian monitoring system."""
    health_threshold: float = 0.7
    alert_cooldown_minutes: int = 5
    max_alerts: int = 50


class Guardian:
    """System health monitor with configurable thresholds and recovery suggestions."""

    def __init__(self, config: GuardianConfig = None):
        self.config = config or GuardianConfig()
        self.health = 1.0
        self.alerts = []
        self.suggestions = []
        self.recovery = []
        self.tuner = {}
        self.ozone = {}
        self._started = False

    def start(self):
        """Initialize the guardian monitoring system."""
        self._started = True
        self.health = 1.0

    def analyze(self, context: Dict[str, Any]):
        """Analyze system context and generate health metrics, alerts, and suggestions."""
        self.alerts = []
        self.suggestions = []
        self.recovery = []

        # Calculate health score
        health_score = 1.0

        # Check inventory health
        inventory = context.get("inventory", {})
        total_skus = inventory.get("total_skus", 0)
        total_stock = inventory.get("total_stock", 0)

        if total_skus == 0:
            health_score -= 0.3
            self.alerts.append({"level": "WARNING", "message": "Inventory is empty — no SKUs registered"})
        elif total_stock < 10:
            health_score -= 0.2
            self.alerts.append({"level": "WARNING", "message": f"Low total stock: {total_stock} units"})

        # Check order backlog
        pending_orders = context.get("pending_orders", 0)
        if pending_orders > 25:
            health_score -= 0.25
            self.alerts.append({"level": "CRITICAL", "message": f"Order backlog critical: {pending_orders} pending orders"})
        elif pending_orders > 15:
            health_score -= 0.15
            self.alerts.append({"level": "WARNING", "message": f"Order backlog elevated: {pending_orders} pending orders"})

        # Check sync queue
        queue = context.get("sync_queue", {})
        queued = queue.get("queued", 0)
        if queued > 50:
            health_score -= 0.2
            self.alerts.append({"level": "WARNING", "message": f"Sync queue backlog: {queued} items pending"})

        # Check low stock
        low_stock = context.get("low_stock_skus", [])
        if len(low_stock) > 5:
            health_score -= 0.1
            self.alerts.append({"level": "INFO", "message": f"{len(low_stock)} SKUs below minimum stock threshold"})

        # Generate suggestions
        if pending_orders > 10:
            self.suggestions.append("Consider enabling batch order processing to reduce backlog")
        if queued > 20:
            self.suggestions.append("Run manual sync or enable auto-sync to clear queue")
        if len(low_stock) > 0:
            self.suggestions.append(f"Review and reorder {len(low_stock)} low-stock items")
        if total_skus < 5:
            self.suggestions.append("Import inventory data to populate the system")

        # Generate recovery actions
        if health_score < 0.5:
            self.recovery.append("Immediate: Process all pending orders and sync queue")
            self.recovery.append("Review staffing levels for order fulfillment")
        if health_score < 0.7:
            self.recovery.append("Review inventory levels and place emergency reorders")

        self.health = max(0.0, min(1.0, health_score))

        # Tuner data
        self.tuner = {
            "health_score": round(self.health, 2),
            "total_skus": total_skus,
            "total_stock": total_stock,
            "pending_orders": pending_orders,
            "queued_items": queued,
            "low_stock_count": len(low_stock),
            "failed_logins": context.get("failed_logins_last_5m", 0)
        }

        # Ozone metrics
        self.ozone = {
            "status": "HEALTHY" if self.health > 0.8 else "DEGRADED" if self.health > 0.5 else "CRITICAL",
            "uptime": "99.8%",
            "last_check": "Just now",
            "next_check": "In 5 minutes"
        }
