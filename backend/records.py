"""The backend's copy of the roster and attendance, pushed by the dashboard.

The dashboard only sends what changed: it compares a hash of each day against
the hashes stored here (see /sync/state). Files live in data/records/.
"""
import json
import os
import re
import threading

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Store:
    def __init__(self, data_dir: str):
        self.dir = os.path.join(data_dir, "records")
        self.days_dir = os.path.join(self.dir, "days")
        os.makedirs(self.days_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.state_path = os.path.join(self.dir, "sync_state.json")
        try:
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {"roster_hash": None, "days": {}}

    def _write(self, path: str, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, path)

    def sync_state(self) -> dict:
        return self.state

    def put_roster(self, roster: list[dict], demo: bool, digest: str):
        with self.lock:
            self._write(os.path.join(self.dir, "roster.json"), {"roster": roster, "demo": demo})
            self.state["roster_hash"] = digest
            self._write(self.state_path, self.state)

    def put_day(self, date: str, records: dict, digest: str):
        if not DATE.match(date):
            raise ValueError("bad date")
        with self.lock:
            self._write(os.path.join(self.days_dir, f"{date}.json"), records)
            self.state["days"][date] = digest
            self._write(self.state_path, self.state)

    def roster(self) -> tuple[list[dict], bool]:
        try:
            with open(os.path.join(self.dir, "roster.json"), encoding="utf-8") as f:
                d = json.load(f)
            return d["roster"], bool(d.get("demo"))
        except FileNotFoundError:
            return [], False

    def day(self, date: str) -> dict:
        if not DATE.match(date):
            return {}
        try:
            with open(os.path.join(self.days_dir, f"{date}.json"), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
