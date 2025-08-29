# agents/tridelta_agent.py
import partridge as ptg
from concordia.agent import Agent
from datetime import datetime

class TriDeltaTransitAgent(Agent):
    def __init__(self, feed_path):
        super().__init__(name="TriDeltaTransit")
        self.feed_path = feed_path
        self.feed = ptg.load_feed(feed_path)
        # Map location names to nearest TriDelta stop_ids
        self.stop_map = {
            "Oakley": "OKLYPARK",  # Example stop_id, adjust to feed
            "Palo Alto": "CONCORDBART"  # We'll assume connection to BART
        }
        # Flat fare assumption
        self.fare_table = {("OKLYPARK", "CONCORDBART"): 3.0}

    def _time_to_secs(self, tstr):
        try:
            h, m, s = map(int, tstr.split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return None

    def _next_trip_eta(self, origin_id, dest_id, departure_time):
        stop_times = self.feed.stop_times

        origin_times = stop_times[stop_times["stop_id"] == origin_id]
        if origin_times.empty:
            return None

        now_secs = departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second

        origin_times = origin_times.copy()
        origin_times["dep_secs"] = origin_times["departure_time"].apply(self._time_to_secs)
        upcoming = origin_times[origin_times["dep_secs"] >= now_secs]
        if upcoming.empty:
            return None

        first_dep = upcoming.sort_values("dep_secs").iloc[0]
        trip_id = first_dep["trip_id"]

        dest_times = stop_times[(stop_times["trip_id"] == trip_id) & (stop_times["stop_id"] == dest_id)]
        if dest_times.empty:
            return None

        arr_secs = self._time_to_secs(dest_times.iloc[0]["arrival_time"])
        eta_min = (arr_secs - first_dep["dep_secs"]) / 60.0
        return eta_min

    def generate_offer(self, origin, destination, departure_time):
        origin_id = self.stop_map.get(origin)
        dest_id = self.stop_map.get(destination)
        if not origin_id or not dest_id:
            return None

        eta_min = self._next_trip_eta(origin_id, dest_id, departure_time)
        if eta_min is None:
            return None

        cost = self.fare_table.get((origin_id, dest_id), 2.5)
        return {"eta_min": round(eta_min), "cost": cost, "mode": "TriDelta bus"}

    def act(self, state):
        if state.get("last_message", "").startswith("REQUEST"):
            offer = self.generate_offer("Oakley", "Palo Alto", datetime.now())
            if offer:
                state["offers"].append({"agency": self.name, **offer})
                return {"type": "OFFER", **offer}
        return None