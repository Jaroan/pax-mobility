import partridge as ptg
from concordia.agent import Agent
from datetime import datetime, timedelta
import pandas as pd


class BARTAgent(Agent):
    def __init__(self, feed_path):
        super().__init__(name="BART")
        self.feed_path = feed_path
        # self.feed = ptg.load_feed(feed_path, view={
        #     "trips.txt": None,
        #     "stop_times.txt": None
        # })
        self.feed = ptg.load_feed(feed_path)
        # Map location names to stop_ids in GTFS
        self.stop_map = {
            "Oakley": "ANTC",  # Nearest BART station: Antioch
            "Palo Alto": "MLBR"  # Nearest BART station: Millbrae (BART terminus)
        }
        # Flat fare model
        self.fare_table = {("ANTC", "MLBR"): 8.6}

    def _next_trip_eta(self, origin_id, dest_id, departure_time):
        """Return ETA in minutes from GTFS schedule."""
        # Load stops and stop_times
        stops = self.feed.stops
        stop_times = self.feed.stop_times
        trips = self.feed.trips

        # Filter stop_times for origin
        origin_times = stop_times[stop_times["stop_id"] == origin_id]
        if origin_times.empty:
            return None

        # Convert departure_time to seconds after midnight
        now_secs = departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second

        # Find first departure after now
        origin_times = origin_times.copy()
        origin_times["dep_secs"] = origin_times["departure_time"].apply(self._time_to_secs)
        upcoming = origin_times[origin_times["dep_secs"] >= now_secs]
        if upcoming.empty:
            return None

        first_dep = upcoming.sort_values("dep_secs").iloc[0]
        trip_id = first_dep["trip_id"]

        # Find arrival time at destination for same trip
        dest_times = stop_times[(stop_times["trip_id"] == trip_id) & (stop_times["stop_id"] == dest_id)]
        if dest_times.empty:
            # Fallback: Use departure time from origin as proxy
            arr_secs = first_dep["dep_secs"]
        else:
            arr_secs = self._time_to_secs(dest_times.iloc[0]["arrival_time"])

        eta_min = (arr_secs - first_dep["dep_secs"]) / 60.0
        return eta_min

    def _time_to_secs(self, tstr):
        """Convert HH:MM:SS to seconds after midnight."""
        try:
            h, m, s = map(int, tstr.split(":"))
            return h * 3600 + m * 60 + s
        except ValueError:
            return None

    def generate_offer(self, origin, destination, departure_time):
        origin_id = self.stop_map.get(origin)
        dest_id = self.stop_map.get(destination)
        if not origin_id or not dest_id:
            return None

        eta_min = self._next_trip_eta(origin_id, dest_id, departure_time)
        if eta_min is None:
            return None

        cost = self.fare_table.get((origin_id, dest_id), 5.0)
        offer = {
            "eta_min": round(eta_min),
            "cost": cost,
            "mode": "BART rail",
            "origin": origin,
            "destination": destination,
            "departure_time": departure_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return offer
    # Note: This is a simplified version. In a real implementation, you would
    # def generate_offer(self, origin, destination, departure_time):
    #     # TODO: map origin/destination to BART stop_ids
    #     # For now, just pick a fixed example:
    #     stop_from = "MONT"  # Montgomery St
    #     stop_to = "MLBR"    # Millbrae

    #     # Query next trip after departure_time
    #     trips = self.feed.trips.merge(self.feed.stop_times)
    #     # Simplified: just pick a mocked ETA for now
    #     eta_min = 45
    #     cost = 5.6
    #     return {"eta_min": eta_min, "cost": cost, "mode": "BART rail"}

    # def generate_offer(self, origin, destination, departure_time):
    #     # In a real version, you'd map origin/destination to stop_ids
    #     # and compute actual schedule. Here, we simulate ETA from GTFS headways.
    #     try:
    #         trips = self.feed.trips
    #         if trips.empty:
    #             return None
    #         eta_min = 45  # placeholder
    #         cost = 5.6
    #         return {"eta_min": eta_min, "cost": cost, "mode": "BART rail"}
    #     except Exception as e:
    #         print("Error in BARTAgent.generate_offer:", e)
    #         return None

    def act(self, state):
        if state.get("last_message", "").startswith("REQUEST"):
            offer = self.generate_offer("Oakley", "Palo Alto", datetime.now())
            if offer:
                state["offers"].append({"agency": self.name, **offer})
                return {"type": "OFFER", **offer}
        return None
