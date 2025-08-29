import concordia
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs

from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

from agents.fluffy_policy import FluffyPolicy

from agents.bart_agent import BARTAgent
from agents.tridelta_agent import TriDeltaTransitAgent
from agents.mock_transit_agent import MockTransitAgent

class PassengerAgent(Agent):
    def __init__(self, policy):
        super().__init__(name="Passenger")
        self.policy = policy
        self.request = None

    def observe_human_input(self, text):
        self.request = text

    def act(self, state):
        # first turn: publish request
        if not state.get("request_posted"):
            state["request_posted"] = True
            return f"REQUEST: {self.request}"
        # otherwise consult the Fluffy policy to accept/counter
        offers = state.get("offers", [])
        # build observation for policy
        obs = {"offers": offers, "self": {"preferences": {"speed": True}}}
        return self.policy.propose_action(obs, state.get("memory", {}))


class TransitAgencyAgent(Agent):
    def __init__(self, name, offer_dict):
        super().__init__(name=name)
        self.offer = offer_dict
    def act(self, state):
        # when request arrives, post an OFFER structured in state
        if state.get("last_message", "").startswith("REQUEST"):
            state["offers"].append({"agency": self.name, **self.offer})
            return {"type": "OFFER", **self.offer}
        return None


# instantiate policy (wrap fluffy logic into POLICY class)
passenger_policy = FluffyPolicy()  # implement a thin wrapper around the finalist code

passenger = PassengerAgent(passenger_policy)
# Mock offers (eta_min, cost, mode, legs)
agents = [
    passenger,
    TriDeltaTransitAgent("data/tri_delta_transit_ntd-90162-202508110437.zip"),
    BARTAgent("data/bart_mdb-53-202508080326.zip"),
    MockTransitAgent("Archer", {"eta_min": 50, "cost": 120, "mode": "eVTOL via Concord"}),
    MockTransitAgent("Uber", {"eta_min": 100, "cost": 95, "mode": "direct_uber"})
]

gm = GameMaster(agents, initial_state={"request_posted": False, "offers": []})

# human input
passenger.observe_human_input("From Oakley to Palo Alto, leave now; prefer fastest")

gm.run(steps=10)  # run the turn loop
