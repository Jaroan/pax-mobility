# oakley_paloalto_world.py
from concordia.gamemaster import GameMaster
from concordia.agent import Agent
from agents.fluffy_fluffyagentv16sub import FluffyPolicy  # assume we've refactored it into this import

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
            return f"REQUEST: {self.request}"
        # otherwise consult the Fluffy policy to accept/counter
        offers = state.get("offers", [])
        # build observation for policy
        obs = {"offers": offers, "self": {"preferences": {"speed": True}}}
        action_text = self.policy.propose_action(obs, state.get("memory", {}))
        return action_text

class TransitAgencyAgent(Agent):
    def __init__(self, name, offer_dict):
        super().__init__(name=name)
        self.offer = offer_dict
    def act(self, state):
        # when request arrives, post an OFFER structured in state
        if state.get("last_message", "").startswith("REQUEST"):
            return {"type": "OFFER", "agency": self.name, **self.offer}
        return None

# instantiate policy (wrap fluffy logic into POLICY class)
passenger_policy = FluffyPolicy()  # implement a thin wrapper around the finalist code

passenger = PassengerAgent(passenger_policy)
# Mock offers (eta_min, cost, mode, legs)
agents = [
    passenger,
    TransitAgencyAgent("TriDeltaTransit", {"eta_min": 135, "cost": 17, "mode": "TriDelta mixed"}),
    TransitAgencyAgent("BART", {"eta_min": 120, "cost": 25, "mode": "BART+Caltrain"}),
    TransitAgencyAgent("Archer", {"eta_min": 50, "cost": 120, "mode": "eVTOL via Concord"}),
    TransitAgencyAgent("Uber", {"eta_min": 100, "cost": 95, "mode": "direct_uber"})
]

gm = GameMaster(agents, initial_state={"request_posted": False, "offers": []})

# human input
passenger.observe_human_input("From Oakley to Palo Alto, leave now; prefer fastest")

gm.run(steps=10)  # run the turn loop
