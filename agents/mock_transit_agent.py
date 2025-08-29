# agents/mock_transit_agent.py
from concordia.agent import Agent

class MockTransitAgent(Agent):
    def __init__(self, name, offer_dict):
        super().__init__(name=name)
        self.offer = offer_dict

    def act(self, state):
        if state.get("last_message", "").startswith("REQUEST"):
            state["offers"].append({"agency": self.name, **self.offer})
            return {"type": "OFFER", **self.offer}
        return None