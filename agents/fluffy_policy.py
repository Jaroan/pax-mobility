# fluffy_policy.py
from agents.fluffy_fluffyagentv16sub import (
    ConcatActComponent,
)

from agents.utils import build_action_prompt, parse_final_action
# some_llm_client.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt, max_tokens=512, model="gpt-4o-mini"):
    """
    Call the LLM with the given prompt and return the response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

class FluffyPolicy:
    def __init__(self, character="passenger", max_tokens=1024):
        self.character = character
        self.max_tokens = max_tokens
        self.memory = []

    def propose_action(self, observation, memory):
        """
        observation: dict containing offers, preferences, etc.
        memory: previous conversation turns or summary
        """
        # Build a scenario prompt using original Fluffy templates
        prompt = build_action_prompt(
            role=self.character,
            observation=observation,
            memory=memory
        )

        # Add original Fluffy "ConcatActComponent"-style formatting
        concat_component = ConcatActComponent()
        final_prompt = concat_component.assemble(prompt)

        # Call your LLM backend
        llm_output = call_llm(final_prompt, max_tokens=self.max_tokens)

        # Keep the conversation short-term memory
        self.memory.append({"obs": observation, "output": llm_output})

        # Extract the actionable decision
        action = parse_final_action(llm_output)
        return action


