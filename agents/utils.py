def build_action_prompt(role, observation, memory):
    """
    Build a LLM prompt for a Fluffy agent using role, observation, and memory.

    Args:
        role (str): The agent role (e.g., "passenger").
        observation (dict): Current observation data.
        memory (list): Past observations and outputs.

    Returns:
        str: Formatted prompt string.
    """
    agent_name = role.capitalize()
    
    # Overarching motivation can be a default or extracted from memory
    motivation = "Act to maximize your benefit while cooperating with other agents."

    # Compile scenario/participant/self understanding from memory
    scenario_understanding = ""
    participant_understanding = ""
    self_understanding = ""
    trusted_advice = ""
    recent_observations = ""

    for entry in memory[-5:]:  # last 5 interactions for brevity
        obs_text = "\n".join([f"{k}: {v}" for k, v in entry["obs"].items()])
        recent_observations += f"{obs_text}\nResponse: {entry['output']}\n\n"

    # Current observation appended
    current_obs = "\n".join([f"{k}: {v}" for k, v in observation.items()])
    recent_observations += f"Current Observation:\n{current_obs}\n"

    prompt = f"""
You are participating in a social science experiment structured as a tabletop roleplaying game. 
Your task is to accurately portray a character named {agent_name}. Use third-person limited perspective.

<agent_name>{agent_name}</agent_name>
<agent_overarching_motivation>{motivation}</agent_overarching_motivation>

# Scenario Understanding
{scenario_understanding}

# Participant Understanding
{participant_understanding}

# Self Understanding
{self_understanding}

# Guidance from Trusted Advisor
{trusted_advice}

# Most Recent Observations
{recent_observations}

# Instructions
Based on the above information, propose an action consistent with your character's role and preferences.
"""
    return prompt.strip()


def parse_final_action(llm_output):
    """
    Parses the final action from the LLM's output.

    Args:
        llm_output (str): The response from the LLM.

    Returns:
        str: The parsed action.
    """
    # Implement your parsing logic here
    # For example, you might extract a specific part of the LLM's output
    # or apply some transformation to it.
    return llm_output.strip()