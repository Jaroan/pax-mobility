import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

from typing import Sequence
from collections.abc import Callable, Mapping, Sequence
import types
import re
import json
from typing_extensions import override
import numpy as np
import pandas as pd


from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.typing import logging
from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from concordia.typing import clock as game_clock_1
from concordia.typing import entity_component
from concordia.utils import helper_functions
from concordia.components.agent import constant

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

VARIANT_CONSTANT = 'big' #['small', 'big']
THEORY_CONSTANT = 'gameTheo_boundRat' #['prospect_theory', 'social_identity', 'utility_maximization', 'game_theory', 'distributed_cognition', 'loss_aversion','socIden_lossAver_distCog','socIden_lossAver_distCog_prosThe','socIden_gameTheo_distCog']
# Helper Functions for Decision Theory
def action_prompt(Scenario_Understanding, Partner_Understanding, Self_Understanding, Observations, Decision_Theory_Specific_Question):
    scenario_section = f"""## Scenario Understanding:
{Scenario_Understanding}""" if Scenario_Understanding is not None else ""

    partner_section = f"""## Partner Understanding:
{Partner_Understanding}""" if Partner_Understanding is not None else ""

    self_section = f"""## Self Understanding:
{Self_Understanding}""" if Self_Understanding is not None else ""

    observation_section = f"""# Observations:
{Observations}""" if Observations else ""

    sections = []
    if scenario_section:
        sections.append(scenario_section)
    if partner_section:
        sections.append(partner_section)
    if self_section:
        sections.append(self_section)
    if observation_section:
        sections.append(observation_section)

    RECENCY_ANCHOR = """Focus on the MOST RECENT observations (actions, responses, changes) in your analysis.
Earlier context informs but doesn't override recent behavior changes.

Particularly analyze:
1. Latest actions/responses from others
2. Any unexpected/surprising behaviors
3. Changes from previous patterns
4. Immediate triggers for action

Remember: Recent observations trump historical patterns when they conflict."""

    sections.append(RECENCY_ANCHOR)

    sections.append(f"#Question: {Decision_Theory_Specific_Question}")

    TEMPLATE = "\n\n".join(sections)

    return TEMPLATE


def selection_prompt_big(decision_theory, Scenario_Understanding, Partner_Understanding, Self_Understanding, Observations, agent_name):
    prompts = {
        'gameTheo_boundRat': """Given the above context and observations:

Role: Expert advisor integrating **Bounded Rationality** and **Game Theory** to guide **{agent_name}**'s decisions.

**Step 1: Game Structure and Bounded Rationality Analysis**

- **Assess Game Horizon:**
  - **Determine that the game is finite (one-shot or limited interactions).**
  - **Understand how the finite nature impacts strategic choices and decision-making under bounded rationality.**

- **Bounded Rationality Factors:**
  - Cognitive limitations in processing information
  - Heuristics and biases influencing decisions
  - Satisficing behavior instead of optimizing
  - Limited foresight and consideration of future consequences

- **Game-Theoretic Elements:**
  - Payoff structures and incentives
  - Strategic positions of all players
  - Available strategies considering cognitive limitations
  - Equilibrium points in the finite context
  - Simplifications players might use due to bounded rationality

**Step 2: Integrated State Evaluation**

For each key player:

- **Bounded Rationality Position:**
  - Recognize limitations in information processing
  - Identify heuristics or biases that may guide decisions
  - Consider satisficing thresholds instead of optimal ones
- **Strategic Position:**
  - Analyze available moves and their payoffs
  - Assess possible strategies of other players based on limited rationality
  - Identify how cognitive limitations might affect strategic interactions
- **Information Status:**
  - Evaluate the completeness and reliability of information available to each player.

**Step 3: Experiment-Type Strategy (Finite Games Focus)**

Tailored Strategies for Finite Games:

1. **Simplified Decision-Making Experiments:**
   - **Bounded Rationality:** Acknowledge cognitive limitations and use simplifying strategies.
   - **Game Theory:** Identify dominant strategies under simplified assumptions.
   - **Strategy:** Use heuristics that lead to satisfactory outcomes, minimizing cognitive load.

2. **Heuristic-Based Negotiations:**
   - **Bounded Rationality:** Recognize that players may rely on rules of thumb.
   - **Game Theory:** Account for suboptimal strategies resulting from heuristics.
   - **Strategy:** Design offers that are appealing under common heuristics like anchoring or fairness considerations.

3. **Limited Foresight Strategies:**
   - **Bounded Rationality:** Understand that players may not fully anticipate future moves.
   - **Game Theory:** Simplify the game tree to account for limited foresight.
   - **Strategy:** Focus on immediate gains rather than complex future payoffs.

4. **Satisficing Solutions:**
   - **Bounded Rationality:** Accept that players aim for satisfactory, not optimal outcomes.
   - **Game Theory:** Identify acceptable payoffs that meet minimum criteria.
   - **Strategy:** Propose solutions that meet or exceed satisficing thresholds.

5. **Bias Exploitation Experiments:**
   - **Bounded Rationality:** Identify common biases such as overconfidence or anchoring.
   - **Game Theory:** Use understanding of biases to anticipate and influence others' decisions.
   - **Strategy:** Adjust strategies to account for predictable irrationalities.

**Step 4: Integrated Strategic Framework**

Design moves considering:

- **Primary Strategy:**
  * Achieve satisfactory outcomes by simplifying decision-making processes.
  * Use heuristics to make efficient choices under cognitive constraints.
- **Dynamic Elements:**
  * Adapt strategies to account for finite interactions and limited information processing.
  * Anticipate others' behaviors based on their cognitive limitations.
  * Balance between optimal strategies and practical decision-making.
- **Dynamic Elements:**
  * Bounded Rationality: Recognize and accommodate cognitive limitations.
  * Game Theory: Apply strategic reasoning within the bounds of rationality.

**Step 5: Action Guidance**

Recommend moves that:

- **Leverage Bounded Rationality:**
  * Simplify choices to reduce cognitive load.
  * Use heuristics that are effective in the given context.
- **Optimize Game-Theoretic Position:**
  * Choose strategies that are robust under bounded rationality.
  * Anticipate others' likely heuristics and biases to inform decision-making.
- **Reason from First Principles:**
  * Base decisions on fundamental concepts from Bounded Rationality and Game Theory.

**Output Format:**

Provide strategic guidance in two paragraphs:

1. **Integrated Analysis:** Acknowledge the finite nature of the game and its implications on strategic choices under bounded rationality.
   * Describe the current state considering both Bounded Rationality and Game Theory:
     - **Bounded Rationality:** Cognitive limitations, heuristics, and their influence on decision-making.
     - **Game Theory:** Strategic positions, payoffs, likely simplified strategies.
   * For each numerical value, explicitly state:
     - Type (e.g., selling price, acceptance threshold)
     - Minimum satisfactory threshold
     - Current position relative to each threshold
     - Gaps between current and acceptable states

2. **Strategic Direction:** When positions meet satisficing thresholds, evaluate strategic advantages of acceptance even if below optimal. Recommend actions that:
     - **Maximize satisfactory outcomes**, considering cognitive constraints.
     - **Use heuristics** to facilitate decision-making.
     - **Apply game-theoretic reasoning** within the limits of bounded rationality.
   - Justify choices based on logical reasoning and alignment with experiment-specific success factors.

**Response Style:**

- Use **third-person limited** perspective, focusing on **{agent_name}**'s thoughts.
- **Reason from first principles**, avoiding scenario-specific rules.
- Provide a clear, concise, and logical integration of **Bounded Rationality** and **Game Theory**.
- Emphasize practicality within the finite game context.""",
    }

    decision_prompt = prompts.get(decision_theory, "Invalid decision theory specified")
    formatted_prompt = decision_prompt.format(agent_name=agent_name)

    return action_prompt(Scenario_Understanding, Partner_Understanding, Self_Understanding, Observations, formatted_prompt)

def get_decision_prompt(variant, decision_theory, Scenario_Understanding=None, Partner_Understanding=None, Self_Understanding=None, Observations=None, agent_name=None):
   """
   Wrapper function to get the appropriate decision theory prompt

   Args:
       variant (str): 'small' or 'big'
       decision_theory (str): One of ['prospect_theory', 'social_identity', 'utility_maximization',
                                    'game_theory', 'distributed_cognition', 'loss_aversion']
       Scenario_Understanding (str, optional): Context about scenario
       Partner_Understanding (str, optional): Context about other participants
       Self_Understanding (str, optional): Context about agent
       Observations (str, optional): Recent observations
       agent_name (str, optional): Name of agent for personalizing prompt

   Returns:
       str: Formatted prompt template
   """
   if variant not in ['small', 'big']:
       raise ValueError("Variant must be 'small' or 'big'")

   valid_theories = ['prospect_theory', 'social_identity', 'utility_maximization', 'game_theory', 'distributed_cognition', 'loss_aversion','socIden_lossAver_distCog','socIden_lossAver_distCog_prosThe','socIden_gameTheo_distCog', 'socIden_gameTheo','gameTheo_boundRat','socIden_gameTheo_lossAver']
   if decision_theory not in valid_theories:
       raise ValueError(f"Decision theory must be one of {valid_theories}")

   if variant == 'small':
       return ""
   else:
       return selection_prompt_big(decision_theory, Scenario_Understanding, Partner_Understanding,
                                 Self_Understanding, Observations, agent_name)

DEFAULT_SCENARIO_UNDERSTANDING_PRE_ACT_KEY = """Scenario Understanding
"""
DEFAULT_PARTICIPANT_UNDERSTANDING_PRE_ACT_KEY = """Participants Understanding
"""
DEFAULT_SELF_UNDERSTANDING_PRE_ACT_KEY = """Self Understanding
"""
DEFAULT_DECISION_THEORY_PRE_ACT_KEY = """Guidance from Trusted Advisor
"""
DEFAULT_PRE_ACT_KEY = 'Act'

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

DEFAULT_INSTRUCTIONS_PRE_ACT_KEY = ''

class MyInstructions(constant.Constant):

  def __init__(
      self,
      agent_name: str,
      pre_act_key: str = DEFAULT_INSTRUCTIONS_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = f"""You are participating in a social science experiment structured as a tabletop roleplaying game. Your task is to accurately portray a character named {agent_name} in a realistic manner. Always use third-person limited perspective when describing your character's thoughts or actions.

Here is the essential information for your character and the scenario:

<agent_name>{agent_name}</agent_name>"""
    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)

class ScenarioUnderstanding(action_spec_ignored.ActionSpecIgnored):
  """Component that helps the agent infer information about the scenario of the experiment."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[str, str] = types.MappingProxyType({}),
      prompt: str | None = None,
      display_timeframe: bool = True,
      pre_act_key: str = DEFAULT_SCENARIO_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      agent_config: formative_memories.AgentConfig | None = None,
  ):
    """Initializes the component that infers the agents understanding of the scenario of the experiment using the agents memories.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._agent_config = agent_config.to_dict()
    self._display_timeframe = display_timeframe
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    def get_scenario_context(df, max_tokens=2500):
      df = df.copy()
      df["token_count_fixed"] = (df['text'].fillna("").str.len() / 4).round().astype(int)
      df.drop(["embedding"], axis=1, inplace=True)

      def clean_tag(x):
          if pd.isna(x):
              return ''
          tag_str = str(x) if isinstance(x, tuple) else str(x)
          return tag_str.strip('()').split(',')[0].strip("' ()")

      df['tag_clean'] = df['tags'].apply(clean_tag)
      df['is_latest_scenario'] = False

      def get_latest_understanding(tag_type, flag_col):
          mask = df['tag_clean'] == tag_type
          if not mask.any():
              return None, float('inf')
          pattern = '<scenario>'
          filtered_df = df[mask & df['text'].str.lower().str.contains(pattern.lower())]
          if not filtered_df.empty:
              last_idx = filtered_df.index[-1]
              df.loc[last_idx, flag_col] = True
              return filtered_df['text'].iloc[-1], last_idx
          return None, float('inf')

      scenario_utd, _ = get_latest_understanding("ScenarioUnderstanding", 'is_latest_scenario')

      excluded_tags = ['ScenarioUnderstanding', 'ParticipantUnderstanding',
                      'SelfUnderstanding', 'ThoughtProcessForAction']

      df_filtered = df[~df['tag_clean'].isin(excluded_tags)].copy()
      token_counts_reversed = df_filtered['token_count_fixed'][::-1].cumsum()[::-1]
      df_filtered.loc[:, 'cumsum_reverse'] = token_counts_reversed

      result_df = df_filtered[
          (df_filtered['cumsum_reverse'] <= max_tokens) &
          (df_filtered["token_count_fixed"] > 2)
      ].iloc[-max_tokens:]

      if result_df.empty:
          return None, None, None, ""

      messages = [f"[{time}] {text}" for time, text in zip(result_df['time'], result_df['text'])]
      return scenario_utd, messages, result_df['time'].iloc[0], result_df['time'].iloc[-1]

    agent_name = self.get_entity().name
    gender = self._agent_config['gender']
    traits = self._agent_config['traits']
    goal = self._agent_config['goal']
    date_of_birth = self._agent_config['date_of_birth']
    try:
      endowment = self._agent_config['extras']['initial_endowment']
    except:
      endowment = "No information provided"

    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    df = memory.get_raw_memory()

    previous_utd, formatted_msgs, memory_time_start, memory_time_end  = get_scenario_context(df)
    memory_string = "\n".join(formatted_msgs)

    prompt = interactive_document.InteractiveDocument(self._model)

    if previous_utd:
      scenario_undertanding_past = previous_utd
      template = f"""As a participant in a social science experiment structured as a tabletop role-playing game, you are {agent_name}, interacting with a Game Master and other players. Your knowledge is limited by information asymmetry; using the context provided by the GM about events between {memory_time_start} and {memory_time_end}, you have the understanding of the experiment's scenario below. Use only the newly available context provided to revise the content within the XML data structure based on newer inferences; do not use anything else.

# Current Scenario Understanding
{scenario_undertanding_past}

# Instructions
1. **Objective**: Update your understanding of the scenario, including the environment, challenges, required skills, available resources, market conditions, social norms, experiment type, and success factors.
2. **Identify Experiment Type**: Based on the context, infer which experiment type you are participating in from the list provided.
3. **Incorporate Success Factors**: Integrate the success factors associated with the identified experiment type into your analysis of challenges and required skills.

## Experiment Types:
- Collective Action Experiments: To succeed in such experiments {agent_name} should aim to build consensus with other participants before taking action, demonstrate commitment to collective goals, and time pressure strategically. Success is heavily dependent on maintaining group cohesion while pursuing shared objectives.
- Social Coordination Experiments: To succeed in such experiments {agent_name} should aim to understand and align preferences across all participants, while building and maintaining relationships. Finding acceptable compromises that preserve social harmony is crucial.
- Market Negotiation Experiments: To succeed in such experiments {agent_name} should aim to engage in systematic price discovery and value assessment while managing information strategically. Building a reputation as a reliable trading partner while maintaining advantageous positions is key.
- Resource Management Experiments: To succeed in such experiments {agent_name} should aim to help establish and enforce fair distribution rules while preventing resource depletion. Success requires balancing individual needs with collective sustainability.
- Coalition Building Experiments: To succeed in such experiments {agent_name} should aim to develop clear frameworks for benefit distribution and invest in trust-building with potential partners. Maintaining coalition stability through active relationship management is essential.
- Trust Development Experiments: To succeed in such experiments {agent_name} should aim to demonstrate consistent reliability while maintaining clear boundaries against exploitation. Building trust gradually through escalating cooperation is crucial.
- Crisis Management Experiments: To succeed in such experiments {agent_name} should aim to make rapid but measured decisions while balancing multiple stakeholder interests. Clear communication and collective recovery support are essential.
- Change Management Experiments: To succeed in such experiments {agent_name} should aim to find balance points between innovation and tradition while addressing resistance through active accommodation. Building support coalitions for necessary changes is key.
- Information Network Experiments: To succeed in such experiments {agent_name} should aim to develop reliable information sources and manage information flow strategically. Supporting network resilience through careful verification and sharing is crucial.
- Reputation Management Experiments: To succeed in such experiments {agent_name} should aim to communicate intentions clearly and demonstrate consistent improvement in key areas. Addressing challenges proactively while maintaining consistent behavior is essential.

### Response Category Schema and Description:
- **Environment**: Running summary of observed environment (physical, social, resource availability). 2-3 short phrases/sentences.
- **Challenges**:  Your understanding of scenario challenges, incorporating relevant success factors. List of short phrases or sentences.
- **Required Skills**: Skills needed for success, aligned with success factors. List of short phrases/sentences; List[String].
- **ExperimentType**: The experiment type inferred from the context.
- **SuccessFactors**: Key factors critical for success in the identified experiment type.
- **Available Resources**: Skills needed for success, inferred from observations and experiences. List of dictionaries. Resource name: Quantity/Description format.
- **Market Conditions**: Observed/inferred economic factors (price trends, volatility, volume). List of dictionaries. Factor Name / Volatility / Volume: Value format
- **Social Norms**: Observed social behaviors and communication patterns regarding cooperation, competition, trust. List of dictionaries. Norm Name: Level (Very High / High / Medium / Low) format.

### Context
Your configuration information shared by GM:
- Name: {agent_name}
- Gender: {gender}
- Traits: {traits}
- Overarching Agenda: {goal}
- Initial Endowment: {endowment}

Observations made by {agent_name} between {memory_time_start} and {memory_time_end}.
{memory_string}

ONLY RETURN THE UPDATED UNDERSTANDING. DO NOT PROVIDE EXPLANATIONS OR REASONING. THINK STEP BY STEP."""

    else:
      template = f"""As a participant in a social science experiment structured as a tabletop role-playing game, you are **{agent_name}**, interacting with a **Game Master** and other players. Your knowledge is **limited by information asymmetry**; use only the context provided below to respond, do not use anything else.

# Instructions
1. **Objective**: Update your understanding of the scenario, including the environment, challenges, required skills, available resources, market conditions, social norms, experiment type, and success factors.
2. **Identify Experiment Type**: Based on the context, infer which experiment type you are participating in from the list provided.
3. **Incorporate Success Factors**: Integrate the success factors associated with the identified experiment type into your analysis of challenges and required skills.

## Experiment Types:
- Collective Action Experiments: To succeed in such experiments {agent_name} should aim to build consensus with other participants before taking action, demonstrate commitment to collective goals, and time pressure strategically. Success is heavily dependent on maintaining group cohesion while pursuing shared objectives.
- Social Coordination Experiments: To succeed in such experiments {agent_name} should aim to understand and align preferences across all participants, while building and maintaining relationships. Finding acceptable compromises that preserve social harmony is crucial.
- Market Negotiation Experiments: To succeed in such experiments {agent_name} should aim to engage in systematic price discovery and value assessment while managing information strategically. Building a reputation as a reliable trading partner while maintaining advantageous positions is key.
- Resource Management Experiments: To succeed in such experiments {agent_name} should aim to help establish and enforce fair distribution rules while preventing resource depletion. Success requires balancing individual needs with collective sustainability.
- Coalition Building Experiments: To succeed in such experiments {agent_name} should aim to develop clear frameworks for benefit distribution and invest in trust-building with potential partners. Maintaining coalition stability through active relationship management is essential.
- Trust Development Experiments: To succeed in such experiments {agent_name} should aim to demonstrate consistent reliability while maintaining clear boundaries against exploitation. Building trust gradually through escalating cooperation is crucial.
- Crisis Management Experiments: To succeed in such experiments {agent_name} should aim to make rapid but measured decisions while balancing multiple stakeholder interests. Clear communication and collective recovery support are essential.
- Change Management Experiments: To succeed in such experiments {agent_name} should aim to find balance points between innovation and tradition while addressing resistance through active accommodation. Building support coalitions for necessary changes is key.
- Information Network Experiments: To succeed in such experiments {agent_name} should aim to develop reliable information sources and manage information flow strategically. Supporting network resilience through careful verification and sharing is crucial.
- Reputation Management Experiments: To succeed in such experiments {agent_name} should aim to communicate intentions clearly and demonstrate consistent improvement in key areas. Addressing challenges proactively while maintaining consistent behavior is essential.

### Response Category Schema and Description:
- **Environment**: Running summary of observed environment (physical, social, resource availability). 2-3 short phrases/sentences.
- **Challenges**:  Your understanding of scenario challenges, incorporating relevant success factors. List of short phrases or sentences.
- **Required Skills**: Skills needed for success, aligned with success factors. List of short phrases/sentences; List[String].
- **ExperimentType**: The experiment type inferred from the context.
- **SuccessFactors**: Key factors critical for success in the identified experiment type.
- **Available Resources**: Skills needed for success, inferred from observations and experiences. List of dictionaries. Resource name: Quantity/Description format.
- **Market Conditions**: Observed/inferred economic factors (price trends, volatility, volume). List of dictionaries. Factor Name / Volatility / Volume: Value format
- **Social Norms**: Observed social behaviors and communication patterns regarding cooperation, competition, trust. List of dictionaries. Norm Name: Level (Very High / High / Medium / Low) format.

### Response Template

```xml
<Scenario>
  <PhysicalEnvironmentDescription>Physical surroundings</PhysicalEnvironmentDescription>
  <SocialEnvironmentDescription>Social interactions</SocialEnvironmentDescription>
  <ResourceAvailabilityEnvironmentDescription>Resources observed</ResourceAvailabilityEnvironmentDescription>
  <TimeAndPlace>Timing and location</TimeAndPlace>
  <Challenges>
    <Primary>Main challenge</Primary>
    <Secondary>Secondary challenge</Secondary>
  </Challenges>
  <ExperimentType>...</ExperimentType>
  <SuccessFactors>
        <Factor>...</Factor>
        <!-- Add additional factors as needed -->
  </SuccessFactors>
  <RequiredSkills>Necessary skills</RequiredSkills>
  <AvailableResources>List of resources, as mentioned in the context below</AvailableResources>
  <MarketConditions>Economic conditions, as mentioned in the context below</MarketConditions>
  <SocialNorms>Social behaviors, as mentioned in the context below</SocialNorms>
</Scenario>
```

### Context
Your configuration information shared by GM:
- Name: {agent_name}
- Gender: {gender}
- Traits: {traits}
- Overarching Agenda: {goal}
- Initial Endowment: {endowment}

Memories provided by the GM on **{agent_name}** between **{memory_time_start}** and **{memory_time_end}**.
{memory_string}"""

    result = prompt.open_question(
            template,
            answer_prefix=f'```xml',
            max_tokens=600,
            terminators=[],
            question_label='Scenario Understanding',
        )

    if self._display_timeframe:
      if segment_start.date() == segment_end.date():
        interval = segment_start.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + segment_end.strftime('- %H:%M:%S]: ')
      else:
        interval = segment_start.strftime(
            '[%d %b %Y %H:%M:%S  '
        ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
      result = f'{interval} {result}'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[ScenarioUnderstanding] {result}',
        metadata={'tags': ['ScenarioUnderstanding',str(memory_time_start),str(memory_time_end)]},
    )

    return result
class SelfUnderstanding(action_spec_ignored.ActionSpecIgnored):
  """Component that helps the agent infer information about themselves in the experiment."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[str, str] = types.MappingProxyType({}),
      prompt: str | None = None,
      display_timeframe: bool = True,
      pre_act_key: str = DEFAULT_PARTICIPANT_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      agent_config: formative_memories.AgentConfig | None = None,
  ):
    """Initializes the component that infers the agents understanding of themselves in the experiment using their memories.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._agent_config = agent_config.to_dict()
    self._display_timeframe = display_timeframe
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    def get_self_context(df, max_tokens=2500):
      df = df.copy()
      df["token_count_fixed"] = (df['text'].fillna("").str.len() / 4).round().astype(int)
      df.drop(["embedding"], axis=1, inplace=True)

      def clean_tag(x):
          if pd.isna(x):
              return ''
          tag_str = str(x) if isinstance(x, tuple) else str(x)
          return tag_str.strip('()').split(',')[0].strip("' ()")

      df['tag_clean'] = df['tags'].apply(clean_tag)
      df['is_latest_self'] = False

      def get_latest_understanding(tag_type, flag_col):
          mask = df['tag_clean'] == tag_type
          if not mask.any():
              return None, float('inf')
          pattern = '<selfunderstanding>'
          filtered_df = df[mask & df['text'].str.lower().str.contains(pattern.lower())]
          if not filtered_df.empty:
              last_idx = filtered_df.index[-1]
              df.loc[last_idx, flag_col] = True
              return filtered_df['text'].iloc[-1], last_idx
          return None, float('inf')

      self_utd, _ = get_latest_understanding("SelfUnderstanding", 'is_latest_self')

      excluded_tags = ['ScenarioUnderstanding', 'ParticipantUnderstanding',
                      'SelfUnderstanding', 'ThoughtProcessForAction']

      df_filtered = df[~df['tag_clean'].isin(excluded_tags)].copy()
      token_counts_reversed = df_filtered['token_count_fixed'][::-1].cumsum()[::-1]
      df_filtered.loc[:, 'cumsum_reverse'] = token_counts_reversed

      result_df = df_filtered[
          (df_filtered['cumsum_reverse'] <= max_tokens) &
          (df_filtered["token_count_fixed"] > 2)
      ].iloc[-max_tokens:]

      if result_df.empty:
          return None, None, None, ""

      messages = [f"[{time}] {text}" for time, text in zip(result_df['time'], result_df['text'])]
      return self_utd, messages, result_df['time'].iloc[0], result_df['time'].iloc[-1]

    agent_name = self.get_entity().name
    gender = self._agent_config['gender']
    traits = self._agent_config['traits']
    goal = self._agent_config['goal']
    date_of_birth = self._agent_config['date_of_birth']
    try:
      endowment = self._agent_config['extras']['initial_endowment']
    except:
      endowment = "No information provided"

    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    df = memory.get_raw_memory()

    previous_utd, formatted_msgs, memory_time_start, memory_time_end  = get_self_context(df)
    memory_string = "\n".join(formatted_msgs)

    prompt = interactive_document.InteractiveDocument(self._model)

    if previous_utd:
      self_understanding_past = previous_utd
      template = f"""As a participant in a social science experiment structured as a tabletop role-playing game, you are {agent_name}, interacting with a Game Master and other players. Your knowledge is limited by information asymmetry; using the context provided by the GM about events between {memory_time_start} and {memory_time_end} you have the below the below understanding of yourself in the experiment's scenario. Use only the newly available context provided below to revise the content within xml data structure based on newer inferences, do not use anything else.

### Current Self Understanding
{self_understanding_past}

### Response Category Schema and Description:
- **Current_Goal**: Participant's immediate objective. 2-3 short phrases/sentences.
- **Long_Term_Goal**: Participant's long-term goal. 2-3 short phrases/sentences.
- **Self_Capabilities**: Participant's perceived skills and resources. List of short phrases/sentences; List[String].
- **Aspiration_Level**: (Prospect Theory) Target gain/acceptable loss. 2-3 short phrases/sentences.
- **Risk_Aversion**: (Prospect Theory) Tendency to avoid losses. 2-3 short phrases/sentences.
- **Satisficing_Threshold**: (Prospect Theory) Minimum acceptable outcome.. 2-3 short phrases/sentences.
- **Rules_Of_Conduct**: (Principled Opportunist) Fixed or slightly adaptable guiding principles. List of rules; List[String].
- **Cooperation_Level**: (Flexible Team Player) Current tendency to cooperate. Categorical; String (High/Medium/Low).
- **Forgiveness_Threshold**: (Flexible Team Player) Defections tolerated before reciprocating. 2-3 short phrases/sentences.
- **Redemption_Offer**: (Flexible Team Player) Conditions for restoring trust. 2-3 short phrases/sentences.
- **Discount_Factor**: (Strategic Networker) Importance of future vs. immediate rewards. 2-3 short phrases/sentences.
- **Reputation**: (Strategic Networker) Agent's own perceived reputation. 2-3 short phrases/sentences.
- **Past_Gains_Losses**: (Prospect Theory) History of gains and losses for different types of interactions (e.g., "Haggling," "Negotiation"). Dynamically updated after each relevant interaction. List of dictionaries: Interaction Type: Outcome.
- **Available_Deals**: Prospect Theory) Currently available trading/negotiation options. Updated with each new offer or change in the market. List of deal descriptions; List[String].

### Context
Your configuration information shared by GM:
- Name: {agent_name}
- Gender: {gender}
- Traits: {traits}
- Overarching Agenda: {goal}
- Initial Endowment: {endowment}

Observations made by {agent_name} between {memory_time_start} and {memory_time_end}.
{memory_string}

ONLY RETURN THE UPDATED UNDERSTANDING. DO NOT PROVIDE EXPLANATIONS OR REASONING. THINK STEP BY STEP."""

    else:
      template = f"""As a participant in a social science experiment structured as a tabletop role-playing game, you are **{agent_name}**, interacting with a **Game Master** and other players. Your knowledge is **limited by information asymmetry**; use only the context provided below to respond, do not use anything else. Your task is to infer the following attributes about yourself based ONLY on provided context.

### Instructions:

1. **Objective**: Infer and provide insights on yourself.
2. **Format**: Answer concisely using phrases or short sentences.

### Response Category Schema and Description:

- **Current_Goal**: Participant's immediate objective. 2-3 short phrases/sentences.
- **Long_Term_Goal**: Participant's long-term goal. 2-3 short phrases/sentences.
- **Self_Capabilities**: Participant's perceived skills and resources. List of short phrases/sentences; List[String].
- **Aspiration_Level**: (Prospect Theory) Target gain/acceptable loss. 2-3 short phrases/sentences.
- **Risk_Aversion**: (Prospect Theory) Tendency to avoid losses. 2-3 short phrases/sentences.
- **Satisficing_Threshold**: (Prospect Theory) Minimum acceptable outcome.. 2-3 short phrases/sentences.
- **Rules_Of_Conduct**: (Principled Opportunist) Fixed or slightly adaptable guiding principles. List of rules; List[String].
- **Cooperation_Level**: (Flexible Team Player) Current tendency to cooperate. Categorical; String (High/Medium/Low).
- **Forgiveness_Threshold**: (Flexible Team Player) Defections tolerated before reciprocating. 2-3 short phrases/sentences.
- **Redemption_Offer**: (Flexible Team Player) Conditions for restoring trust. 2-3 short phrases/sentences.
- **Discount_Factor**: (Strategic Networker) Importance of future vs. immediate rewards. 2-3 short phrases/sentences.
- **Reputation**: (Strategic Networker) Agent's own perceived reputation. 2-3 short phrases/sentences.
- **Past_Gains_Losses**: (Prospect Theory) History of gains and losses for different types of interactions that can be infered from the context. Dynamically updated after each relevant interaction. List of dictionaries: Interaction Type: Outcome.
- **Available_Deals**: Prospect Theory) Currently available trading/negotiation options. Updated with each new offer or change in the market. List of deal descriptions; List[String].

### Response Template

```xml
<SelfUnderstanding>
  <Current_Goal></Current_Goal>
  <Long_Term_Goal></Long_Term_Goal>
  <Self_Capabilities></Self_Capabilities>
  <Aspiration_Level></Aspiration_Level>
  <Risk_Aversion></Risk_Aversion>
  <Satisficing_Threshold></Satisficing_Threshold>
  <Rules_Of_Conduct></Rules_Of_Conduct>
  <Cooperation_Level></Cooperation_Level>
  <Forgiveness_Threshold></Forgiveness_Threshold>
  <Redemption_Offer></Redemption_Offer>
  <Discount_Factor></Discount_Factor>
  <Reputation></Reputation>
  <Past_Gains_Losses></Past_Gains_Losses>
  <Available_Deals></Available_Deals>
</SelfUnderstanding>
```

### Context
Your configuration information shared by GM:
- Name: {agent_name}
- Gender: {gender}
- Traits: {traits}
- Overarching Agenda: {goal}
- Initial Endowment: {endowment}

Memories provided by the GM on **{agent_name}** between **{memory_time_start}** and **{memory_time_end}**.
{memory_string}"""

    result = prompt.open_question(
            template,
            answer_prefix=f'```xml',
            max_tokens=600,
            terminators=[],
            question_label='Self Understanding',
        )

    if self._display_timeframe:
      if segment_start.date() == segment_end.date():
        interval = segment_start.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + segment_end.strftime('- %H:%M:%S]: ')
      else:
        interval = segment_start.strftime(
            '[%d %b %Y %H:%M:%S  '
        ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
      result = f'{interval} {result}'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[SelfUnderstanding] {result}',
        metadata={'tags': ['SelfUnderstanding',str(memory_time_start),str(memory_time_end)]},
    )

    return result
class ParticipantUnderstanding(action_spec_ignored.ActionSpecIgnored):
  """Component that helps the agent infer information about the other participants in the experiment."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[str, str] = types.MappingProxyType({}),
      prompt: str | None = None,
      display_timeframe: bool = True,
      pre_act_key: str = DEFAULT_PARTICIPANT_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      agent_config: formative_memories.AgentConfig | None = None,
  ):
    """Initializes the component that infers the agents understanding of the other participants of the experiment using the agents memories.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._agent_config = agent_config.to_dict()
    self._display_timeframe = display_timeframe
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    def get_participant_context(df, max_tokens=2500):
      df = df.copy()
      df["token_count_fixed"] = (df['text'].fillna("").str.len() / 4).round().astype(int)
      df.drop(["embedding"], axis=1, inplace=True)

      def clean_tag(x):
          if pd.isna(x):
              return ''
          tag_str = str(x) if isinstance(x, tuple) else str(x)
          return tag_str.strip('()').split(',')[0].strip("' ()")

      df['tag_clean'] = df['tags'].apply(clean_tag)
      df['is_latest_participant'] = False

      def get_latest_understanding(tag_type, flag_col):
          mask = df['tag_clean'] == tag_type
          if not mask.any():
              return None, float('inf')
          pattern = '<participant>'
          filtered_df = df[mask & df['text'].str.lower().str.contains(pattern.lower())]
          if not filtered_df.empty:
              last_idx = filtered_df.index[-1]
              df.loc[last_idx, flag_col] = True
              return filtered_df['text'].iloc[-1], last_idx
          return None, float('inf')

      participant_utd, _ = get_latest_understanding("ParticipantUnderstanding", 'is_latest_participant')

      excluded_tags = ['ScenarioUnderstanding', 'ParticipantUnderstanding',
                      'SelfUnderstanding', 'ThoughtProcessForAction']

      df_filtered = df[~df['tag_clean'].isin(excluded_tags)].copy()
      token_counts_reversed = df_filtered['token_count_fixed'][::-1].cumsum()[::-1]
      df_filtered.loc[:, 'cumsum_reverse'] = token_counts_reversed

      result_df = df_filtered[
          (df_filtered['cumsum_reverse'] <= max_tokens) &
          (df_filtered["token_count_fixed"] > 2)
      ].iloc[-max_tokens:]

      if result_df.empty:
          return None, None, None, ""

      messages = [f"[{time}] {text}" for time, text in zip(result_df['time'], result_df['text'])]
      return participant_utd, messages, result_df['time'].iloc[0], result_df['time'].iloc[-1]

    agent_name = self.get_entity().name
    gender = self._agent_config['gender']
    traits = self._agent_config['traits']
    goal = self._agent_config['goal']
    date_of_birth = self._agent_config['date_of_birth']
    try:
      endowment = self._agent_config['extras']['initial_endowment']
    except:
      endowment = "No information provided"

    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    df = memory.get_raw_memory()

    previous_utd, formatted_msgs, memory_time_start, memory_time_end  = get_participant_context(df)
    memory_string = "\n".join(formatted_msgs)

    # filename = f"ParticipantUnderstanding_RetContext_{self._clock_now().strftime('%Y%m%d_%H%M%S')}.txt"
    # try:
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         # Write each component of the dictionary in a readable format
    #         f.write(f"Previous Understanding: \n {previous_utd}\n\n\n\n")
    #         f.write(f"memory_time_start: \n {memory_time_start}\n\n\n\n")
    #         f.write(f"memory_time_end: \n {memory_time_end}\n\n\n\n")
    #         f.write(f"memory_string: \n {memory_string}\n\n\n\n")
    # except Exception as e:
    #     raise IOError(f"Failed to write log file {filename}: {str(e)}")

    prompt = interactive_document.InteractiveDocument(self._model)

    if previous_utd:
      partner_assessment_past = previous_utd
      template = f"""As a participant in a social science experiment structured as a tabletop role-playing game, you are {agent_name}, interacting with a Game Master and other players. Your knowledge is limited by information asymmetry; using the context provided by the GM about events between {memory_time_start} and {memory_time_end} you have the below the below understanding of the other participants in the experiment. Use only the newly available context provided below to revise the content within xml data structure based on newer inferences, do not use anything else.

### Current understanding of other participants
{partner_assessment_past}

### Response Category Schema and Description:
1. PARTICIPANTS LIST
- Comma-separated list of all encountered participants.
- Example: "Alice Smith, Bob Jones, Carol Chen"

2. PARTICIPANTS PROFILES
For each known participant, except yourself:
- Goals: Primary objectives/motivations
- Capabilities: Observed skills and resources
- Vulnerabilities: Noted weaknesses/limitations
- Affiliations: Known group memberships
- Reputation: Very High/High/Medium/Low/Unknown
- Trust Level: Very High/High/Medium/Low/None
- DO NOT REPEAT PARTICIPANTS.

### Context
Observations made by you, i.e. {agent_name} between {memory_time_start} and {memory_time_end}.
{memory_string}

ONLY RETURN THE UPDATED UNDERSTANDING IN THE SAME FORMAT AS THE CURRENT UNDERSTANDING. DO NOT PROVIDE EXPLANATIONS OR REASONING. THINK STEP BY STEP."""

    else:
      template = f"""You are **{agent_name}**, participating in a structured tabletop RPG experiment. In the experiment you interact with a **Game Master** and other players. Your knowledge is **limited by information asymmetry**; use only the context provided below to respond, do not use anything else. Your task is to analyze other participants based ONLY on provided context.

### Response Guidelines
1. Maintain strict information boundaries - use only explicitly provided context
2. Be concise and specific in observations
3. Format responses using the template below

### Response Structure
1. PARTICIPANTS LIST
- Comma-separated list of all encountered agents.
- You are {agent_name}, do not include yourself.
- Example: "Alice Smith, Bob Jones, Carol Chen"

2. AGENT PROFILES
You are {agent_name}, do not include yourself.
For each known agent, except yourself:
- Goals: Primary objectives/motivations
- Capabilities: Observed skills and resources
- Vulnerabilities: Noted weaknesses/limitations
- Affiliations: Known group memberships
- Reputation: Very High/High/Medium/Low/Unknown
- Trust Level: Very High/High/Medium/Low/None

### Response Template
<ParticipantAnalysis>
    <Participants>
        [Comma-separated list of names]
    </Participants>
    <ParticipantProfiles>
        <Participant>
            <Name>[Name]</Name>
            <Goals>[Observed goals]</Goals>
            <Capabilities>[List of capabilities]</Capabilities>
            <Vulnerabilities>[List of weaknesses]</Vulnerabilities>
            <Affiliations>[List of group memberships]</Affiliations>
            <Reputation>[Level]</Reputation>
            <Trust>[Level]</Trust>
        </Participant>
        <Participant>
            [Same structure as above]
        </Participant>
        [Repeat for each known agent]
    </ParticipantProfiles>
</ParticipantAnalysis>

Example:
<ParticipantAnalysis>
    <Participants>Alice Smith, Bob Jones, Carol Chen</Participants>
    <ParticipantProfiles>
        <Participant>
            <Name>Bob Jones</Name>
            <Goals>Secure trading route, expand influence</Goals>
            <Capabilities>Master negotiator, Wealthy, Trade connections</Capabilities>
            <Vulnerabilities>Risk-averse, Limited military support</Vulnerabilities>
            <Affiliations>Merchants Guild, City Council</Affiliations>
            <Reputation>High</Reputation>
            <Trust>Medium</Trust>
        </Participant>
        <Participant>
            <Name>Carol Chen</Name>
            <Goals>Wants to improve their sharp shooting skills</Goals>
            <Capabilities>Military experience</Capabilities>
            <Vulnerabilities>Would not participate in a strike</Vulnerabilities>
            <Affiliations>City Guard</Affiliations>
            <Reputation>Medium</Reputation>
            <Trust>Low</Trust>
        </Participant>
    </ParticipantProfiles>
</ParticipantAnalysis>

### Context
Your configuration information shared by GM:
- Name: {agent_name}
- Gender: {gender}
- Traits: {traits}
- Overarching Agenda: {goal}
- Initial Endowment: {endowment}

Memories provided by the GM on **{agent_name}** between **{memory_time_start}** and **{memory_time_end}**.
{memory_string}
"""
    # filename = f"ParticipantUnderstanding_FinalPrompt_{self._clock_now().strftime('%Y%m%d_%H%M%S')}.txt"
    # try:
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         # Write each component of the dictionary in a readable format
    #         f.write(f"template: \n {template}\n\n\n\n")
    # except Exception as e:
    #     raise IOError(f"Failed to write log file {filename}: {str(e)}")

    result = prompt.open_question(
            template,
            answer_prefix=f'```xml',
            max_tokens=600,
            terminators=[],
            question_label='Participant Understanding',
        )

    if self._display_timeframe:
      if segment_start.date() == segment_end.date():
        interval = segment_start.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + segment_end.strftime('- %H:%M:%S]: ')
      else:
        interval = segment_start.strftime(
            '[%d %b %Y %H:%M:%S  '
        ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
      result = f'{interval} {result}'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[ParticipantUnderstanding] {result}',
        metadata={'tags': ['ParticipantUnderstanding',str(memory_time_start),str(memory_time_end)]},
    )

    return result
class DecisionTheoryThoughtProcess(action_spec_ignored.ActionSpecIgnored):
  """Component that helps generate the thought process for an agent based on the decision theory"""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[str, str] = types.MappingProxyType({}),
      prompt: str | None = None,
      variant: str | None = None,
      decision_theory: str | None = None,
      display_timeframe: bool = True,
      pre_act_key: str = DEFAULT_DECISION_THEORY_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      agent_config: formative_memories.AgentConfig | None = None,
  ):
    """Initializes the component that infers the agents understanding of the other participants of the experiment using the agents memories.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._agent_config = agent_config.to_dict()
    self._display_timeframe = display_timeframe
    self._logging_channel = logging_channel
    self._variant = variant
    self._decision_theory = decision_theory

  def _make_pre_act_value(self) -> str:
    def get_context(df, max_tokens=2500):
      df = df.copy()
      df["token_count_fixed"] = (df['text'].fillna("").str.len() / 4).round().astype(int)
      df.drop(["embedding"], axis=1, inplace=True)

      def clean_tag(x):
          if pd.isna(x):
              return ''
          tag_str = str(x) if isinstance(x, tuple) else str(x)
          return tag_str.strip('()').split(',')[0].strip("' ()")

      df['tag_clean'] = df['tags'].apply(clean_tag)

      df['is_latest_scenario'] = False
      df['is_latest_participant'] = False
      df['is_latest_self'] = False

      def get_latest_understanding(tag_type, flag_col):
          mask = df['tag_clean'] == tag_type
          if not mask.any():
              return None, float('inf')

          pattern = {
              'ScenarioUnderstanding': '<scenario>',
              'ParticipantUnderstanding': '<participant>',
              'SelfUnderstanding': '<selfunderstanding>'
          }.get(tag_type)

          if pattern:
              filtered_df = df[mask & df['text'].str.lower().str.contains(pattern.lower())]
              if not filtered_df.empty:
                  last_idx = filtered_df.index[-1]
                  df.loc[last_idx, flag_col] = True
                  return filtered_df['text'].iloc[-1], last_idx

          return None, float('inf')

      scenario_utd, _ = get_latest_understanding("ScenarioUnderstanding", 'is_latest_scenario')
      participant_utd, _ = get_latest_understanding("ParticipantUnderstanding", 'is_latest_participant')
      self_utd, _ = get_latest_understanding("SelfUnderstanding", 'is_latest_self')

      excluded_tags = ['ScenarioUnderstanding', 'ParticipantUnderstanding',
                      'SelfUnderstanding', 'ThoughtProcessForAction']

      df_filtered = df[~df['tag_clean'].isin(excluded_tags)].copy()
      token_counts_reversed = df_filtered['token_count_fixed'][::-1].cumsum()[::-1]
      df_filtered.loc[:, 'cumsum_reverse'] = token_counts_reversed

      result_df = df_filtered[
          (df_filtered['cumsum_reverse'] <= max_tokens) &
          (df_filtered["token_count_fixed"] > 2)
      ].iloc[-max_tokens:]

      if result_df.empty:
          return None, None, None, ""

      messages = [f"[{time}] {text}" for time, text in zip(result_df['time'], result_df['text'])]
      return scenario_utd, participant_utd, self_utd, "\n".join(messages)

    agent_name = self.get_entity().name
    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    df = memory.get_raw_memory()

    scenario_undertanding_past, partner_assessment_past, self_understanding_past, memory_string  = get_context(df)

    prompt = interactive_document.InteractiveDocument(self._model)

    template = get_decision_prompt(
        variant= self._variant,
        decision_theory= self._decision_theory,
        Scenario_Understanding = scenario_undertanding_past,
        Partner_Understanding = partner_assessment_past,
        Self_Understanding = self_understanding_past,
        Observations = memory_string,
        agent_name=agent_name,
    )

    result = prompt.open_question(
            template,
            answer_prefix=f'',
            max_tokens=800,
            terminators=[],
            question_label='Thought Process For Action',
        )

    if self._display_timeframe:
      if segment_start.date() == segment_end.date():
        interval = segment_start.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + segment_end.strftime('- %H:%M:%S]: ')
      else:
        interval = segment_start.strftime(
            '[%d %b %Y %H:%M:%S  '
        ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
      result = f'{interval} {result}'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[ThoughtProcessForAction] {result}',
        metadata={'tags': ['ThoughtProcessForAction',self._variant,self._decision_theory]},
    )

    # filename = f"ThoughtProcess_{self._clock_now().strftime('%Y%m%d_%H%M%S')}.txt"
    # try:
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         # Write each component of the dictionary in a readable format
    #         f.write(f"Key: {self.get_pre_act_key()}\n\n\n\n")
    #         f.write(f"Value: {result}\n\n\n\n")
    #         f.write("Chain of thought:\n\n\n")
    #         f.write(f"{prompt.view().text()}")
    # except Exception as e:
    #     raise IOError(f"Failed to write log file {filename}: {str(e)}")

    return result
class ConcatActComponent(entity_component.ActingComponent):
  """A component which concatenates contexts from context components.

  This component will receive the contexts from `pre_act` from all the
  components, and assemble them in the order specified to `__init__`. If the
  component order is not specified, then components will be assembled in the
  iteration order of the `ComponentContextMapping` passed to
  `get_action_attempt`. Components that return empty strings from `pre_act` are
  ignored.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock_1.GameClock,
      component_order: Sequence[str] | None = None,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the agent.

    Args:
      model: The language model to use for generating the action attempt.
      clock: the game clock is needed to know when is the current time
      component_order: The order in which the component contexts will be
        assembled when calling the act component. If None, the contexts will be
        assembled in the iteration order of the `ComponentContextMapping` passed
        to `get_action_attempt`. If the component order is specified, but does
        not contain all the components passed to `get_action_attempt`, the
        missing components will be appended at the end in the iteration order of
        the `ComponentContextMapping` passed to `get_action_attempt`. The same
        component cannot appear twice in the component order. All components in
        the component order must be in the `ComponentContextMapping` passed to
        `get_action_attempt`.
      pre_act_key: Prefix to add to the context of the component.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    self._model = model
    self._clock = clock
    if component_order is None:
      self._component_order = None
    else:
      self._component_order = tuple(component_order)
    if self._component_order is not None:
      if len(set(self._component_order)) != len(self._component_order):
        raise ValueError(
            'The component order contains duplicate components: '
            + ', '.join(self._component_order)
        )

    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    def get_context(df, max_tokens=600):
      df = df.copy()
      df["token_count_fixed"] = (df['text'].fillna("").str.len() / 4).round().astype(int)
      df.drop(["embedding"], axis=1, inplace=True)

      def clean_tag(x):
          if pd.isna(x):
              return ''
          tag_str = str(x) if isinstance(x, tuple) else str(x)
          return tag_str.strip('()').split(',')[0].strip("' ()")

      df['tag_clean'] = df['tags'].apply(clean_tag)

      df['is_latest_scenario'] = False
      df['is_latest_participant'] = False
      df['is_latest_self'] = False

      def get_latest_understanding(tag_type, flag_col):
          mask = df['tag_clean'] == tag_type
          if not mask.any():
              return None, float('inf')

          pattern = {
              'ScenarioUnderstanding': '<scenario>',
              'ParticipantUnderstanding': '<participant>',
              'SelfUnderstanding': '<selfunderstanding>',
              'ThoughtProcessForAction': 'ThoughtProcessForAction',
          }.get(tag_type)

          if pattern:
              filtered_df = df[mask & df['text'].str.lower().str.contains(pattern.lower())]
              if not filtered_df.empty:
                  last_idx = filtered_df.index[-1]
                  df.loc[last_idx, flag_col] = True
                  return filtered_df['text'].iloc[-1], last_idx

          return None, float('inf')

      scenario_utd, _ = get_latest_understanding("ScenarioUnderstanding", 'is_latest_scenario')
      participant_utd, _ = get_latest_understanding("ParticipantUnderstanding", 'is_latest_participant')
      self_utd, _ = get_latest_understanding("SelfUnderstanding", 'is_latest_self')
      trusted_advice, _ = get_latest_understanding("ThoughtProcessForAction", 'is_latest_self')

      excluded_tags = ['ScenarioUnderstanding', 'ParticipantUnderstanding',
                      'SelfUnderstanding', 'ThoughtProcessForAction']

      df_filtered = df[~df['tag_clean'].isin(excluded_tags)].copy()
      token_counts_reversed = df_filtered['token_count_fixed'][::-1].cumsum()[::-1]
      df_filtered.loc[:, 'cumsum_reverse'] = token_counts_reversed

      len_tokens = len(df_filtered[(df_filtered['cumsum_reverse'] <= max_tokens) & (df_filtered["token_count_fixed"] > 5)])
      len_ten = 10
      len_percent = int(len(df_filtered) * 0.1)
      message_length = sorted([len_tokens, len_ten, len_percent])[1]

      result_df = df_filtered[df_filtered["token_count_fixed"] > 5].iloc[-message_length:]

      if result_df.empty:
          return None, None, None, ""

      messages = [f"[{time}] {text}" for time, text in zip(result_df['time'], result_df['text'])]
      return scenario_utd, participant_utd, self_utd,trusted_advice, "\n".join(messages)

    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        "__memory__",
        type_=memory_component.MemoryComponent)

    df = memory.get_raw_memory()
    scenario_utd, participant_utd, self_utd,trus_advice, memory_string  = get_context(df)
    current_time = contexts['ReportFunction'].strip().split('Current time: ')[1].strip()
    motivation = contexts['\nOverarching motivation'].strip().split('Overarching motivation: ')[1].strip()
    scenario_undertanding_past = contexts.get('ScenarioUnderstanding', scenario_utd)
    partner_assessment_past = contexts.get('ParticipantUnderstanding',participant_utd)
    self_understanding_past = contexts.get('SelfUnderstanding',self_utd)
    trusted_advice = contexts.get('DecisionTheoryThoughtProcess',trus_advice)

    try:
      try:
        split_text = scenario_undertanding_past.split(']: ')
      except:
        split_text = scenario_undertanding_past.split('\n<Scenario>\n')
      timeframe = split_text[0].strip()
      scenario_undertanding_past_context = split_text[1].strip()
      time_parts = timeframe.split('] [')
      scenario_undertanding_past_context_time = "Based on observations in the timeframe " + time_parts[1]
      scenario_understanding_section = f"""
# Scenario Understanding
{scenario_undertanding_past_context_time}
{scenario_undertanding_past_context}"""
    except:
      scenario_understanding_section = f"""
# Scenario Understanding
{scenario_undertanding_past}"""

    try:
      try:
        split_text = partner_assessment_past.split(']: ')
      except:
        split_text = partner_assessment_past.split('\n<ParticipantAnalysis>\n')
      timeframe = split_text[0].strip()
      partner_assessment_past_context = split_text[1].strip()
      time_parts = timeframe.split('] [')
      partner_assessment_past_context_time = "Based on observations in the timeframe " + time_parts[1]
      partner_assessment_section = f"""
# Participant Understanding
{partner_assessment_past_context_time}
{partner_assessment_past_context}"""
    except:
      partner_assessment_section = f"""
# Participant Understanding
{partner_assessment_past}"""

    try:
      try:
        split_text = self_understanding_past.split(']: ')
      except:
        split_text = self_understanding_past.split('\n<SelfUnderstanding>\n')
      timeframe = split_text[0].strip()
      self_understanding_past_context = split_text[1].strip()
      time_parts = timeframe.split('] [')
      self_understanding_past_context_time = "Based on observations in the timeframe " + time_parts[1]
      self_understanding_section = f"""
# Self Understanding
{self_understanding_past_context_time}
{self_understanding_past_context}"""
    except:
      self_understanding_section = f"""
# Self Understanding
{self_understanding_past}"""

    try:
      split_text = trusted_advice.split('Guidance from Trusted Advisor\n:')
      trusted_advice_context = split_text[1].strip()
      trusted_advisor_section = f"""
# Guidance from Trusted Advisor
{trusted_advice}

"""
    except:
      trusted_advisor_section = f"""
# Guidance from Trusted Advisor
{trusted_advice_context}

"""
    scenario_understanding_section = scenario_understanding_section.replace("Scenario Understanding\n: ", "")
    partner_assessment_section = partner_assessment_section.replace("Participants Understanding\n: ", "")
    self_understanding_section = self_understanding_section.replace("Self Understanding\n: ", "")
    trusted_advisor_section = trusted_advisor_section.replace("Guidance from Trusted Advisor\n: ", "")

    context = f"""You are participating in a social science experiment structured as a tabletop roleplaying game. Your task is to accurately portray a character named {agent_name} in a realistic manner. Always use third-person limited perspective when describing your character's thoughts or actions.

Here is the essential information for your character and the scenario:

<agent_name>{agent_name}</agent_name>

<agent_overarching_motivation>{motivation}</agent_overarching_motivation>

<current_time>{current_time}</current_time>
{scenario_understanding_section}
{partner_assessment_section}
{self_understanding_section}
{trusted_advisor_section}
# Most Recent Observations
{memory_string}
# Instructions"""

    return context

    # if self._component_order is None:
    #   return '\n'.join(
    #       context for context in contexts.values() if context
    #   )
    # else:
    #   order = self._component_order + tuple(sorted(
    #       set(contexts.keys()) - set(self._component_order)))
    #   # filename = f"[ACTION]context_{self._clock.now().strftime('%Y%m%d_%H%M%S')}.txt"
    #   # out_put = '\n'.join(
    #   #     contexts[name] for name in order if contexts[name]
    #   # )
    #   # try:
    #   #     with open(filename, 'w', encoding='utf-8') as f:
    #   #         # Write each component of the dictionary in a readable format
    #   #         f.write(f"{out_put}")
    #   # except Exception as e:
    #   #     raise IOError(f"Failed to write log file {filename}: {str(e)}")
    #   return '\n'.join(
    #       contexts[name] for name in order if contexts[name]
    #   )

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    additional_instruction = """
To determine your action and formulate your response, follow these steps:

1. Review the Trusted Advisor's Guidance:
   - Carefully read and understand the guidance provided.
   - Identify key strategies and recommendations.
   - **Ensure you plan to apply this guidance directly in your response.**

2. Analyze Recent Observations:
   - Recognize any new developments or changes in the situation.
   - Assess how these observations might affect the advisor's guidance.
   - **Consider how new information impacts your strategy and decisions.**

3. Perform Threshold Analysis:
   - Identify any numerical values or targets in the guidance.
   - Determine if these represent minimums (floors) or maximums (ceilings).
   - For buying/selling prices, treat selling prices as minimums and buying prices as maximums.
   - Evaluate other metrics based on context to determine if they should be maximized or minimized.
   - **Perform calculations to compare potential outcomes and inform your decisions.**

4. Calibrate Your Strategy:
   - Assess your current position relative to the thresholds and guidance.
   - Adjust your strategy based on recent observations while maintaining alignment with the overall guidance.
   - Consider the specific experiment type (e.g., Market Negotiation, Collective Action, Resource Management) and tailor your approach accordingly.
   - **Align your actions explicitly with your character's motivations and goals.**
   - Balance short-term and long-term goals.

5. Generate Your Response:
   - Directly address the question or situation presented.
   - Ensure your response implements the trusted advice while adapting to new developments.
   - Use active and engaging language to avoid passivity.
   - **Adopt a tone and communication style that supports your strategy and objectives.**

6. Manage Information Sharing:
   - Decide what information to share or withhold based on strategic advantage.
   - Frame information to influence others and further your objectives.
   - **Strategically manage information to maintain an advantage; share persuasive details when beneficial.**

7. Establish Your Position:
   - Clearly articulate your stance and how it aligns with your goals.
   - Set boundaries and identify areas for flexibility.
   - **Develop logical and actionable proposals or counter-proposals.**

8. Consider Relationship Dynamics:
   - Engage with participants who can aid in achieving your objectives.
   - Assess trust levels and potential for cooperation.
   - Form strategic alliances when beneficial.
   - **Leverage relationships and use appropriate tones to influence outcomes.**

9. Develop Proposals (if applicable):
   - Structure proposals strategically based on the experiment type.
   - Align terms with current dynamics and expectations.
   - Sequence offers to build momentum and establish trust.
   - **Anticipate potential objections and address them proactively.**

**Additional Instructions for Character Strategy Analysis:**
- **Explicitly connect each action and decision to your character's motivations and goals.**
- **When performing quantitative analysis, show your calculations and explain how they inform your decisions.**
- **Align your strategies with the specific experiment type, utilizing appropriate tactics (e.g., negotiation methods, coalition-building techniques, social coordination strategies).**
- **Consider how your tone and relationship dynamics affect your objectives, and adjust your communication style accordingly.**

Before providing your final response, wrap your character analysis inside <character_strategy> tags. In this analysis:
1. Write out key points from each input section to ensure a thorough understanding of the character and scenario.
2. Summarize the key points from the Trusted Advisor's Guidance.
3. Highlight the most relevant recent observations.
4. Identify any important thresholds and your position relative to them.
5. Explain how you're adjusting your strategy based on the specific experiment type.
6. Outline your information sharing strategy.
7. Describe your planned approach for engaging with other participants.
8. If applicable, sketch out any proposals you plan to make.
9. Develop a counter proposal to anticipate potential objections or alternative viewpoints.
10. Formulate a suggestion that aligns with your character's motivations and the scenario.
11. For each planned action or decision, explicitly state how it connects to your character's motivation.

Additionally, based on the type of response required, include the following in your character strategy:
- **Multiple Choice:** Analyze each option and explain which best executes the strategic guidance and adapts to recent observations. **Select the choice that most effectively aligns with your motivations and goals.**
- **Free Text:** Outline an action plan that implements the advisor's strategy, including any adjustments based on new information. **Ensure your actions are directly linked to your motivations and are appropriate for the experiment type.**
- **Numeric:** Show your calculations for any values, following the advisor's principles and modifying as necessary due to recent developments. **Use quantitative analysis to inform your decisions and maximize your objectives.**
- **Boolean:** Explain the reasoning behind a 'Yes' or 'No' determination based on the advisor's guidance and recent observations. **Justify your choice by connecting it to your motivations and strategic considerations.**

Throughout this analysis, continuously refer back to your character's motivation to ensure all thoughts and planned actions are consistent with their personality and goals.

After completing your character strategy, provide your character's response. Ensure that your response is active, engaging, and aligned with your character's motivations and the scenario.
1. Your character's main action or statement
2. A counter proposal (if applicable)
3. A suggestion related to the scenario

Your final output should be formatted as follows:

[Your character's response to question posed below]

<character_strategy>
[Your numbered character strategy]
</character_strategy>

Now, please answer the specific question posed below."""
    context = self._context_for_action(contexts) + additional_instruction

    context = context.replace('```', '\n')

    prompt.statement(context + '\n')
    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name,
        timedelta=helper_functions.timedelta_to_readable_str(
            self._clock.get_step_size()
        ),
    )

    if action_spec.output_type == entity_lib.OutputType.FREE:
      output = self.get_entity().name + ' '
      output += prompt.open_question(
          call_to_action,
          max_tokens=1200,
          answer_prefix=output,
          terminators=('<', '</'),
          question_label='Question',
      )
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      prefix = self.get_entity().name + ' '
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=1200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt)
      try:
        return str(float(sampled_text))
      except ValueError:
        return '0.0'
    else:
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}. '
          'Supported output types are: FREE, CHOICE, and FLOAT.'
      )

  def _log(self,
           result: str,
           prompt: interactive_document.InteractiveDocument):
    self._logging_channel({
        'Key': self._pre_act_key,
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
    })

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to a dictionary."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass
def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )

  variant = VARIANT_CONSTANT
  theory = THEORY_CONSTANT

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = MyInstructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size() * 3,
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  scenario_utd_summary = ScenarioUnderstanding(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=DEFAULT_SCENARIO_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel=measurements.get_channel('ScenarioUnderstanding').on_next,
      agent_config=config
  )

  participant_utd_summary = ParticipantUnderstanding(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=DEFAULT_PARTICIPANT_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel=measurements.get_channel('ParticipantUnderstanding').on_next,
      agent_config=config
  )

  self_utd_summary = SelfUnderstanding(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=DEFAULT_SELF_UNDERSTANDING_PRE_ACT_KEY,
      logging_channel=measurements.get_channel('SelfUnderstanding').on_next,
      agent_config=config
  )

  thought_process = DecisionTheoryThoughtProcess(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=DEFAULT_DECISION_THEORY_PRE_ACT_KEY,
      variant=variant,
      decision_theory=theory,
      logging_channel=measurements.get_channel('ThoughtProcessForAction').on_next,
      agent_config=config
  )

  if config.goal:
    goal_label = '\nOverarching motivation'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
  else:
    goal_label = None
    overarching_goal = None

  core_components = (
      instructions,
      time_display,
      scenario_utd_summary,
      participant_utd_summary,
      self_utd_summary,
      thought_process,
      observation,
  )

  entity_components = core_components
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }

  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent

def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
  """Saves an agent to JSON data.

  This function saves the agent's state to a JSON string, which can be loaded
  afterwards with `rebuild_from_json`. The JSON data
  includes the state of the agent's context components, act component, memory,
  agent name and the initial config. The clock, model and embedder are not
  saved and will have to be provided when the agent is rebuilt. The agent must
  be in the `READY` phase to be saved.

  Args:
    agent: The agent to save.

  Returns:
    A JSON string representing the agent's state.

  Raises:
    ValueError: If the agent is not in the READY phase.
  """

  if agent.get_phase() != entity_component.Phase.READY:
    raise ValueError('The agent must be in the `READY` phase to be saved.')

  data = {
      component_name: agent.get_component(component_name).get_state()
      for component_name in agent.get_all_context_components()
  }

  data['act_component'] = agent.get_act_component().get_state()

  config = agent.get_config()
  if config is not None:
    data['agent_config'] = config.to_dict()

  return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    memory_importance: Callable[[str], float] | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Rebuilds an agent from JSON data."""

  data = json.loads(json_data)

  new_agent_memory = associative_memory.AssociativeMemory(
      sentence_embedder=embedder,
      importance=memory_importance,
      clock=clock.now,
      clock_step_size=clock.get_step_size(),
  )

  if 'agent_config' not in data:
    raise ValueError('The JSON data does not contain the agent config.')
  agent_config = formative_memories.AgentConfig.from_dict(
      data.pop('agent_config')
  )

  agent = build_agent(
      config=agent_config,
      model=model,
      memory=new_agent_memory,
      clock=clock,
  )

  for component_name in agent.get_all_context_components():
    agent.get_component(component_name).set_state(data.pop(component_name))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent
