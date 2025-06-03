# ateam

This is a simple library for building AI agents.

It is built on top of [openai](https://github.com/openai/openai-python) (the `responses` api) and [pydantic](https://github.com/pydantic/pydantic).

This library was heavily inspired by [agency-swarm](https://github.com/VRSEN/agency-swarm) - and I do recommend you to check it out for some more advanced usecases.

### Motivation

The motivation behind this library is to leverate the `responses` api of the openai python library, at least until `agency-swarm` catches up. Also I wanted to have a simple, lightweight library for building AI agents, where one can easily understand how the underlying LLMs are called (e.g. which instructions go to the "system_prompt")

## Installation

For now, you can install the library directly from GitHub.
The use of `uv` is recommended.

```zsh
uv add git+https://github.com/JovanVeljanoski/ateam.git
```

You can also clone the repository and install it in development mode.

```zsh
git clone git@github.com:JovanVeljanoski/ateam.git
cd ateam
uv add -e .
```

The pip interface can also be used:
```bash
uv pip install git+https://github.com/JovanVeljanoski/ateam.git
```

## Usage


### Simple example
The following is a simple example of the key features of the library.
Note that the tools are for illustrations only.

```python
from ateam import Agent, BaseTool

# Structure output example
class Output(pydantic.BaseModel):
    answer: str
    internal_thoughts: str

# Define some tools
class CheckWeather(BaseTool):
    """This is used when we want to check the weather, see if the wind is favorable or if a storm is coming"""

    def run(self) -> str:
        return "Steady wind from the north, sunny, no clouds at all."

class SetNewCourse(BaseTool):
    """Order the helmsman to set a new course"""
    new_course: str = pydantic.Field(..., description="The new set of instructions to be passed to the helmsman, on how to change course.")

    def run(self) -> str:
        return f"Aye captain! {self.new_course}!. Course adjusted as ordered."

helm = Agent(
        model='gpt-4.1',
        role="You are a seasoned navigator and helmsman, and an expert at sailing. You recieve orders from the captain about the course and navigation.",
        tools=[SetNewCourse],
        verbose=True,
        temperature=0.5,
        reasoning='high',
        output_format=Output
    )
HelmTool = helm.as_tool(name='Helmsman', description='The helmsman is responsible for course changes')

captain = Agent(
        model='gpt-4.1',
        role="You are a seasoned pirate captain, and an expert at sailing. You take care of your crew and ship. You give clear orders and reports. You often use archaic terms and phrases.",
        tools=[CheckWeather, HelmTool],
        verbose=True,
        temperature=0.5,
        reasoning='high'
    )
print(captain.run("See if there is a storm coming. If there is a storm coming do not change course, otherwise head south."))
```

### Creating tools

To create a tool, you need to create a pydantic model that inherits from `BaseTool`.
The fields of the models are the arguments of the tool. Make sure you write good descriptions so that the agent/LLM can understand the tool usage and pass appropriate values to the arguments. The docstring of the models is used to describe the tool to the agent. The model needs to implement the `run` method, which is the actual implementation of the tool. See the example below:

```python
class CheckWeatherLocation(BaseTool):
    """Check the weather at a specific location"""
    location: str = pydantic.Field(..., description="The location for which we want to check the weather")

    def run(self) -> str:
        # Here you would implement the actual logic of the tool
        return f"The weather at {self.location} is sunny and warm."
```

### Agents as tools
Agents themselves can be converted to tools. This was one agent, when appropriate, can pass instructions to another agent, which in turns has access to other tooks and agents. This allows for more complex workflows.

To convert an agent to a tool, you can use the `as_tool` method. This will return a pydantic model that inherits from `BaseTool` and implements the `run` method. When converting an agent to a tool, you need to specify the name and description of the tool.

Example:
```python
helm = Agent(
    model='gpt-4.1',
    role="You are a seasoned navigator and helmsman, and an expert at sailing. You recieve orders from the captain about the course and navigation.",
    tools=[SetNewCourse],
    verbose=True,
    temperature=0.5,
    reasoning='medium',
    output_format=Output
)
HelmTool = helm.as_tool(name='Helmsman', description='The helmsman is responsible for course changes')
```

### Shared State
The library also supports convenient way to share information (data) across agents and tools. This is useful when you want to share information, and potentially reduce the number of calls you make to the underlying LLM.

<details>
<summary> Shared State Example </summary>

```python
from ateam import Agent, BaseTool
from pydantic import Field

class SetUserPreferenceTool(BaseTool):
    """
    Sets user preferences that can be shared across other tools.
    """
    preference_name: str = Field(..., description="Name of the preference to set (e.g., 'temperature_unit')")
    preference_value: str = Field(..., description="Value of the preference (e.g., 'Fahrenheit' or 'Celsius')")

    def run(self) -> str:
        # Store the preference in shared state
        self._shared_state.set(self.preference_name, self.preference_value)
        print(f"[SharedState SET] '{self.preference_name}': '{self.preference_value}'")
        return f"✅ Preference set: {self.preference_name} = {self.preference_value}"


class GetWeatherTool(BaseTool):
    """
    Gets weather information for a city. Uses temperature_unit from shared state if available.
    """
    city: str = Field(..., description="City name to get weather for")

    def run(self) -> str:
        # Get temperature unit preference from shared state
        preferred_unit = self._shared_state.get("temperature_unit", "Celsius")
        print(f"[SharedState GET] 'temperature_unit': '{preferred_unit}'")

        # Mock weather data (in Celsius)
        mock_weather = {
            "london": 15,
            "tokyo": 22,
            "new york": 18,
            "paris": 12,
            "sydney": 25
        }

        city_lower = self.city.lower()
        if city_lower not in mock_weather:
            return f"❌ Weather data not available for {self.city}"

        temp_celsius = mock_weather[city_lower]

        # Convert temperature based on preference
        if preferred_unit.lower() == "fahrenheit":
            temp_fahrenheit = (temp_celsius * 9/5) + 32
            temperature_str = f"{temp_fahrenheit:.1f}°F"
            print(f"[Temperature Conversion] {temp_celsius}°C → {temp_fahrenheit:.1f}°F")
        else:
            temperature_str = f"{temp_celsius}°C"

        weather_data = {
            "city": self.city,
            "temperature": temperature_str,
            "unit": preferred_unit,
            "condition": "Partly cloudy"  # Mock condition
        }

        return f"🌤️ Weather in {self.city}: {temperature_str}, {weather_data['condition']}"


def demonstrate_agent_with_shared_state():
    """
    Demonstrates SharedState functionality using the real Agent class.
    """

    # Create agent with tools
    agent = Agent(
        role="You are a helpful assistant that can set user preferences and get weather information. "
             "When getting weather, always check if the user has set a temperature unit preference first.",
        tools=[SetUserPreferenceTool, GetWeatherTool],
        verbose=True
    )
    # Test scenarios
    scenarios = [
        "What's the weather in London?",
        "Please set my temperature preference to Fahrenheit",
        "What's the weather in London now?",
        "What's the weather in Tokyo?",
        "Set my temperature preference to Celsius",
        "What's the weather in Paris?",
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*50}")
        print(f"USER QUERY {i}: {scenario}")
        print('='*50)

        response = agent.run(scenario)
        print(f"ASSISTANT: {response}")

    print(f"\n{'='*50}")
    print("FINAL SHARED STATE DATA")
    print('='*50)
    print("Current shared state data:")
    agent.shared_state.print_data()

if __name__ == "__main__":
    demonstrate_agent_with_shared_state()
```

<details>

<summary> Output </summary>

```
==================================================
USER QUERY 1: What's the weather in London?
==================================================
Function call: GetWeatherTool
[SharedState GET] 'temperature_unit': 'Celsius'
Function GetWeatherTool output: 🌤️ Weather in London: 15°C, Partly cloudy
Function call: SetUserPreferenceTool
[SharedState SET] 'temperature_unit': 'Celsius'
Function SetUserPreferenceTool output: ✅ Preference set: temperature_unit = Celsius
ASSISTANT: The weather in London is currently 15°C and partly cloudy.

==================================================
USER QUERY 2: Please set my temperature preference to Fahrenheit
==================================================
Function call: SetUserPreferenceTool
[SharedState SET] 'temperature_unit': 'Fahrenheit'
Function SetUserPreferenceTool output: ✅ Preference set: temperature_unit = Fahrenheit
ASSISTANT: Your temperature preference has been set to Fahrenheit. Would you like to see the current weather in any specific city?

==================================================
USER QUERY 3: What's the weather in London now?
==================================================
Function call: GetWeatherTool
[SharedState GET] 'temperature_unit': 'Fahrenheit'
[Temperature Conversion] 15°C → 59.0°F
Function GetWeatherTool output: 🌤️ Weather in London: 59.0°F, Partly cloudy
ASSISTANT: The weather in London is currently 59°F with partly cloudy skies. Would you like to see the weather in a different city or change the temperature unit?

==================================================
USER QUERY 4: What's the weather in Tokyo?
==================================================
Function call: GetWeatherTool
[SharedState GET] 'temperature_unit': 'Fahrenheit'
[Temperature Conversion] 22°C → 71.6°F
Function GetWeatherTool output: 🌤️ Weather in Tokyo: 71.6°F, Partly cloudy
Function call: SetUserPreferenceTool
[SharedState SET] 'city': 'Tokyo'
Function SetUserPreferenceTool output: ✅ Preference set: city = Tokyo
ASSISTANT: The weather in Tokyo is partly cloudy with a temperature of 71.6°F. Would you like to know anything else?

==================================================
USER QUERY 5: Set my temperature preference to Celsius
==================================================
Function call: SetUserPreferenceTool
[SharedState SET] 'temperature_unit': 'Celsius'
Function SetUserPreferenceTool output: ✅ Preference set: temperature_unit = Celsius
ASSISTANT: Your temperature preference has been set to Celsius. Would you like to know the weather forecast for a specific city?

==================================================
USER QUERY 6: What's the weather in Paris?
==================================================
Function call: GetWeatherTool
[SharedState GET] 'temperature_unit': 'Celsius'
Function GetWeatherTool output: 🌤️ Weather in Paris: 12°C, Partly cloudy
Function call: GetWeatherTool
[SharedState GET] 'temperature_unit': 'Celsius'
Function GetWeatherTool output: 🌤️ Weather in Paris: 12°C, Partly cloudy
ASSISTANT: The weather in Paris is currently 12°C with partly cloudy skies. Would you like to know the weather in another city or need any other assistance?

==================================================
FINAL SHARED STATE DATA
==================================================
Current shared state data:
temperature_unit: Celsius
city: Tokyo
```

</details>

</details>


### Chat API

If you want to use models that are not supported by the `responses` api, typically models provided by providers other than OpenAI, you can use the `chat` API. It is currently only tested with OpenAI and Google Gemini models. The use is generally the same:

```python
import os
from ateam.chat import Agent

gemini_agent = Agent(
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=os.getenv('GEMINI_API_KEY'),
    model='gemini-2.5-flash',
    role='You are a helpful personal assistant managing my calendar and appointments.',
    tools=[CheckCalendar, MakeNewAppointment, ReScheduleAppointment, CancelAppointment]
)

output = gemini_agent.run('What is my schedule for today?')
print(output)
```

#### _Known issues and limitations_

- Gemini models do not simultaneously support both tools and structured output.
- Gemini models do not support reasoning.

## General usage when creating agents and tools

For better results, creating the comprehensive instructions (roles, descriptions, names, etc.) is key. Make sure you provide sufficient and clear details, to get maximum from your agentic workflow.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)