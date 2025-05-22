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

## General usage when creating agents and tools

For better results, creating the comprehensive instructions (roles, descriptions, names, etc.) is key. Make sure you provide sufficient and clear details, to get maximum from your agentic workflow.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)