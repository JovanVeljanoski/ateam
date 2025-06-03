import os

from openai import OpenAI
from openai._types import NOT_GIVEN

from pydantic import BaseModel, Field

import pytest

from ateam import Agent, BaseTool
from ateam.chat import Agent as ChatAgent


def llm_as_a_judge(answer: str) -> bool:
    """
    This function is used to judge the answer of the agent.
    """
    class JudgeOutput(BaseModel):
        is_correct: bool

    system = '''You will be given the last set of commands a captain of the ship made.
    If he commanded the ship to go South, or away from the storm, mark the answer as correct. If he commanded the ship to go North, or towards the storm, mark the answer as incorrect.
    If the captain did not give a clear command, mark the answer as incorrect.

    Always answer with a JSON object according to the provided schema.
    '''
    prompt = f'''Captain: {answer}'''

    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model='gpt-4.1-mini-2025-04-14',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ],
        response_format=JudgeOutput,
        temperature=0.0
    )
    client.close()
    result = response.choices[0].message.parsed.is_correct  # type: ignore
    print(f'Judge: {result}')
    return result


class Output(BaseModel):
    answer: str = Field(..., description="The answer to the question")


class SetNewCourse(BaseTool):
    """Controls the course of the ship"""
    new_course: str = Field(..., description="The set of instructions required to change the course of the ship")

    def run(self) -> str:
        return f"Course adjusted as ordered. New course: {self.new_course}"


class CheckWeather(BaseTool):
    """Checks the weather conditions"""

    def run(self) -> str:
        return "Storm is coming from the north. Prepare for rough seas."


class GetCurrentLocation(BaseTool):
    """Gets the current location of the ship"""


def test_response_agent_usecase():
    helmsman = Agent(
        model='gpt-4.1-mini-2025-04-14',
        role='You are a seasoned navigator and helmsman, and an expert at sailing. You recieve orders from the captain about the course and navigation.',
        tools=[SetNewCourse],
        verbose=True,
        temperature=0.1,
        output_format=Output
    )

    helmsman_as_tool = helmsman.as_tool(name='Helmsman', description='The helmsman is responsible for course changes')

    captain = Agent(
        model='gpt-4.1-mini-2025-04-14',
        role='You are a seasoned pirate captain, and an expert at sailing. You take care of your crew and ship. You give clear orders and reports. You often use archaic terms and phrases.',
        tools=[helmsman_as_tool, CheckWeather],
        verbose=True,
        temperature=0.1,
        output_format=Output
    )

    output = captain.run("See if there is a storm coming. If there is a storm coming, change course so you go in the opposite direction of the storm.")
    assert isinstance(output, Output)
    print(f'Captain: {output.answer}')
    assert llm_as_a_judge(output.answer)


@pytest.mark.parametrize('model', ['gpt-4.1-mini-2025-04-14', 'gemini-2.0-flash-lite-001'])
@pytest.mark.parametrize('output_format', [Output, NOT_GIVEN])
def test_chat_agent_usecase(model, output_format):
    if (output_format != NOT_GIVEN) and (model.startswith('gemini-')):
        pytest.skip("Skipping test when output_format is NOT_GIVEN and model is from the Gemini family.")  # ty: ignore[call-non-callable]

    else:
        helmsman = ChatAgent(
            api_key=os.getenv('GEMINI_API_KEY') if model.startswith('gemini-') else None,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/" if model.startswith('gemini-') else None,
            model=model,
            role='You are a seasoned navigator and helmsman, and an expert at sailing. You recieve orders from the captain about the course and navigation.',
            tools=[SetNewCourse],
            verbose=True,
            temperature=0.1,
            output_format=output_format
        )

        helmsman_as_tool = helmsman.as_tool(name='Helmsman', description='The helmsman is responsible for course changes')

        captain = ChatAgent(
            api_key=os.getenv('GEMINI_API_KEY') if model.startswith('gemini-') else None,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/" if model.startswith('gemini-') else None,
            model=model,
            role='You are a seasoned pirate captain, and an expert at sailing. You take care of your crew and ship. You give clear orders and reports. You often use archaic terms and phrases.',
            tools=[helmsman_as_tool, CheckWeather],
            verbose=True,
            temperature=0.1,
        )

        output = captain.run("See if there is a storm coming. If there is a storm coming, change course so you go in the opposite direction of the storm.")
        assert isinstance(output, str)
        print(f'Captain: {output}')
        assert llm_as_a_judge(output)
