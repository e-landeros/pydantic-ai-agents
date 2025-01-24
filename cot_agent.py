from pydantic_ai.models.ollama import OllamaModel
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from typing import List
import nest_asyncio


nest_asyncio.apply()


model = OllamaModel(
    model_name='qwen2.5:7b',  
    base_url='http://localhost:11434/v1/',  
)


agent = Agent(
    model=model,
    system_prompt='You are an expert in reasoning. Break down complex problems step-by-step before arriving at a conclusion.',
)

@dataclass
class CoTDependencies:
    facts: List[str]  # Facts to use for reasoning
    user_question: str  # Question posed by the user


@agent.tool
async def retrieve_relevant_facts(ctx: RunContext[CoTDependencies]) -> List[str]:
    """Extracts facts relevant to the user's question."""
    # Filter facts based on the question (dummy example)
    return [fact for fact in ctx.deps.facts if 'relevant' in fact]

@agent.tool
async def analyze_facts(ctx: RunContext[CoTDependencies], facts: List[str]) -> str:
    """Analyzes the given facts to draw intermediate conclusions."""
    return f"Based on the facts {facts}, we conclude step 1."

@agent.tool
async def final_conclusion(ctx: RunContext[CoTDependencies], analysis: str) -> str:
    """Combines analysis into a final conclusion."""
    return f"Using the analysis: {analysis}, the final answer is clear."


@agent.system_prompt
async def include_context(ctx: RunContext[CoTDependencies]) -> str:
    return f"The user has asked: {ctx.deps.user_question}. Think step-by-step using these facts: {ctx.deps.facts}."

# deps = CoTDependencies(
#         facts=["Fact 1: relevant data", "Fact 2: irrelevant data", "Fact 3: relevant insight"],
#         user_question="What can we deduce from relevant data?"
#     )

#     # Execute the agent with a complex query
# result = await agent.run("Explain why the sky appears blue during the day but red at sunset.", deps=deps)
# print(result.data)

async def main():
    # Define dependencies
    deps = CoTDependencies(
        facts=["Fact 1: relevant data", "Fact 2: irrelevant data", "Fact 3: relevant insight"],
        user_question="What can we deduce from relevant data?"
    )

    # Execute the agent with a complex query
    result = await agent.run("Explain why the sky appears blue during the day but red at sunset.", deps=deps)
    print(result.data)

# Use an event loop to execute the main function
import asyncio
asyncio.run(main())

