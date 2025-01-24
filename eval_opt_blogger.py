from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
import asyncio
import nest_asyncio


nest_asyncio.apply()


# Instantiate the model
model = OllamaModel(
    model_name='qwen2.5:7b',
    base_url='http://localhost:11434/v1/',
)

# Dependencies for the evaluator
class BlogEvalDependencies(BaseModel):
    target_audience: str = Field(description="Target audience of the blog post")
    target_readability: float = Field(description="Target readability score (0-100)")

# Result model for evaluator
class BlogEvalResult(BaseModel):
    coherence_score: float = Field(ge=0, le=1, description="Coherence of the blog")
    originality_score: float = Field(ge=0, le=1, description="Originality of the content")
    readability_score: float = Field(ge=0, le=100, description="Readability score (Flesch–Kincaid)")
    improvement_suggestions: str = Field(description="Suggestions to improve the blog")

# Agent to write and evaluate blogs
blog_agent = Agent(
    model=model,
    deps_type=BlogEvalDependencies,
    result_type=BlogEvalResult,
    system_prompt=(
        "You are a blog writing assistant for AI engineering and data science."
        " Write content that is coherent, original, and readable for the specified audience."
    ),
)

# Tool to evaluate coherence
@blog_agent.tool
async def evaluate_coherence(ctx: RunContext[BlogEvalDependencies], text: str) -> float:
    """Evaluate the coherence of the provided text."""
    # Mocked coherence calculation logic
    coherence = len(set(text.split())) / len(text.split())
    return round(coherence, 2)

# Tool to evaluate readability
@blog_agent.tool
async def evaluate_readability(ctx: RunContext[BlogEvalDependencies], text: str) -> float:
    """Evaluate the readability of the provided text (Flesch-Kincaid)."""
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    syllables = sum(sum(1 for char in word if char in "aeiouAEIOU") for word in words)
    readability = 206.835 - 1.015 * (len(words) / max(1, sentences)) - 84.6 * (syllables / len(words))
    return round(max(0, min(100, readability)), 2)

# Tool to evaluate originality
@blog_agent.tool
async def evaluate_originality(ctx: RunContext[BlogEvalDependencies], text: str) -> float:
    """Evaluate the originality of the provided text."""
    # Mock originality logic (real implementation may involve similarity checks)
    originality = 1.0  # Assume fully original for mock
    return originality

# Generate improvement suggestions
@blog_agent.system_prompt
async def generate_improvement_suggestions(ctx: RunContext[BlogEvalDependencies], text: str) -> str:
    return (
        f"Please suggest improvements for a blog targeting '{ctx.deps.target_audience}' "
        f"to achieve a readability score of at least {ctx.deps.target_readability}."
    )

# Main evaluation logic
async def main():
    deps = BlogEvalDependencies(target_audience="AI practitioners", target_readability=60)
    blog_text = "Artificial Intelligence is a revolutionary field changing industries worldwide."
    
    # Run evaluation
    result = await blog_agent.run(blog_text, deps=deps)
    
    print("Evaluation Results:")
    print(f"Coherence Score: {result.data.coherence_score}")
    print(f"Originality Score: {result.data.originality_score}")
    print(f"Readability Score: {result.data.readability_score}")
    print(f"Improvement Suggestions: {result.data.improvement_suggestions}")

# Trigger the main evaluation process
asyncio.run(main())
