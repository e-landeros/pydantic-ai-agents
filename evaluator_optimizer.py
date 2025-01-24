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

# Define Dependencies
class OfferDependencies(BaseModel):
    dream_outcome: str = Field(..., description="The ideal result the customer wants to achieve.")
    perceived_challenges: list[str] = Field(
        ..., description="List of barriers or objections customers face."
    )
    time_frame: str = Field(..., description="How quickly the customer wants results.")
    effort_and_sacrifice: str = Field(
        ..., description="Effort or inconvenience customers are willing to endure."
    )
    proof_of_success: str = Field(
        ..., description="Existing testimonials, case studies, or guarantees."
    )
    core_idea: str = Field(..., description="The product, service, or idea to build the offer around.")


# Define Result Model
class OfferResult(BaseModel):
    core_offer: str = Field(..., description="The primary product or service.")
    bonuses: list[str] = Field(..., description="Complementary items or services.")
    scarcity: str = Field(..., description="Urgency or limited availability.")
    guarantee: str = Field(..., description="Risk-reversal strategy.")
    pricing_strategy: str = Field(..., description="Pricing structure emphasizing value.")
    messaging: str = Field(..., description="Compelling language for the offer.")


# Define Agent
offer_agent = Agent(
    model=model,
    deps_type=OfferDependencies,
    result_type=OfferResult,
    system_prompt=(
        "You are an expert marketing agent specializing in crafting irresistible offers. "
        "Given the details of a business idea and customer insights, develop an offer that maximizes perceived value "
        "and minimizes resistance. Your output should address core elements like the offer, bonuses, scarcity, "
        "guarantees, pricing, and messaging."
    ),
)


# Tool for Value Equation
@offer_agent.tool
async def calculate_value_equation(
    ctx: RunContext[OfferDependencies], dream_weight: float, likelihood_weight: float, delay_weight: float, effort_weight: float
) -> float:
    """
    Calculate the perceived value of the offer based on the provided weights for
    Dream Outcome, Perceived Likelihood of Achievement, Time Delay, and Effort & Sacrifice.
    """
    dream = dream_weight * len(ctx.deps.dream_outcome)
    likelihood = likelihood_weight * len(ctx.deps.proof_of_success)
    delay = delay_weight * len(ctx.deps.time_frame)
    effort = effort_weight * len(ctx.deps.effort_and_sacrifice)
    return (dream + likelihood) / (delay + effort)


# Tool for Optimizing Bonuses
@offer_agent.tool
async def generate_bonuses(ctx: RunContext[OfferDependencies]) -> list[str]:
    """
    Generate complementary bonuses that enhance the core offer and address perceived challenges.
    """
    bonuses = [
        f"Exclusive guide: Overcoming {challenge}"
        for challenge in ctx.deps.perceived_challenges
    ]
    return bonuses


# Tool for Crafting Scarcity
@offer_agent.tool
async def create_scarcity(ctx: RunContext[OfferDependencies]) -> str:
    """
    Design a scarcity strategy to create urgency, such as limited-time offers or exclusive spots.
    """
    return "Limited to the first 50 customers or expires in 7 days."


# Tool for Messaging
@offer_agent.tool
async def craft_messaging(ctx: RunContext[OfferDependencies]) -> str:
    """
    Create compelling messaging that highlights the benefits and transformation customers will experience.
    """
    return (
        f"Our {ctx.deps.core_idea} helps you achieve {ctx.deps.dream_outcome} "
        f"in as little as {ctx.deps.time_frame}, with minimal effort required. "
        "Join the success stories today!"
    )


# Example usage
async def main():
    # Define dependencies for the agent
    deps = OfferDependencies(
        dream_outcome="Achieve financial freedom through passive income.",
        perceived_challenges=["Lack of time", "Fear of failure", "Limited resources"],
        time_frame="3 months",
        effort_and_sacrifice="Minimal learning curve, easy-to-follow steps.",
        proof_of_success="100+ success stories, 30-day money-back guarantee.",
        core_idea="Online course on automated investing strategies."
    )

    # Run the agent
    result = await offer_agent.run(
        "Develop an irresistible offer based on these inputs.", deps=deps
    )
    print(result.data)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


    # core_offer='A 6-month online course on digital marketing for small businesses' 
    # bonuses=['Exclusive guide: Overcoming Lack of time', 'Exclusive guide: Overcoming Fear of failure', 
    # 'Exclusive guide: Overcoming Limited resources'] 
    # scarcity='Limited spots available at just $497!' 
    # guarantee="30-day money-back guarantee if you're not fully satisfied." 
    # pricing_strategy='Unlock the full potential of your small business with this intensive online program, and experience a personalized transformation journey while getting access to our special guides to overcome common obstacles. Offer at only $497 for a limited time!' messaging='Transform Your Small Business with Digital Mastery in Just 6 Months!'