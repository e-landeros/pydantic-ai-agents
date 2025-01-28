from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel, Field
from typing import Optional, List



import nest_asyncio
nest_asyncio.apply()

model = OllamaModel(
    model_name='qwen2.5:7b',
    base_url='http://localhost:11434/v1/',
)




### models ####
class ProductDescription(BaseModel):
    """Input for the product description agent."""
    product_or_service_description: str

class BenefitGenerationOutput(BaseModel):
    """Output for the benefit generation agent."""
    consumer_benefits: List[str]

class AdCreativeInput(BaseModel):
    """Input for the ad creative agent."""
    product_or_service_description: str  # Changed to description
    consumer_benefits: List[str]

class AdCreativeOutput(BaseModel):
    """Output for the ad creative agent."""
    hypothesis: str
    headline: str
    body_copy: str
    score: Optional[int] = None

class EvaluatorInput(BaseModel):
    """Input for the evaluator agent."""
    ad_creative: AdCreativeOutput

class EvaluatorOutput(BaseModel):
    """Output for the evaluator agent."""
    evaluation: str
    score: int

benefit_agent = Agent(
    model=model,
    retries=2,
    result_type=BenefitGenerationOutput,
    deps_type=ProductDescription,
)

@benefit_agent.system_prompt
async def benefit_system_prompt(ctx: RunContext[ProductDescription]) -> str:
    """Generates a system prompt using dependency data."""
    product_or_service_description = ctx.deps.product_or_service_description
    return (
        f"""Generate a list of consumer benefits for the following product/service description: {product_or_service_description}"""
    )

creative_agent = Agent(
    model=model,
    retries=2,
    result_type=AdCreativeOutput,
    deps_type=AdCreativeInput,
)

@creative_agent.system_prompt
async def creative_system_prompt(ctx: RunContext[AdCreativeInput]) -> str:
    """Generates a system prompt using dependency data."""
    product_or_service_description = ctx.deps.product_or_service_description  # Use description
    consumer_benefits = ctx.deps.consumer_benefits
    return (
        f"""
        You are a master offer creator. Your task is to craft an irresistible offer that compels the target audience to take action.
        Generate ad creatives for the following product/service: {product_or_service_description}
        Find innovative ways to present these benefits to the target audience: {consumer_benefits}
        """
    )

evaluator_agent = Agent(
    model=model,
    retries=2,
    result_type=EvaluatorOutput,
    deps_type=EvaluatorInput,
)

@evaluator_agent.system_prompt
async def evaluator_system_prompt(ctx: RunContext[EvaluatorInput]) -> str:
    """Generates a system prompt using dependency data."""
    ad_creative = ctx.deps.ad_creative
    return (
        f"Evaluate the following ad creative: {ad_creative}"
    )

###################################
# Example usage
product_description = "an no code tool to bui8ld better workflows."
###################################

def print_markdown(label, message):
    """Formats and prints the output in Markdown style."""
    print(f"### {label}\n{message}\n")

# Generate consumer benefits
benefits_output = benefit_agent.run_sync(
    user_prompt="Generate consumer benefits. Focus on the top 3 emotional and monetary benefits that will make this persons life better.",
    deps=ProductDescription(product_or_service_description=product_description)
)
consumer_benefits = benefits_output.data.consumer_benefits
print_markdown("Consumer Benefits", consumer_benefits)  # Print intermediate output

product_info = AdCreativeInput(
    product_or_service_description=product_description,  # Use description here
    consumer_benefits=consumer_benefits
)
threshold = 9

current_score = 0
while current_score < threshold:
    ad_creative = creative_agent.run_sync(
        user_prompt="Generate ad creative",
        deps=product_info
    )
    print_markdown("Ad Creative", ad_creative.data)  # Print intermediate output

    evaluation = evaluator_agent.run_sync(
        user_prompt="Evaluate ad creative",
        deps=EvaluatorInput(ad_creative=ad_creative.data)
    )
    current_score = evaluation.data.score
    print_markdown("Evaluation", evaluation.data.evaluation)  # Print evaluation output
    print_markdown("Score", str(current_score))  # Print score output