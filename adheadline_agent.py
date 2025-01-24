from pydantic_ai.models.ollama import OllamaModel
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class MarketingDeps:
    product_description: str
    benefits: list[str] = None
    hypotheses: list[str] = None



model = OllamaModel(
    model_name='qwen2.5:7b',  
    base_url='http://localhost:11434/v1/',  
)

#### Agent Definition

marketing_agent = Agent(
    model=model,
    deps_type=MarketingDeps,
    system_prompt="Your task is to create a marketing ad headline in 4 steps.",
    result_type=dict,  # Final output will include all stages
)

#### Stages and Tools
#Define stages using PydanticAI's system prompt customization.

##1. **Stage 1: Understand the Product/Service**

@marketing_agent.system_prompt
async def describe_product(ctx: RunContext[MarketingDeps]) -> str:
    return (
        "You are a marketing expert. Start by understanding the product or service "
        f"based on this input: {ctx.deps.product_description}. "
        "Provide a structured and concise description."
    )


#2. **Stage 2: List Benefits**

@marketing_agent.tool
async def list_benefits(ctx: RunContext[MarketingDeps]) -> list[str]:
    return (
        "Based on the product description, list the top 5 benefits to the end user. "
        "Keep the benefits concise and focused."
    )

#3. **Stage 3: Generate Hypotheses**

@marketing_agent.tool
async def generate_hypotheses(ctx: RunContext[MarketingDeps]) -> list[str]:
    return (
        "Use the product benefits to create 3 imaginative and engaging hypotheses "
        "to draw customer interest. Ensure they appeal to the target audience."
    )


#4. **Stage 4: Refine Outputs**

@marketing_agent.tool
async def refine_outputs(ctx: RunContext[MarketingDeps]) -> list[str]:
    return (
        "Given the product description, benefits, and hypotheses, refine them into "
        "3 polished marketing headlines with short supporting copy for each. "
        "Make sure they are compelling and professional."
    )


#### Running the Agent
#Use an async workflow to process inputs through each stage:

description = "An wearable glucose monitor."
deps = MarketingDeps(product_description=description)

    # Stage 1: Describe the product
await marketing_agent.run("Describe the product.", deps=deps)

    # Stage 2: List benefits
deps.benefits = await marketing_agent.run("List benefits.", deps=deps)

    # Stage 3: Generate hypotheses
deps.hypotheses = await marketing_agent.run("Generate hypotheses.", deps=deps)

    # Stage 4: Refine outputs
final_outputs = await marketing_agent.run("Refine outputs.", deps=deps)

print(final_outputs)

