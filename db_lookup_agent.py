from dataclasses import dataclass
from pydantic import BaseModel, Field,ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry

from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
from datetime import date
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()


model = OllamaModel(
    model_name='qwen2.5:7b',  
    base_url='http://localhost:11434/v1/',  
)


# Simulated database
class OrderDatabase:
    data = {
        1: [{"order_id": 101, "item": "Laptop", "quantity": 1}],
        2: [{"order_id": 102, "item": "Headphones", "quantity": 2}],
    }

    @classmethod
    async def get_orders(cls, client_id: int) -> List[Dict[str, Any]]:
        if client_id not in cls.data:
            raise ModelRetry(f"No orders found for client_id: {client_id}")
        return cls.data[client_id]


# Dependencies for the agent
@dataclass
class Deps:
    client_id: int
    db: OrderDatabase


# Define the result model
class OrderResult(BaseModel):
    client_id: int
    orders: List[Dict[str, Any]] = Field(
        ..., description="List of orders belonging to the client."
    )


# Create the agent
order_lookup_agent = Agent(
    model=model,
    deps_type=Deps,
    result_type=OrderResult,
    retries=2,  # Retry twice in case of transient issues
    system_prompt="You are an assistant retrieving client orders from the database.",
)


# Define the tool for order lookup
@order_lookup_agent.tool
async def get_client_orders(ctx: RunContext[Deps]) -> List[Dict[str, Any]]:
    """Fetch orders for a client based on their ID."""
    if not isinstance(ctx.deps.client_id, int):
        return "### Error: Client ID must be an integer."
    return await ctx.deps.db.get_orders(ctx.deps.client_id)


# Example usage
if __name__ == "__main__":
    # Simulate a client request
    deps = Deps(client_id=1, db=OrderDatabase())
    try:
        result = order_lookup_agent.run_sync("Retrieve orders for the client", deps=deps)
        print(result.data)
    except ValidationError as e:
        print("Validation error:", e)
    except ModelRetry as e:
        print("Retryable error:", e)

