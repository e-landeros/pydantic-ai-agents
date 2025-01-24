from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
import asyncio
import nest_asyncio


nest_asyncio.apply()


# Define the Pydantic model for the structured response
class QualityReport(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset being analyzed.")
    issues_found: List[str] = Field(..., description="List of quality issues identified in the dataset.")

# Define the dependencies
@dataclass
class DatasetAnalysisDependencies:
    datasets: List[str]

# Instantiate the model
model = OllamaModel(
    model_name='qwen2.5:7b',
    base_url='http://localhost:11434/v1/',
)

# Create the agent
agent = Agent(
    model=model,
    deps_type=DatasetAnalysisDependencies,
    result_type=QualityReport,
    system_prompt=(
        "You are a data quality expert. Analyze each dataset independently and provide a report of quality issues."
    ),
)

# Define the parallel task executor
async def analyze_dataset(ctx: RunContext[DatasetAnalysisDependencies], dataset: str) -> QualityReport:
    """Analyze a single dataset for quality issues."""
    response = await agent.run(
        f"Analyze the dataset '{dataset}' for quality issues.",
        deps=DatasetAnalysisDependencies(datasets=[dataset]),
    )
    return response.data

# Define the main function to process datasets in parallel
async def process_datasets_in_parallel(datasets: List[str]):
    deps = DatasetAnalysisDependencies(datasets=datasets)
    tasks = [
        analyze_dataset(RunContext(deps), dataset) for dataset in datasets
    ]
    return await asyncio.gather(*tasks)

# Example datasets for analysis
datasets_to_analyze = ["Dataset A", "Dataset B", "Dataset C"]

# Run the parallel analysis
async def main():
    results = await process_datasets_in_parallel(datasets_to_analyze)
    for result in results:
        print(result.json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())