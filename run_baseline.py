# -*- coding: utf-8 -*-
# @Date    : 4/1/2025 21:01 PM
# @Author  : didi
# @Desc    : Run baseline to test prompts 

import asyncio
from typing import Literal
from scripts.operators import Custom
from scripts.workflow import Workflow
from scripts.async_llm import LLMsConfig
from benchmarks.hotpotqa import HotpotQABenchmark
from scripts.logs import logger
from scripts.async_llm import create_llm_instance

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class PoWorkflow(Workflow):
    def __init__(
        self,
        test_prompt: str,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.prompt = test_prompt
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.response = Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.response(input=problem, instruction=self.prompt)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]

async def main():
    r1_llm_config = LLMsConfig.default().get("deepseek-reasoner")
    spo_prompt = "think step by step"
    spo_test_workflow = PoWorkflow(test_prompt=spo_prompt, name="spo_test", llm_config=r1_llm_config, dataset="HotpotQA")
    # file path refer to the dataset you want to use.
    # log_path refer to the folder of output csv.
    test_hotpotqa_benchmark = HotpotQABenchmark(name="HotpotQA", file_path="data/hotpotqa_validate.jsonl", log_path="")
    
    results = await test_hotpotqa_benchmark.run_baseline(spo_test_workflow)

if __name__ == "__main__":
    asyncio.run(main())
