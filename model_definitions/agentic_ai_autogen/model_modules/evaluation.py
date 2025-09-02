"""
Evaluation module for autogen agent.

This module provides evaluation capabilities for the conversational agent,
including response quality metrics and conversation analysis.
"""

import json
from tmo import ModelContext

def evaluate(context: ModelContext, **kwargs):
    print("Evaluating performance...")
    config = {
        "LLM_MODEL": context.hyperparams["LLM_MODEL"],
        "LLM_BASE_URL": context.hyperparams["LLM_BASE_URL"],
        "LLM_API_KEY": context.hyperparams["LLM_API_KEY"],
    }

    with open(f"{context.artifact_output_path}/model_config.json", "w") as f:
        json.dump(config, f)
