"""
Training module for Autogen agent.

Since this is a conversational agent using pre-trained models,
traditional training is not required. This module can be used for:
- Configuration validation
- Knowledge base setup
- Agent initialization testing
"""

import os
import json
from tmo import ModelContext

def train(context: ModelContext, **kwargs):
    print("Training")
    config = {
         "LLM_MODEL": context.hyperparams["LLM_MODEL"],
         "LLM_BASE_URL": context.hyperparams["LLM_BASE_URL"],
         "LLM_API_KEY": context.hyperparams["LLM_API_KEY"],
      }
    with open(f"{context.artifact_output_path}/model_config.json", "w") as f:
          json.dump(config, f)
