#!/usr/bin/env python3
import json
from datetime import datetime
import math
from os import getenv
from typing import Any, Dict, List
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Scoring function for evaluating the model performance - Creating the model_config.json file under artifacts dir
def score(context: ModelContext, **kwargs):
    aoa_create_context()
    print("started_scoring")
    
    
