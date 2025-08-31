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
    config = {
      "LLM_MODEL": context.hyperparams["LLM_MODEL"],
      "LLM_BASE_URL": context.hyperparams["LLM_BASE_URL"],
      "LLM_API_KEY": context.hyperparams["LLM_API_KEY"],
   }

    with open(f"{context.artifact_output_path}/model_config.json", "w") as f:
        json.dump(config, f)
    
class ModelScorer(object):
    """
    Model scorer for Autogen agent without tool use capabilities. - Tools will be added later!
    """
    
    def __init__(self):
        """Initialize the Autogen model with tools"""

        with open("artifacts/input/model_config.json", "r") as f:
            config = json.load(f)

        model_client = OpenAIChatCompletionClient(
            model=config["LLM_MODEL"],
            base_url=config["LLM_BASE_URL"],
            api_key=config["LLM_API_KEY"],
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.R1,
                "structured_output": True,
            },
        )

        self.planner = AssistantAgent(
            "planner",
            description="An agent for research planning and orchestration.",
            handoffs=["research_agent", "content_writer_agent","user"],
            model_client=model_client,
            system_message="""
            You are a Research and Content Writer Coordinator. Coordinate research and writing by delegating to specialized agents:
            - research_agent: An agent for conducting research and analysis
            - content_writer_agent: An agent for writing content based on research.

            You must first respond first with a concise answer why you are handing off to a particular agent, and then hand off to one of the available agents.
            Always handoff to a single agent at a time.
            Use TERMINATE when research and writing content is complete or Hand off to user if the work is done.
            """
        )
        self.research_agent = AssistantAgent(
            "research_agent",
            description="An agent for conducting research and analysis.",
            handoffs=["planner"],
            model_client=model_client,
            system_message="""
            You are a Senior Research Analyst. Your goal is to discover new insights.
            You're an expert at finding interesting information across various domains.
            Always provide well-researched, factual information with clear explanations.
            Focus on delivering comprehensive analysis and interesting insights.
            Always handoff back to planner when research is complete.
            """,
        )

        self.content_writer_agent = AssistantAgent(
            "content_writer_agent",
            description="An agent for writing the content.",
            handoffs=["planner"],
            model_client=model_client,
            system_message="""
            You are a Content Writer. Your goal is to write engaging content.
            You're a talented writer who simplifies complex information into clear, concise content.
            Take research findings and transform them into readable, engaging blog posts or articles.
            Focus on clarity, engagement, and making complex topics accessible.
            Always handoff back to planner when content writing is complete.
            """,
        )

        # Define termination condition
        termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")

        self.research_team = Swarm(
            participants=[self.planner, self.research_agent, self.content_writer_agent], termination_condition=termination
        )

        return self.research_team
    
    # Extract only the final output content
    def get_final_output(self,response):
        """Extract the final meaningful content from the agent response"""
        # Get all text messages from agents (excluding handoff messages and tool calls)
        text_messages = [msg for msg in response.messages if hasattr(msg, 'content') and hasattr(msg, 'source') and msg.source != 'user' and msg.source == 'content_writer_agent' and hasattr(msg, 'type') and msg.type == 'TextMessage']
        
        if text_messages:
            content = text_messages[-1].content
            # Remove handoff text and JSON from the end
            content = content.split("**[Handing off to")[0].strip()
            return content
        else:
            return "No final output found"
        
    async def invoke(self, query):
        """
        Make predictions using the Autogen agent.

        Args:
            features: Input data (str)

        Returns:
            json response
        """

        query = query["message"]
            
        try:
            result = await self.research_team.run_stream(task=query)
            response = str(self.get_final_output(result))
        except Exception as e:
            try:
                print(f"Agent failed, trying direct LLM call: {str(e)}")
                fallback_result = await self.content_writer_agent.run_stream(task=query)
                response = str(self.get_final_output(fallback_result))
            except Exception as fallback_e:
                response = f"Error: Agent failed - {str(e)}, Direct LLM also failed - {str(fallback_e)}"

        return response

    def explain(self, features):
        """
        Provide explanations for predictions (optional method).
        For now, this just returns the same as predict.
        """
        return self.invoke(features)
