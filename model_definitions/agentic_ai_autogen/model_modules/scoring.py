"""
Scoring module for autogen agent.

This module provides scoring capabilities for the conversational agent.
"""
import os
import json
from tmo import ModelContext
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

class ModelScorer(object):
    """
    Model scorer using CrewAI agents for collaborative tasks.
    """
    def __init__(self):
        """Initialize CrewAI agents and tasks based on config."""

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
                "structured_output": True
            }
        )

        # Define the agents
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

    def invoke(self, query):
        """
        Invoke this application using the Autogen agent.

        Args:
            features: Input data (str)

        Returns:
            json response
        """
        import asyncio
        async def _collect_async(agen):
            return [item async for item in agen]

        def sync_list(gen):
            if hasattr(gen, "__aiter__"):
                return asyncio.run(_collect_async(gen))
            else:
                return list(gen)

        try:
            result_stream = self.research_team.run_stream(task=query)
            result_list = sync_list(result_stream)
            class Response:
                def __init__(self, messages):
                    self.messages = messages
            result = Response(result_list)
            response = str(self.get_final_output(result))
        except Exception as e:
            print(f"Agent failed, returning without agent/LLM call: {str(e)}")
            response = "Error: Agent failed - " + str(e)

        return response

