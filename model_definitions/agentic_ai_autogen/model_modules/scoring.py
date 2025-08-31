#!/usr/bin/env python3
import json
from datetime import datetime
from typing import Any, Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from tmo import ModelContext
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Scoring function for evaluating the model performance
def score(context: ModelContext, **kwargs):
    print("Started scoring")
    
    try:
        # Get query from input data or use default
        input_data = kwargs.get('data', {})
        if isinstance(input_data, list) and len(input_data) > 0:
            query = input_data[0].get('query', 'Explain self attention in transformers')
        else:
            query = kwargs.get('query', 'Explain self attention in transformers')
        
        model_client = OpenAIChatCompletionClient(
            model=context.hyperparams["LLM_MODEL"],
            base_url=context.hyperparams["LLM_BASE_URL"],
            api_key=context.hyperparams["LLM_API_KEY"],
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.R1,
                "structured_output": True
            }
        )

        planner = AssistantAgent(
            "planner",
            description="An agent for research planning and orchestration.",
            handoffs=["research_agent", "content_writer_agent","user"],
            model_client=model_client,
            system_message="""
            You are a Research and Content Writer Coordinator. Coordinate research and writing by delegating to specialized agents:
            - research_agent: An agent for conducting research and analysis
            - content_writer_agent: An agent for writing content based on research.

            You must first respond with a concise answer why you are handing off to a particular agent, and then hand off to one of the available agents.
            Always handoff to a single agent at a time.
            Use TERMINATE when research and writing content is complete or Hand off to user if the work is done.
            """
        )
        
        research_agent = AssistantAgent(
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

        content_writer_agent = AssistantAgent(
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

        research_team = Swarm(
            participants=[planner, research_agent, content_writer_agent], 
            termination_condition=termination
        )
        
        print(f"Processing query: {query}")
        response = research_team.run_stream(task=query)
        
        # Get all text messages from content writer agent
        text_messages = [
            msg for msg in response.messages 
            if hasattr(msg, 'content') and hasattr(msg, 'source') 
            and msg.source == 'content_writer_agent' 
            and hasattr(msg, 'type') and msg.type == 'TextMessage'
        ]

        if text_messages:
            content = text_messages[-1].content
            # Clean up handoff instructions and formatting
            content = content.split("**[Handing off to")[0].strip()
            if content.endswith("---"):
                content = content[:-3].strip()
            
            print("Scoring completed successfully")
            return {
                "content": content,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "agent_count": len(response.messages)
            }
        else:
            print("No content found from content writer agent")
            return {
                "content": "No final output found",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": "No content writer messages found"
            }
            
    except KeyError as e:
        error_msg = f"Missing required configuration: {str(e)}"
        print(f"Configuration error: {error_msg}")
        return {
            "content": f"Configuration error: {error_msg}",
            "query": kwargs.get('query', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        print(f"Execution error: {error_msg}")
        return {
            "content": f"Error: {error_msg}",
            "query": kwargs.get('query', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        }
