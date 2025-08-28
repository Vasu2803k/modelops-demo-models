import warnings
import os
from crewai import Agent, Task, Crew, LLM
from groq import Groq
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def score(context: ModelContext, **kwargs):
    aoa_create_context()
    print("started_scoring")
    os.environ["GROQ_API_KEY"] = "gsk_"
    model = "groq/openai/gpt-oss-20b"
    llm = LLM(
        model=model,
        max_completion_tokens=1024
    )

    researcher = Agent(
        role='Senior Research Analyst',
        goal='Discover new insights',
        backstory="You're an expert at finding interesting information",
        llm=llm,
        verbose=True
    )

    writer = Agent(
        role='Content Writer',
        goal='Write engaging content',
        backstory="You're a talented writer who simplifies complex information",
        llm=llm,
        verbose=True
    )

    research_task = Task(
        description='Find interesting facts about AI in healthcare',
        agent=researcher,
        expected_output="A list of interesting facts about AI applications in healthcare."
    )

    write_task = Task(
        description='Write a short blog post about AI in healthcare',
        agent=writer,
        expected_output="A concise blog post (~50 words) about AI's impact on healthcare.")

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=True
    )

    result = crew.kickoff()
    print("CrewAI Result:\n", result)

    print("Scoring")
