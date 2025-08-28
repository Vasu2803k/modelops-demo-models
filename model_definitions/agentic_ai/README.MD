# Autogen Agentic AI Model

## Overview

This model demonstrates content writing capabilities using a multi-agent system built with the Autogen framework. The system employs collaborative AI agents that work together to research, plan, and generate high-quality content through structured group chat interactions.

## Model Description

- **Name**: Autogen Agentic AI
- **ID**: c809938c-ad04-4677-b439-b2ed5c3ed158
- **Language**: Python
- **Framework**: Microsoft Autogen
- **Purpose**: Content writing with multi-agent collaboration

## Architecture

The model implements a swarm-based multi-agent system with the following key agents:

### 1. Planner Agent
- **Role**: Research and Content Writer Coordinator
- **Responsibilities**: 
  - Orchestrates the workflow between specialized agents
  - Delegates tasks to appropriate agents
  - Manages handoffs and termination conditions
- **Handoffs**: Can transfer control to `research_agent`, `content_writer_agent`, or `user`

### 2. Research Agent
- **Role**: Senior Research Analyst
- **Responsibilities**:
  - Conducts comprehensive research and analysis
  - Discovers new insights across various domains
  - Provides well-researched, factual information
  - Delivers comprehensive analysis and interesting insights
- **Handoffs**: Returns control to `planner` when research is complete

### 3. Content Writer Agent
- **Role**: Content Writer
- **Responsibilities**:
  - Transforms research findings into readable, engaging content
  - Simplifies complex information into clear, concise articles
  - Focuses on clarity, engagement, and accessibility
  - Formats content for blog posts or articles
- **Handoffs**: Returns control to `planner` when content writing is complete

## Configuration

### Hyperparameters

The model uses the following configurable parameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `LLM_MODEL` | `gpt-3.5-turbo` | The language model to use for agent interactions |
| `LLM_BASE_URL` | `http://localhost:4000` | Base URL for the LLM API endpoint |
| `LLM_API_KEY` | `sk-1234` | API key for accessing the LLM service |

### Termination Conditions

The system implements two termination conditions:
1. **HandoffTermination**: Terminates when control is handed off to the user
2. **TextMentionTermination**: Terminates when "TERMINATE" is mentioned in the conversation

## Usage

### Input Format

The model accepts natural language prompts that describe content creation tasks. Example:

```
Find interesting facts about AI in healthcare. Focus on recent developments, 
applications, and impact. Provide a comprehensive list of facts that would 
be useful for a blog post.
```

### Output Format

The model returns a `TaskResult` object containing:
- **messages**: Complete conversation history between agents
- **stop_reason**: Reason for termination
- **final_content**: Clean, formatted content ready for publication

### Extracting Final Output

To get only the final content without agent conversation metadata:

```python
# Extract clean final output
def get_final_output(response):
    content_writer_messages = [msg for msg in response.messages 
                              if hasattr(msg, 'source') and msg.source == 'content_writer_agent' 
                              and hasattr(msg, 'type') and msg.type == 'TextMessage']

    if content_writer_messages:
        content = content_writer_messages[-1].content
        # Clean up handoff instructions
        content = content.split("**[Handing off to")[0].strip()
        if content.endswith("---"):
            content = content[:-3].strip()
        return content
    return "No content found"
```

## Features

### Multi-Agent Collaboration
- Agents communicate through structured handoffs
- Each agent has specialized expertise and responsibilities
- Coordinated workflow ensures comprehensive content creation

### Content Quality
- Research-backed information with factual accuracy
- Professional formatting suitable for publication
- Engaging writing style with clear structure

### Flexibility
- Configurable LLM backends (OpenAI, local models, etc.)
- Customizable agent behaviors through system messages
- Adaptable to various content types and domains

## File Structure

```
agentic_ai/
├── README.MD           # This documentation file
├── config.json         # Model configuration and hyperparameters
├── model.json          # Model metadata and description
└── model_modules/
    ├── __init__.py     # Python package initialization
    ├── training.py     # Model training logic
    ├── scoring.py      # Model inference/scoring logic
    ├── evaluation.py   # Model evaluation metrics
    └── requirements.txt # Python dependencies
```

## Dependencies

Key dependencies include:
- `autogen-agentchat`: Core agent framework
- `autogen-ext`: Extensions for various LLM providers
- `autogen-core`: Core autogen functionality

See `model_modules/requirements.txt` for complete dependency list.

## Use Cases

This model is particularly effective for:

1. **Blog Post Creation**: Research and write comprehensive blog articles
2. **Content Marketing**: Generate engaging marketing content
3. **Technical Documentation**: Create detailed technical articles
4. **Research Reports**: Compile and present research findings
5. **Educational Content**: Develop informative educational materials

## Example Workflow

1. **Input**: User provides a content creation prompt
2. **Planning**: Planner agent analyzes the request and delegates to research agent
3. **Research**: Research agent gathers comprehensive information on the topic
4. **Content Creation**: Content writer agent transforms research into engaging content
5. **Output**: Final formatted content ready for publication

## Best Practices

- Provide clear, specific prompts for better results
- Allow sufficient time for multi-agent collaboration
- Review and fact-check generated content before publication
- Customize system messages for domain-specific requirements

## Limitations

- Content quality depends on the underlying LLM capabilities
- May require internet access for real-time research (depending on LLM)
- Processing time increases with complex multi-agent interactions
- Generated content should be reviewed for accuracy and bias

## Support

For issues or questions about this model:
1. Check the model logs for error details
2. Verify configuration parameters
3. Ensure all dependencies are properly installed
4. Review agent system messages for customization needs
