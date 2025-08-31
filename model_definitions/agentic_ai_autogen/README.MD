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

The model uses the following configurable parameters in `config.json`:

| Parameter | Description |
|-----------|-------------|
| `LLM_MODEL` | The language model to use for agent interactions |
| `LLM_BASE_URL` | Base URL for the LLM API endpoint |
| `LLM_API_KEY` | API key for accessing the LLM service |

**Important**: Replace `REPLACE_WITH_YOUR_API_KEY` in `config.json` with your actual API key before deployment.

### Termination Conditions

The system implements two termination conditions:
1. **HandoffTermination**: Terminates when control is handed off to the user
2. **TextMentionTermination**: Terminates when "TERMINATE" is mentioned in the conversation

## Usage

### Input Format

The model accepts input in the following formats:

1. **Direct query parameter**:
```python
score(context, query="Your research topic here")
```

2. **Data structure**:
```python
score(context, data=[{"query": "Your research topic here"}])
```

3. **Default behavior**: If no query is provided, uses a default example about transformers.

### Output Format

The model returns a dictionary containing:
- **content**: Clean, formatted content ready for publication
- **query**: The processed query
- **timestamp**: When the processing completed
- **agent_count**: Number of messages exchanged between agents
- **error**: Error message if any issues occurred

### Example Response
```python
{
    "content": "Generated article content here...",
    "query": "Explain self attention in transformers",
    "timestamp": "2025-08-31T10:30:00",
    "agent_count": 15
}
```

## Features

### Multi-Agent Collaboration
- Agents communicate through structured handoffs
- Each agent has specialized expertise and responsibilities
- Coordinated workflow ensures comprehensive content creation

### Error Handling
- Graceful handling of configuration errors
- Detailed error messages for troubleshooting
- Fallback responses when agent execution fails

### Content Quality
- Research-backed information with factual accuracy
- Professional formatting suitable for publication
- Engaging writing style with clear structure

### Flexibility
- Configurable LLM backends (OpenAI, Groq, local models, etc.)
- Customizable agent behaviors through system messages
- Adaptable to various content types and domains

## File Structure

```
agentic_ai_autogen/
├── README.MD           # This documentation file
├── config.json         # Model configuration and hyperparameters
├── model.json          # Model metadata and description
└── model_modules/
    ├── __init__.py     # Python package initialization
    ├── training.py     # Model training logic (placeholder)
    ├── scoring.py      # Model inference/scoring logic
    ├── evaluation.py   # Model evaluation metrics (placeholder)
    └── requirements.txt # Python dependencies
```

## Dependencies

Key dependencies include:
- `autogen-agentchat`: Core agent framework
- `autogen-ext`: Extensions for various LLM providers
- `autogen-core`: Core autogen functionality
- `teradatamodelops`: ModelOps integration
- `teradataml`: Teradata ML library

See `model_modules/requirements.txt` for complete dependency list.

## Setup Instructions

1. **Install Dependencies**: Ensure all packages in `requirements.txt` are installed
2. **Configure API**: Update `config.json` with your actual API key and endpoints
3. **Test Connection**: Verify your LLM service is accessible from the deployment environment
4. **Deploy**: Use ModelOps deployment procedures for your environment

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
- Monitor API usage and costs when using external LLM services

## Limitations

- Content quality depends on the underlying LLM capabilities
- May require internet access for real-time research (depending on LLM)
- Processing time increases with complex multi-agent interactions
- Generated content should be reviewed for accuracy and bias
- API costs may vary based on usage and selected model

## Troubleshooting

### Common Issues

1. **Configuration Error**: Ensure all hyperparameters are correctly set in `config.json`
2. **API Authentication**: Verify your API key is valid and has sufficient permissions
3. **Network Issues**: Check connectivity to the LLM service endpoint
4. **Import Errors**: Ensure all required packages are installed

### Error Messages

The model provides detailed error messages in the output dictionary:
- Configuration errors: Missing or invalid hyperparameters
- Execution errors: Issues during agent communication or LLM calls
- Content errors: Problems extracting final content from agent responses

## Support

For issues or questions about this model:
1. Check the model logs for error details
2. Verify configuration parameters in `config.json`
3. Ensure all dependencies are properly installed
4. Review agent system messages for customization needs
5. Test LLM connectivity independently
