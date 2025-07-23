# 🔍 LangSmith Tracing Setup Guide

## Overview
This guide helps you set up LangSmith tracing for conversational threads in the warehouse management system.

## 🚀 Quick Setup

### 1. Environment Variables
Add these variables to your `.env` file in the backend directory:

```bash
# LangSmith Tracing Configuration
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=warehouse-management-system
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 2. Get Your LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up or log in
3. Navigate to Settings > API Keys
4. Create a new API key
5. Copy the key to your `.env` file

### 3. Install Required Dependencies
The required packages should already be in `requirements.txt`:
```bash
pip install langsmith
```

## 🔧 Configuration Details

### Environment Variables Explained:

- **`LANGSMITH_TRACING`**: Enable/disable tracing (`true` or `false`)
- **`LANGSMITH_PROJECT`**: Project name in LangSmith dashboard
- **`LANGSMITH_API_KEY`**: Your LangSmith API key
- **`LANGSMITH_ENDPOINT`**: LangSmith API endpoint (usually default)

### 📊 What Gets Traced:

1. **Agent Runs**: Every agent interaction with metadata
2. **Knowledge Base Queries**: RAG retrieval operations
3. **Conversation Threads**: Thread-specific tracking
4. **Agent Selection**: Which agent handles each query
5. **Message Processing**: Complete conversation flow

### 🔍 Tracing Metadata:

Each trace includes:
```json
{
  "role": "manager",
  "conversation_id": "conv_123",
  "user_id": "user_456",
  "query_type": "inventory",
  "thread_id": "conv_123_user_456_1703123456",
  "has_kb_results": true,
  "kb_results_count": 3,
  "enhanced_query": true,
  "agent_type": "manager_agent"
}
```

## 🐛 Troubleshooting

### Common Issues:

1. **"LangSmith client initialization failed"**
   - Check your API key
   - Verify internet connection
   - Ensure LANGSMITH_API_KEY is set

2. **No traces appearing in dashboard**
   - Verify LANGSMITH_TRACING=true
   - Check project name matches
   - Restart the backend server

3. **Thread tracking not working**
   - Ensure conversation_id is consistent
   - Check thread_id format in traces
   - Verify metadata is being passed

### Debug Mode:
To see tracing debug info, set:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_VERBOSE=true
```

## 🔗 Viewing Traces

1. Go to [LangSmith Dashboard](https://smith.langchain.com/)
2. Select your project: `warehouse-management-system`
3. View traces by:
   - **Thread ID**: Track full conversations
   - **Agent Role**: See agent-specific performance
   - **Query Type**: Analyze by operation type

## 📈 Benefits

- **Performance Monitoring**: Track response times and success rates
- **Debug Agent Behavior**: See exactly what happens during execution
- **Conversation Analysis**: Understand user interaction patterns
- **Optimization Insights**: Data-driven agent improvements
- **Error Tracking**: Identify and fix issues quickly 