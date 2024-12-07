
# Multi-Agent Task Solving System with Workspace Sharing

This repository contains a Python-based multi-agent system that facilitates task management and execution using a combination of **manager agents** and **worker agents**. The system is designed to provide a collaborative workspace for agents to share resources such as dataframes and text, enabling them to execute complex tasks effectively.

---

## **Table of Contents**
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [Manager Agent](#manager-agent)
  - [Worker Agent](#worker-agent)
- [Workspace](#workspace)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## **Overview**

This multi-agent system is built for automating complex workflows by decomposing tasks into subtasks, delegating them to specialized worker agents, and integrating their outputs. The manager agent orchestrates this process, ensuring proper execution and utilizing shared resources in the workspace.

---

## **Key Features**

- **Hierarchical Agent Structure**: 
  - Manager agents route user instructions to appropriate worker agents.
  - Worker agents interact with tools to execute specific tasks (e.g., search, coding, data analysis).

- **Shared Workspace**:
  - A collaborative environment for caching resources (e.g., dataframes, text).
  - Enables seamless communication and resource sharing among agents and users.

- **Tool Integration**:
  - Easily integrates tools like search engines, Python code interpreters, and data analysis utilities.

- **Modular and Extensible**:
  - Scalable architecture for adding new tools, agents, and capabilities.

- **Logging and Debugging**:
  - Comprehensive logging for monitoring agent activities.

---

## **Architecture**

### **Manager Agent**

The manager agent coordinates tasks by:
- Parsing user instructions.
- Decomposing tasks into subtasks.
- Assigning subtasks to the appropriate worker agents.
- Maintaining a shared workspace for collaboration.

### **Worker Agent**

Worker agents specialize in executing tasks using integrated tools:
- **Search Agents**: Perform web-based searches to gather information.
- **Coding Agents**: Solve logical problems by running Python code.
- **Data Analysis Agents**: Execute operations on dataframes cached in the workspace.

---

## **Workspace**

The workspace is a shared environment where agents and users can:
- Cache resources like dataframes or text.
- Use metadata to describe the cached resources.
- Perform operations or share outputs to assist in task execution.

---

## **Setup and Installation**

### Prerequisites
- Python 3.8 or higher
- Required libraries: Install from `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Neutrino1998/agentic-task-solver.git
   cd multi-agent-system

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- Update tool configurations and LLM models in the respective files as needed.

------

## **Usage**

### Example 1: Manager Agent with Search and Coding Agents

```python
from agents.manager import ManagerAgent
from agents.worker import WorkerAgent
from tools.search import ddg_search_engine
from tools.code_interpreter import execute_python_code
import asyncio

search_agent = WorkerAgent(
    agent_name="Search Agent",
    tools=[ddg_search_engine]
)

coding_agent = WorkerAgent(
    agent_name="Coding Agent",
    tools=[execute_python_code]
)

manager_agent = ManagerAgent(
    agent_name="Manager Agent",
    subordinates=[search_agent, coding_agent]
)

result = asyncio.run(manager_agent(
    message="What is 7 times the square root of pi?"
))
print(result)
```

### Example 2: Manager Agent with Workspace

```python
from tools.data_loader import load_csv_to_dataframe
from tools.code_interpreter import execute_python_code_with_df
from agents.worker import WorkerAgent
from agents.manager import ManagerAgent
import asyncio

file_path = "path/to/superstore.csv"
df = load_csv_to_dataframe(file_path)

data_analysis_agent = WorkerAgent(
    agent_name="Data Analysis Agent",
    tools=[execute_python_code_with_df]
)

manager_agent = ManagerAgent(
    agent_name="Manager Agent",
    subordinates=[data_analysis_agent],
    workspace={"superstore": {"content": df, "metadata": {"description": "Superstore sales data."}}}
)

result = asyncio.run(manager_agent(
    message="Group sales amount by category in superstore data."
))
print(result)
```

------

## **Future Enhancements**

- **Dynamic Tool Loading**: Add a dynamic mechanism for loading tools at runtime.
- **Improved Workspace Management**: Implement version control for workspace resources.
- **Advanced AI Models**: Integrate more powerful LLMs for task execution and optimization.

------

## **License**

This project is licensed under the MIT License. See the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.
