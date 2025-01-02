from tools.search import bing_search_engine
from tools.code_interpreter import execute_python_code, execute_python_code_with_df
from agents.worker import WorkerAgent
from agents.manager import ManagerAgent

search_agent = WorkerAgent(
    agent_name="Search Agent",
    agent_description="A search agent which can gather information online and solve knowledge related task.",
    recursion_limit=25,
    tools=[bing_search_engine],
    llm="qwen2.5-72b-instruct",
    verbose=True
)

coding_agent = WorkerAgent(
    agent_name="Coding Agent",
    agent_description="A coding agent which can solve logical task with python code.",
    recursion_limit=25,
    tools=[execute_python_code],
    llm="qwen2.5-72b-instruct",
    verbose=True
)

data_analysis_agent = WorkerAgent(
    agent_name="Data Analysis Agent",
    agent_description="A data analysis agent which can execute python code on given dataframe cached in workspace.",
    recursion_limit=25,
    tools=[execute_python_code_with_df],
    llm="qwen2.5-72b-instruct",
    verbose=True
)

manager_agent_with_workspace = ManagerAgent(
    agent_name="Manager Agent",
    agent_description="A manager agent which can direct a search agent with knowledge related task, \
        a coding agent with logic related task, \
        and a data analysis agent which can execute python code on given dataframe cached in workspace.",
    recursion_limit=25,
    tools=[],
    subordinates=[search_agent, coding_agent, data_analysis_agent],
    workspace={},
    llm="qwen2.5-72b-instruct",
    verbose=True)