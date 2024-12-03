from langchain_core.tools import tool

# This is a special placeholder tool to let manager agent be aware how to call subordinate agents.
@tool
def call_subordinate(agent_name:str, message: str, reset: bool) -> None: 
    """
    This function is used to call subordinate agents to solve subtasks.
    Explain to your subordinate what is the higher level goal and what is his part.
    Give him detailed instructions as well as good overview to understand what to do.

    Args:
        agent_name: Your selected subordinate agent.
        message: Message sent to your subordinate agent to instruct his task in detail.
        reset: Use "reset" argument with "True" to start with new subordinate or "False" to continue with existing. For brand new tasks use "True", for followup conversation use "False".

    Returns:
        None
    """
    return