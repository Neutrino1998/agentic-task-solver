from langchain_core.tools import tool

# This is a special placeholder tool to let agent be aware how to response to user.
@tool
def response_to_user(text: str) -> None: 
    """
    This function will pass your final answer to the user.
    Ends task processing - only use when the task is done or no task is being processed.
    Place your result in "text" argument.

    Args:
        text: Final answer for user.

    Returns:
        None
    """
    return