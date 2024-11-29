from inspect import signature
from langchain_core.tools.base import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage

def add_indent(input_str, indent_level: int = 1, use_strip=False):
    indent = "  " * indent_level
    if use_strip:
        return '\n'.join(indent + line for line in input_str.strip().splitlines())
    else:
        return '\n'.join(indent + line for line in input_str.splitlines())
    

def get_xml_tools(tools: list[BaseTool], indent_level: int = 0) -> str:
    """Render the tool name and description in XML format.

    Args:
        tools: The tools to render.

    Returns:
        The rendered XML string.
    """
    # Start XML structure
    xml_output = '<tool_list>'
    
    # Loop through each tool and build its XML representation
    for tool in tools:
        # Check if the tool has a function and retrieve its signature
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            # Prepare tool name, parameters, and return type
            parameter_and_return = str(sig)
            # parameter_and_return = f"({', '.join([f'{k}: {v.annotation}' for k, v in sig.parameters.items()])}) -> {sig.return_annotation.__name__}"
            # Prepare docstring
            docstring = tool.description or ""
        else:
            # If no function is available, just use tool name and description
            parameter_and_return = "N/A"
            docstring = tool.description or ""

        # Format the tool's XML
        xml_output += add_indent(f'\n<tool>', indent_level=1)
        xml_output += add_indent(f'\n<tool_name>{tool.name}</tool_name>', indent_level=2)
        xml_output += add_indent(f'\n<parameter_and_return>{parameter_and_return}</parameter_and_return>', indent_level=2)
        xml_output += add_indent(f'\n<docstring>\n{docstring}\n</docstring>', indent_level=2)
        xml_output += add_indent(f'\n</tool>', indent_level=1)
    
    # Close the tool_list tag
    xml_output += '\n</tool_list>'
    indented_xml_output = add_indent(xml_output, indent_level=indent_level)
    return indented_xml_output

def get_xml_msg_history(
    messages: list[AnyMessage], human_prefix: str = "Human", ai_prefix: str = "AI", indent_level: int = 0, 
) -> str:
    """Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
            Default is "Human".
        ai_prefix: THe prefix to prepend to contents of AIMessages. Default is "AI".

    Returns:
        A single string concatenation of all input messages.

    Raises:
        ValueError: If an unsupported message type is encountered.

    """

    msg_history_list = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        # 添加缩进到 m.content
        name = add_indent(f"\n<name>{m.name}</name>" if m.name is not None else "")
        indented_content = add_indent(m.content)
        message = f"""<{role}>{name}
{indented_content}
</{role}>"""
        msg_history_list.append(message)
    msg_history = add_indent("\n".join(msg_history_list))
    msg_history = f"""
<message_history>
{msg_history}
</message_history>
"""
    return add_indent(msg_history, indent_level=indent_level)


def get_xml_subordinates(subordinates: list, indent_level: int = 0) -> str:
    """Render the tool name and description in XML format.

    Args:
        tools: The tools to render.

    Returns:
        The rendered XML string.
    """
    # Start XML structure
    xml_output = '<agent_list>'
    
    # Loop through each tool and build its XML representation
    for agent in subordinates:
        # Format the tool's XML
        xml_output += add_indent(f'\n<agent>', indent_level=1)
        xml_output += add_indent(f'\n<agent_name>{agent.agent_name}</agent_name>', indent_level=2)
        xml_output += add_indent(f'\n<agent_description>{agent.agent_description}</agent_description>', indent_level=2)
        xml_output += add_indent(f'\n</agent>', indent_level=1)
    
    # Close the tool_list tag
    xml_output += '\n</agent_list>'
    indented_xml_output = add_indent(xml_output, indent_level=indent_level)
    return indented_xml_output

if __name__ == "__main__":
    from tools.search import ddg_search_engine
    from tools.code_interpreter import execute_python_code
    # =======================================================
    # Test Example
    print("="*80+"\n> Testing get_xml_msg_history:")
    test_messages = [
        HumanMessage(f"  So you said you were writing an article on A?"),
        AIMessage(f"Yes!"),
        HumanMessage(f"Would you like to know more info?", name="Data Specialist")
    ]
    print(get_xml_msg_history(messages=test_messages))

    print("="*80+"\n> Testing get_xml_tools:")
    rendered_tools_xml = get_xml_tools([ddg_search_engine, execute_python_code])
    print(rendered_tools_xml)
