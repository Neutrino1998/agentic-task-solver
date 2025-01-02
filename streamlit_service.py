import streamlit as st
import pandas as pd
import json
from langchain_core.messages import HumanMessage, AIMessage
from agents.preconfig_agents import manager_agent_with_workspace
import asyncio
import re
from utility.data_loader import load_csv_to_dataframe
import chardet
import os

# 配置页面信息
st.set_page_config(
    page_title="智能体任务助手",
    page_icon="🤖",
    layout="wide",

)

# Streamlit 页面标题
st.markdown("""
# 🤖智能体任务助手
`task.assistant.2024.12.27`
""")

# 布局分为两列：对话界面和 workspace 展示
chat_col, workspace_col = st.columns([1, 1])

# 左侧：对话界面
with chat_col:
    st.header("对话界面")
    chat_container = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    with chat_container:
        for message in st.session_state['messages']:
            chat_avatar = None
            if message["role"] != "user":
                chat_avatar = ":material/smart_toy:"
            with st.chat_message(message["role"], avatar=chat_avatar):
                st.markdown(message["content"])

    if user_prompt := st.chat_input("有什么可以帮忙的？"):
        
        with chat_container:
            st.session_state['messages'].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

        async def stream_output():
            stream_graph = manager_agent_with_workspace.astream(
                message=HumanMessage(content=user_prompt, name="User")
            )
            async for state in stream_graph:
                if "manager_agent" in state:
                    raw_msg = state.get("manager_agent").get("manager_messages")[0]
                    agent_name = raw_msg.name
                    thoughts = "\n".join(state.get("manager_agent").get("format_messages").get("thoughts"))
                    tool_name = state.get("manager_agent").get("format_messages").get("tool_name")
                    tool_args = state.get("manager_agent").get("format_messages").get("tool_args")
                    subordinate_name = tool_args.get("agent_name")
                    subordinate_prompt = tool_args.get("message")
                    format_msg = f"""
`{agent_name}`
```
🤔 [Thinking...]  
{thoughts}
```
"""
                    if tool_name == "call_subordinate":
                        format_msg += f"""
```
📞 [Calling {subordinate_name}...] 
{subordinate_prompt}              
```     
"""
                    elif tool_name == "response_to_user":
                        format_msg += tool_args.get("text")
                elif "tool_call" in state:
                    raw_msg = state.get("tool_call").get("manager_messages")
                    agent_name = raw_msg.name
                    agent_msg = raw_msg.content
                    # extract agent response from xml tag
                    pattern = r"<agent_call_result>(.*?)</agent_call_result>"
                    # Extracting content from the .content of AIMessage
                    match = re.search(pattern, agent_msg, re.DOTALL)
                    extracted_content = ""
                    if match:
                        # If content is found between tags, update the .content field
                        extracted_content = match.group(1).strip()
                    format_msg = f"""
`{agent_name}`

{extracted_content}
"""
                with chat_container:
                    with st.chat_message(agent_name, avatar=":material/smart_toy:"):
                        st.markdown(format_msg)
                st.session_state['messages'].append({"role": agent_name, "content": format_msg})
        asyncio.run(stream_output())

    def clear_history():
        manager_agent_with_workspace.clear_memory()
        st.session_state['messages'] = []

    st.button(label="清空聊天记录", on_click=clear_history)
    
# 右侧：Workspace 展示和上传功能
with workspace_col:
    st.header("共享 Workspace")

    # 显示当前 Workspace
    workspace = manager_agent_with_workspace.get_workspace()
    if workspace:
        st.write("当前 Workspace 内容：")
        for workspace_key, workspace_content in workspace.items():
            content = workspace_content.get("content")
            description = workspace_content.get("metadata").get("description", "No description.")
            with st.container(border=True):
                st.write(f"**{workspace_key}**: {description}")
                if isinstance(content, pd.DataFrame):
                    st.dataframe(content)
                elif isinstance(content, str):
                    st.markdown(f"""
```
{content}                             
```
""")


    else:
        with st.container(border=True):
            st.write("当前 Workspace 为空。")

    # 上传文件
    st.subheader("上传文件到 Workspace")
    upload_container = st.container(border=True)
    with upload_container:
        uploaded_file = st.file_uploader("选择文件", type=["csv", "txt", "json"])
        description_col, submit_col = st.columns([4, 1], vertical_alignment="bottom")
        with description_col:
            file_description = st.text_input("文件描述", placeholder="为文件添加描述...")
        with submit_col:
            if st.button("上传", use_container_width=True):
                if uploaded_file is not None:
                    file_name = uploaded_file.name
                    file_key = os.path.splitext(file_name)[0]  # 去除文件后缀作为 key
                    chardet_result = chardet.detect(uploaded_file.getvalue())
                    encoding = chardet_result['encoding']

                    try:
                        if file_name.endswith(".csv"):
                            # 读取 CSV 文件
                            content = pd.read_csv(uploaded_file, encoding=encoding)
                        elif file_name.endswith(".json"):
                            # 读取 JSON 文件
                            content = json.load(uploaded_file)
                        elif file_name.endswith(".txt"):
                            # 读取文本文件
                            content = uploaded_file.getvalue().decode(encoding)
                        else:
                            st.warning("不支持的文件类型！")
                            content = None

                        if content is not None:
                            # 更新 workspace，使用去除后缀的文件名作为 key
                            manager_agent_with_workspace.update_workspace(
                                {
                                    file_key: {  # 使用 file_key 而不是 file_name
                                        "content": content,
                                        "metadata": {
                                            "description": file_description
                                        }
                                    }
                                }
                            )
                            st.success(f"文件 '{file_name}' 已成功上传并添加到 Workspace！")
                            st.rerun()
                    except Exception as e:
                        st.error(f"文件读取失败：{e}")