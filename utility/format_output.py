from pydantic import BaseModel, Field
from typing import Type
import re
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages.utils import get_buffer_string
import asyncio
import uuid
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
# model
from models import get_llm
# Log
from my_logger import Logger, LOG_LEVEL, LOG_PATH, LOG_FILE
# 初始化日志记录器
logger = Logger(name="AgentLogger", level=LOG_LEVEL, log_path=LOG_PATH, log_file=LOG_FILE)

class FormatOutputGenerator:
    def __init__(self, llm, verbose: bool = True, max_retries: int = 3, retry_delay: float = 3.0):
        self.llm = llm
        self.verbose = verbose
        self.max_retries = max_retries  # Number of retries
        self.retry_delay = retry_delay  # Delay in seconds between retries
        self.response = None

    def _generate_json_schema(self, pydantic_model: BaseModel, indent_level: int = 1, include_optional: bool = True) -> str:
        schema = pydantic_model.model_json_schema()

        def generate_from_properties(properties, required_fields, indent_level):
            indent = "    " * indent_level
            lines = []

            for key, value in properties.items():
                title = value.get('title', key)
                optional = " (optional)" if include_optional and key not in required_fields else ""

                # 处理可能的 anyOf 类型
                if 'anyOf' in value:
                    for option in value['anyOf']:
                        if 'type' in option:
                            value = option
                            break
                        elif '$ref' in option:
                            value = option
                            break

                # 处理 $ref 引用
                if '$ref' in value:
                    object_definition = resolve_ref(value['$ref'], schema)
                    if object_definition:
                        object_content = generate_from_properties(object_definition['properties'],
                                                                object_definition.get('required', []),
                                                                indent_level + 1)
                        lines.append(f'{indent}"{key}": {{\n{object_content}\n{indent}}}')
                    continue

                # 处理基本类型和 Literal
                if 'type' in value:
                    if value['type'] == 'string':
                        if 'enum' in value:  # 处理 Literal 类型
                            options = " | ".join(f'"{opt}"' for opt in value['enum'])
                            lines.append(f'{indent}"{key}": "<{title}{optional}: {options}>"')
                        else:
                            lines.append(f'{indent}"{key}": "<{title}{optional}>"')
                    elif value['type'] == 'integer':
                        lines.append(f'{indent}"{key}": <{title}{optional}>')
                    elif value['type'] == 'number':
                        lines.append(f'{indent}"{key}": <{title}{optional}>')
                    elif value['type'] == 'array' and 'items' in value:
                        item_type = value['items'].get('type')
                        if item_type == 'string':
                            lines.append(f'{indent}"{key}": [\n{indent}    "<{title}{optional}>",\n{indent}    ...\n{indent}]')
                        elif item_type == 'integer':
                            lines.append(f'{indent}"{key}": [\n{indent}    <{title}{optional}>,\n{indent}    ...\n{indent}]')
                        elif item_type == 'number':
                            lines.append(f'{indent}"{key}": [\n{indent}    <{title}{optional}>,\n{indent}    ...\n{indent}]')
                        else:
                            item_ref = value['items'].get('$ref')
                            item_definition = resolve_ref(item_ref, schema) if item_ref else None
                            if item_definition:
                                array_content = generate_from_properties(item_definition['properties'],
                                                                        item_definition.get('required', []),
                                                                        indent_level + 2)
                                lines.append(f'{indent}"{key}": [\n{indent}    {{\n{array_content}\n{indent}    }},\n{indent}    ...\n{indent}] {optional}')
                # 处理对象类型
                elif value.get('type') == 'object':
                    object_content = generate_from_properties(value['properties'],
                                                            value.get('required', []),
                                                            indent_level + 1)
                    lines.append(f'{indent}"{key}": {{\n{object_content}\n{indent}}}')
            
            return ",\n".join(lines)
        
        def resolve_ref(ref, schema):
            if ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                return schema['$defs'].get(ref_name)
            return None

        root_properties = schema['properties']
        root_required = schema.get('required', [])
        
        return "{\n" + generate_from_properties(root_properties, root_required, indent_level) + "\n}"


    def _generate_output_schema(self, pydantic_model: Type[BaseModel]) -> str:
        return f"""
Please format the output using the following Json structure:
<json_format>
{self._generate_json_schema(pydantic_model, include_optional=True)}
</json_format>"""

    def _extract_json_from_aimessage(self, ai_message: AIMessage) -> AIMessage:
        """
        This function extracts the content inside <json_format> XML-like tags
        from the .content of an AIMessage object. If no tags are found, it returns 
        the original content unchanged.

        :param ai_message: An AIMessage object with content potentially containing <json_format> tags.
        :return: The updated AIMessage object with extracted content, if found.
        """
        # Regular expression to find content between <json_format> tags
        pattern = r"<json_format>(.*?)</json_format>"
        
        # Extracting content from the .content of AIMessage
        match = re.search(pattern, ai_message.content, re.DOTALL)
        
        if match:
            # If content is found between tags, update the .content field
            extracted_content = match.group(1).strip()
            ai_message.content = extracted_content
        # If no match, return the AIMessage with original content
        
        return ai_message

    async def generate(self, pydantic_model: Type[BaseModel], system_message: str, human_message: str) -> dict:
        # Generate a unique ID for this generation call
        generation_id = f"msg-{uuid.uuid4().hex}"

        # 生成输出格式提示词
        output_schema = self._generate_output_schema(pydantic_model)
        system_message = system_message + output_schema
        # 定义完整的消息内容
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]

        msg_str = get_buffer_string(messages)
        if self.verbose:
            logger.logger.debug(f"[{generation_id}] Message → LLM:\n{msg_str}")

        attempts = 0
        while attempts < self.max_retries:
            attempts += 1
            try:
                self.response = await self.llm.ainvoke(messages)

                if self.verbose:
                    input_tokens = self.response.response_metadata.get('token_usage').get("input_tokens")
                    output_tokens = self.response.response_metadata.get('token_usage').get("output_tokens")
                    logger.logger.debug(f"[{generation_id}] LLM Response [input:{input_tokens}, output:{output_tokens}]:\n{self.response.content}")

                output_parser = PydanticOutputParser(pydantic_object=pydantic_model)
                result = output_parser.invoke(self._extract_json_from_aimessage(self.response))
                return result.model_dump()  # If successful, return the result
                
            except Exception as e:
                logger.logger.warning(f"[{generation_id}] ⚠️ Error in generate format response, attempt {attempts}/{self.max_retries}: {str(e)}")
                await asyncio.sleep(self.retry_delay)  # Wait before retrying

        logger.logger.error(f"[{generation_id}] ⚠️ Max retries ({self.max_retries}) reached in generate format response.")
        raise RuntimeError(f"Max retries ({self.max_retries}) reached. Last error: {str(e)}") from e

    def get_raw_response(self, additional_args: dict = {}):
        msg_dict = {**self.response.model_dump(), **additional_args}
        returned_response = AIMessage(**msg_dict)
        return returned_response
    
if __name__ == "__main__":
    # =======================================================
    # 测试用例
    # Step 1: 定义 Generator
    test_llm = get_llm(model_name="qwen2.5-72b-instruct")
    generator = FormatOutputGenerator(test_llm, verbose=True)
    # Step 2: 定义 Pydantic 模型
    class RelatedSubjects(BaseModel):
        topics: list[str] = Field(
            description="Comprehensive list of related subjects as background research.",
        )
    # Step 3: 调用 generate
    max_sections = 5
    query = "Impact of million-plus token context window language models on RAG"
    related_topics = asyncio.run(generator.generate(
        pydantic_model=RelatedSubjects,
        system_message=f"I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend a maximum of {max_sections} Wikipedia pages on closely related subjects. \
I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, \
or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.",
        human_message=f"Topic of interest: {query}"
    )) 
    print("="*80+"\n> Test Results:")
    print(related_topics.get("topics"))
    print("-"*80+"\n> Raw Response:")
    print(generator.get_raw_response())