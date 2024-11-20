import json
import os
import re
from typing import List, Optional, Tuple

from .types.chat.chat_completion import ToolCall, FunctionCall


class ToolParser:
    """Parser for extracting tool calls from model outputs."""

    @staticmethod
    def json_parser(output: str) -> Tuple[str, List[ToolCall]]:
        """Parse JSON format tool calls.
        
        Args:
            output (str): The model output text
            
        Returns:
            Tuple[str, List[ToolCall]]: Cleaned output text and list of tool calls
        """
        tool_calls = []
        json_match = re.search(r'\{.*"tool_calls":\s*\[.*\].*\}', output, re.DOTALL)
        
        if not json_match:
            return output, []
            
        try:
            json_data = json.loads(json_match.group())
            for call in json_data.get("tool_calls", []):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=call["name"],
                            arguments=json.dumps(call["arguments"])
                        ),
                    )
                )
            # Remove the JSON from the output
            output = re.sub(
                r'\{.*"tool_calls":\s*\[.*\].*\}',
                "",
                output,
                flags=re.DOTALL
            ).strip()
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON tool calls: {e}")
            
        return output, tool_calls

    @staticmethod
    def xml_parser(output: str) -> Tuple[str, List[ToolCall]]:
        """Parse XML format tool calls.
        
        Args:
            output (str): The model output text
            
        Returns:
            Tuple[str, List[ToolCall]]: Cleaned output text and list of tool calls
        """
        tool_calls = []
        
        if "<function_calls>" not in output.lower():
            return output, []

        try:
            # Try parsing old format
            function_calls = re.findall(r"<function=(\w+)>\s*({[^<>]+})", output)
            for function_name, args_str in function_calls:
                args = json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(args)
                        ),
                    )
                )

            # Try parsing new XML format
            invoke_blocks = re.findall(
                r"<invoke>(.*?)</invoke>",
                output,
                re.DOTALL | re.IGNORECASE
            )
            for block in invoke_blocks:
                tool_name = re.search(
                    r"<tool_name>(.*?)</tool_name>",
                    block,
                    re.IGNORECASE
                )
                parameters = re.findall(
                    r"<(\w+)>(.*?)</\1>",
                    block,
                    re.IGNORECASE
                )

                if tool_name:
                    args = {
                        param[0].lower(): param[1]
                        for param in parameters
                        if param[0].lower() != "tool_name"
                    }
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{os.urandom(4).hex()}",
                            function=FunctionCall(
                                name=tool_name.group(1),
                                arguments=json.dumps(args)
                            ),
                        )
                    )

            # Remove the function calls from the output
            output = re.sub(
                r"<function_calls>.*</function_calls>",
                "",
                output,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            
        except Exception as e:
            print(f"Error parsing XML function call: {e}")
            
        return output, tool_calls

    @staticmethod
    def functools_parser(output: str) -> Tuple[str, List[ToolCall]]:
        """Parse functools format tool calls.
        
        Args:
            output (str): The model output text
            
        Returns:
            Tuple[str, List[ToolCall]]: Cleaned output text and list of tool calls
        """
        tool_calls = []
        
        if "functools[" not in output:
            return output, []

        try:
            functools_match = re.search(r"functools\[(.*?)\]", output, re.DOTALL)
            if functools_match:
                functools_data = json.loads(f"[{functools_match.group(1)}]")
                for call in functools_data:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{os.urandom(4).hex()}",
                            function=FunctionCall(
                                name=call["name"],
                                arguments=json.dumps(call["arguments"]),
                            ),
                        )
                    )
                # Remove the functools call from the output
                output = re.sub(
                    r"functools\[.*?\]",
                    "",
                    output,
                    flags=re.DOTALL
                ).strip()
                
        except Exception as e:
            print(f"Error parsing functools call: {e}")
            
        return output, tool_calls

    @staticmethod
    def qwen_parser(output: str) -> Tuple[str, List[ToolCall]]:
        """Parse Qwen format tool calls.
        
        Args:
            output (str): The model output text
            
        Returns:
            Tuple[str, List[ToolCall]]: Cleaned output text and list of tool calls
        """
        tool_calls = []
        
        if "<tool_call>" not in output:
            return output, []

        try:
            qwen_calls = re.findall(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                output,
                re.DOTALL
            )
            for call_str in qwen_calls:
                call_data = json.loads(call_str)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=call_data["name"],
                            arguments=json.dumps(call_data["arguments"])
                        ),
                    )
                )
            # Remove all <tool_call>...</tool_call> blocks from the output
            output = re.sub(
                r"<tool_call>\s*\{.*?\}\s*</tool_call>",
                "",
                output,
                flags=re.DOTALL
            ).strip()
            
        except Exception as e:
            print(f"Error parsing Qwen tool calls: {e}")
            
        return output, tool_calls

    @staticmethod
    def auto_parser(output: str) -> Tuple[str, List[ToolCall]]:
        """Automatically detect and parse tool calls using all available parsers.
        
        Args:
            output (str): The model output text
            
        Returns:
            Tuple[str, List[ToolCall]]: Cleaned output text and list of tool calls
        """
        parsers = [
            ToolParser.json_parser,
            ToolParser.xml_parser,
            ToolParser.functools_parser,
            ToolParser.qwen_parser
        ]
        
        for parser in parsers:
            cleaned_output, tool_calls = parser(output)
            if tool_calls:  # If we found any tool calls, stop searching
                return cleaned_output, tool_calls
            
        return output, []  # If no parser found anything, return original output and empty list

    @classmethod
    def get_parser(cls, parser_name: Optional[str] = None):
        """Get the appropriate parser function based on name.
        
        Args:
            parser_name (Optional[str]): Name of the parser to use
            
        Returns:
            Callable: Parser function to use
        """
        parser_map = {
            "json": cls.json_parser,
            "xml": cls.xml_parser,
            "functools": cls.functools_parser,
            "qwen": cls.qwen_parser,
            "auto": cls.auto_parser,
            None: cls.auto_parser
        }
        
        return parser_map.get(parser_name.lower() if parser_name else None, cls.auto_parser) 