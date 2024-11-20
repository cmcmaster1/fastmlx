#!/usr/bin/env python

"""Tests for `fastmlx` package."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import the actual classes and functions
from fastmlx import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelProvider,
    Usage,
    app,
    handle_function_calls,
    ToolParser,
)


# Create mock classes that inherit from the original classes
class MockModelProvider(ModelProvider):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_model(self, model_name: str):
        if model_name not in self.models:
            model_type = "vlm" if "llava" in model_name.lower() else "lm"
            self.models[model_name] = {
                "model": MagicMock(),
                "processor": MagicMock(),
                "tokenizer": MagicMock(),
                "image_processor": MagicMock() if model_type == "vlm" else None,
                "config": {"model_type": model_type},
            }
        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False

    async def get_available_models(self):
        return list(self.models.keys())


# Mock MODELS dictionary
MODELS = {"vlm": ["llava"], "lm": ["phi"]}


# Mock functions
def mock_generate(*args, **kwargs):
    return "generated response", {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def mock_vlm_stream_generate(*args, **kwargs):
    yield "Hello"
    yield " world"
    yield "!"


def mock_lm_stream_generate(*args, **kwargs):
    yield "Testing"
    yield " stream"
    yield " generation"


@pytest.fixture(scope="module")
def client():
    # Apply patches
    with patch("fastmlx.fastmlx.model_provider", MockModelProvider()), patch(
        "fastmlx.fastmlx.vlm_generate", mock_generate
    ), patch("fastmlx.fastmlx.lm_generate", mock_generate), patch(
        "fastmlx.fastmlx.MODELS", MODELS
    ), patch(
        "fastmlx.utils.vlm_stream_generate", mock_vlm_stream_generate
    ), patch(
        "fastmlx.utils.lm_stream_generate", mock_lm_stream_generate
    ):
        yield TestClient(app)


def test_chat_completion_vlm(client):
    request = ChatCompletionRequest(
        model="test_llava_model",
        messages=[ChatMessage(role="user", content="Hello")],
        image="test_image",
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]
    assert "usage" in response.json()
    usage = response.json()["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_chat_completion_lm(client):
    request = ChatCompletionRequest(
        model="test_phi_model", messages=[ChatMessage(role="user", content="Hello")]
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]
    assert "usage" in response.json()
    usage = response.json()["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


@pytest.mark.asyncio
async def test_vlm_streaming(client):

    # Mock the vlm_stream_generate function
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test_llava_model",
            "messages": [{"role": "user", "content": "Describe this image"}],
            "image": "base64_encoded_image_data",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = list(response.iter_lines())
    assert len(chunks) == 8  # 7 content chunks + [DONE]
    for chunk in chunks[:-2]:  # Exclude the [DONE] message
        if chunk:
            chunk = chunk.split("data: ")[1]
            data = json.loads(chunk)
            assert "id" in data
            assert data["object"] == "chat.completion.chunk"
            assert "created" in data
            assert data["model"] == "test_llava_model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["index"] == 0
            assert "delta" in data["choices"][0]
            assert "role" in data["choices"][0]["delta"]
            assert "content" in data["choices"][0]["delta"]
            if "usage" in data:
                usage = data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage

    assert chunks[-2] == "data: [DONE]"


@pytest.mark.asyncio
async def test_lm_streaming(client):

    # Mock the lm_stream_generate function
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test_phi_model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = list(response.iter_lines())
    assert len(chunks) == 8  # 7 content chunks + [DONE]

    for chunk in chunks[:-2]:  # Exclude the [DONE] message
        if chunk:
            chunk = chunk.split("data: ")[1]
            data = json.loads(chunk)
            assert "id" in data
            assert data["object"] == "chat.completion.chunk"
            assert "created" in data
            assert data["model"] == "test_phi_model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["index"] == 0
            assert "delta" in data["choices"][0]
            assert "role" in data["choices"][0]["delta"]
            assert "content" in data["choices"][0]["delta"]
            if "usage" in data:
                usage = data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage

    assert chunks[-2] == "data: [DONE]"


def test_get_supported_models(client):
    response = client.get("/v1/supported_models")
    assert response.status_code == 200
    data = response.json()
    assert "vlm" in data
    assert "lm" in data
    assert data["vlm"] == ["llava"]
    assert data["lm"] == ["phi"]


def test_list_models(client):
    client.post("/v1/models?model_name=test_llava_model")
    client.post("/v1/models?model_name=test_phi_model")

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {model["id"] for model in response.json()["data"]}
    assert model_ids == {"test_llava_model", "test_phi_model"}


def test_add_model(client):
    response = client.post("/v1/models?model_name=new_llava_model")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model new_llava_model added successfully",
    }


def test_remove_model(client):
    # Add a model
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200

    # Verify the model is added
    response = client.get("/v1/models")
    model_ids = {model["id"] for model in response.json()["data"]}
    assert "test_model" in model_ids

    # Remove the model
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204

    # Verify the model is removed
    response = client.get("/v1/models")
    model_ids = {model["id"] for model in response.json()["data"]}
    assert "test_model" not in model_ids

    # Try to remove a non-existent model
    response = client.delete("/v1/models?model_name=non_existent_model")
    assert response.status_code == 404
    assert "Model 'non_existent_model' not found" in response.json()["detail"]


def test_handle_function_calls_json_format():
    output = """Here's the weather forecast:
    {"tool_calls": [{"name": "get_weather", "arguments": {"location": "New York", "date": "2023-08-15"}}]}
    """
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "New York",
        "date": "2023-08-15",
    }
    assert "Here's the weather forecast:" in response.choices[0]["message"]["content"]
    assert '{"tool_calls":' not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_xml_format_old():
    output = """Let me check that for you.
    <function_calls>
    <function=get_stock_price>{"symbol": "AAPL"}</function>
    </function_calls>
    """
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_stock_price"
    assert json.loads(response.tool_calls[0].function.arguments) == {"symbol": "AAPL"}
    assert "Let me check that for you." in response.choices[0]["message"]["content"]
    assert "<function_calls>" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_xml_format_new():
    output = """I'll get that information for you.
    <function_calls>
    <invoke>
    <tool_name>search_database</tool_name>
    <query>latest smartphones</query>
    <limit>5</limit>
    </invoke>
    </function_calls>
    """
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "search_database"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "query": "latest smartphones",
        "limit": "5",
    }
    assert (
        "I'll get that information for you."
        in response.choices[0]["message"]["content"]
    )
    assert "<function_calls>" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_functools_format():
    output = """Here are the results:
    functools[{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}]
    """
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "format": "fahrenheit",
    }
    assert "Here are the results:" in response.choices[0]["message"]["content"]
    assert "functools[" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_multiple_functools():
    output = """Here are the results:
    functools[{"name": "get_weather", "arguments": {"location": "New York"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]
    """
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)
    assert isinstance(response, ChatCompletionResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "New York"
    }
    assert response.tool_calls[1].function.name == "get_time"
    assert json.loads(response.tool_calls[1].function.arguments) == {"timezone": "EST"}
    assert "Here are the results:" in response.choices[0]["message"]["content"]
    assert "functools[" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_qwen_format():
    output = """Let me help you with that.
    <tool_call>{"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}</tool_call>
    Here's what I found.
    <tool_call>{"name": "get_time", "arguments": {"timezone": "GMT"}}</tool_call>
    """
    # Create model_data structure with default auto parser
    model_data = {
        "model_name": "test_model",
        "tool_parser": ToolParser.get_parser()  # This will return the auto_parser
    }
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, model_data, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 2
    
    # Check first tool call
    assert response.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "London",
        "unit": "celsius"
    }
    
    # Check second tool call
    assert response.tool_calls[1].function.name == "get_time"
    assert json.loads(response.tool_calls[1].function.arguments) == {
        "timezone": "GMT"
    }
    
    # Check cleaned content
    assert "Let me help you with that." in response.choices[0]["message"]["content"]
    assert "Here's what I found." in response.choices[0]["message"]["content"]
    assert "<tool_call>" not in response.choices[0]["message"]["content"]
    
    # Check usage info
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_parser_selection():
    # Test output with all formats
    outputs = {
        "json": """Here's the weather forecast:
            {"tool_calls": [{"name": "get_weather", "arguments": {"location": "New York", "date": "2023-08-15"}}]}
            """,
        "xml": """Let me check that for you.
            <function_calls>
            <function=get_stock_price>{"symbol": "AAPL"}</function>
            </function_calls>
            """,
        "functools": """Here are the results:
            functools[{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}]
            """,
        "qwen": """Let me help you with that.
            <tool_call>{"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}</tool_call>
            """
    }
    
    expected_results = {
        "json": ("get_weather", {"location": "New York", "date": "2023-08-15"}),
        "xml": ("get_stock_price", {"symbol": "AAPL"}),
        "functools": ("get_current_weather", {"location": "San Francisco, CA", "format": "fahrenheit"}),
        "qwen": ("get_weather", {"location": "London", "unit": "celsius"})
    }
    
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    # Test each parser specifically
    for format_name, output in outputs.items():
        # Test specific parser
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser(format_name)
        }
        response = handle_function_calls(output, model_data, token_info)
        
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == expected_results[format_name][0]
        assert json.loads(response.tool_calls[0].function.arguments) == expected_results[format_name][1]
        
        # Test auto parser with same output
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser()  # Use auto parser
        }
        auto_response = handle_function_calls(output, model_data, token_info)
        
        # Auto parser should give same results as specific parser
        assert isinstance(auto_response, ChatCompletionResponse)
        assert len(auto_response.tool_calls) == 1
        assert auto_response.tool_calls[0].function.name == expected_results[format_name][0]
        assert json.loads(auto_response.tool_calls[0].function.arguments) == expected_results[format_name][1]


def test_multiple_tool_calls():
    # Test multiple tool calls in different formats
    outputs = {
        "functools": """Here are the results:
            functools[{"name": "get_weather", "arguments": {"location": "New York"}}, 
                     {"name": "get_time", "arguments": {"timezone": "EST"}}]
            """,
        "qwen": """Let me help you with that.
            <tool_call>{"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}</tool_call>
            Here's what I found.
            <tool_call>{"name": "get_time", "arguments": {"timezone": "GMT"}}</tool_call>
            """
    }
    
    expected_results = {
        "functools": [
            ("get_weather", {"location": "New York"}),
            ("get_time", {"timezone": "EST"})
        ],
        "qwen": [
            ("get_weather", {"location": "London", "unit": "celsius"}),
            ("get_time", {"timezone": "GMT"})
        ]
    }
    
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    # Test each format with multiple calls
    for format_name, output in outputs.items():
        # Test specific parser
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser(format_name)
        }
        response = handle_function_calls(output, model_data, token_info)
        
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.tool_calls) == len(expected_results[format_name])
        for i, (expected_name, expected_args) in enumerate(expected_results[format_name]):
            assert response.tool_calls[i].function.name == expected_name
            assert json.loads(response.tool_calls[i].function.arguments) == expected_args
        
        # Test auto parser with same output
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser()
        }
        auto_response = handle_function_calls(output, model_data, token_info)
        
        assert isinstance(auto_response, ChatCompletionResponse)
        assert len(auto_response.tool_calls) == len(expected_results[format_name])
        for i, (expected_name, expected_args) in enumerate(expected_results[format_name]):
            assert auto_response.tool_calls[i].function.name == expected_name
            assert json.loads(auto_response.tool_calls[i].function.arguments) == expected_args


def test_parser_error_handling():
    # Test invalid formats for each parser type
    invalid_outputs = {
        "json": """{"tool_calls": [{"name": "get_weather", "arguments": invalid_json}]}""",
        "xml": """<function_calls><function=get_weather>{invalid json}</function></function_calls>""",
        "functools": """functools[{"name": "get_weather", invalid_json}]""",
        "qwen": """<tool_call>{"name": "get_weather", invalid json}</tool_call>"""
    }
    
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    for format_name, invalid_output in invalid_outputs.items():
        # Test specific parser
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser(format_name)
        }
        response = handle_function_calls(invalid_output, model_data, token_info)
        
        # Should return response with no tool calls when parsing fails
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.tool_calls) == 0
        
        # Test auto parser
        model_data = {
            "model_name": "test_model",
            "tool_parser": ToolParser.get_parser()
        }
        auto_response = handle_function_calls(invalid_output, model_data, token_info)
        
        # Should return response with no tool calls when all parsers fail
        assert isinstance(auto_response, ChatCompletionResponse)
        assert len(auto_response.tool_calls) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
