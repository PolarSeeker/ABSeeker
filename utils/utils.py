"""Tool loading and execution helpers for recover_intermediate_steps_deepseek."""
import ast
import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
import types
from functools import partial
from typing import Any, Dict, List


async def process_tool_call(
    callable_tools,
    tool_name,
    tool_input,
    execution_cwd=os.getcwd(),
    session_env: dict = None,
):
    tool_function = callable_tools[tool_name]
    if not tool_function:
        return "Tool name cannot be empty!"
    try:
        expected_params = inspect.signature(tool_function).parameters.keys()
    except Exception as e:
        return f"Error: Could not inspect the signature of tool '{tool_name}': {e}"
    filtered_args = {
        key: value for key, value in tool_input.items() if key in expected_params
    }
    try:
        if asyncio.iscoroutinefunction(tool_function):
            return await tool_function(**filtered_args)
        loop = asyncio.get_running_loop()
        func_call = partial(tool_function, **filtered_args)
        return await loop.run_in_executor(None, func_call)
    except Exception as e:
        return f"Error during execution of tool '{tool_name}': {e}"


def message_to_dict(message):
    if isinstance(message, dict):
        return message
    return message.model_dump(exclude_none=True)


def generate_schema_from_function(func: callable, tool_name: str) -> Dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return None

    description = inspect.getdoc(func) or "No description available for this auto-generated tool."
    properties = {}
    required = []
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        List: "array",
        Dict: "object",
    }

    for param in signature.parameters.values():
        param_name = param.name
        param_type = "string"
        if param.annotation is not inspect.Parameter.empty and param.annotation in type_mapping:
            param_type = type_mapping[param.annotation]
        properties[param_name] = {"type": param_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _callable_tool_schema(tools_dir: str, tool: str):
    abs_tools_dir = os.path.abspath(tools_dir)
    module_name = tool
    file_path = os.path.join(abs_tools_dir, f"{module_name}.py")
    package_name = os.path.basename(abs_tools_dir)

    if abs_tools_dir not in sys.path:
        parent_dir = os.path.dirname(abs_tools_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        sys.path.insert(0, abs_tools_dir)
    try:
        if not os.path.exists(os.path.join(abs_tools_dir, "__init__.py")):
            with open(os.path.join(abs_tools_dir, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("# Auto-generated package initializer\n")
        __import__(package_name)
    except Exception as e:
        print(f"Warning: Failed to initialize package '{package_name}': {e}")

    module = None
    full_module_name = f"{package_name}.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        print(f"Info: Could not load '{full_module_name}' as package member ({e}), trying file load.")
        if os.path.exists(file_path):
            try:
                spec = importlib.util.spec_from_file_location(full_module_name, file_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[full_module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e_file:
                print(f"Error: Failed to load module '{module_name}' from '{file_path}': {e_file}")
                return {}, None

    if module is None and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            fixed_source = source.replace("false", "False").replace("true", "True")
            try:
                ast.parse(fixed_source)
            except SyntaxError as e:
                print(f"Warning: Syntax error in '{file_path}': {e}")
                fixed_source = (
                    fixed_source.replace(""", '"')
                    .replace(""", '"')
                    .replace("'", "'")
                    .replace("'", "'")
                )
            spec = importlib.util.spec_from_loader(full_module_name, loader=None, origin=file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_module_name] = module
            exec(fixed_source, module.__dict__)
        except Exception as e_file:
            print(f"Error: Failed to load module '{module_name}' from '{file_path}': {e_file}")
            traceback.print_exc()
            return {}, None

    if module is None:
        print(f"Error: Tool module '{tool}' not found or could not be loaded.")
        return {}, None

    schema = {}
    func_obj = None

    if hasattr(module, "TOOL_SCHEMA"):
        schema_str = getattr(module, "TOOL_SCHEMA")
        if isinstance(schema_str, str):
            try:
                schema = json.loads(schema_str)
            except json.JSONDecodeError:
                try:
                    schema = json.loads(
                        schema_str.replace("false", "false")
                        .replace("true", "true")
                        .replace("'", '"')
                    )
                except Exception:
                    print("Warning: Failed to parse TOOL_SCHEMA")
                    schema = {}
        elif isinstance(schema_str, dict):
            schema = schema_str
        elif isinstance(schema_str, list):
            schema = schema_str[0]

    if isinstance(schema, list):
        schema = schema[0]
    tool_name_from_schema = schema.get("function", {}).get("name")

    if (
        tool_name_from_schema
        and hasattr(module, tool_name_from_schema)
        and callable(getattr(module, tool_name_from_schema))
    ):
        func_obj = getattr(module, tool_name_from_schema)
        tool_name = tool_name_from_schema
    elif hasattr(module, tool) and callable(getattr(module, tool)):
        func_obj = getattr(module, tool)
        tool_name = tool
    else:
        local_functions = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, types.FunctionType)
                and not name.startswith("_")
                and getattr(obj, "__module__", None) == module.__name__
            ):
                local_functions.append((name, obj))
        if len(local_functions) == 1:
            tool_name, func_obj = local_functions[0]
        elif len(local_functions) > 1:
            func_names = [name for name, _ in local_functions]
            print(
                f"Warning: Multiple functions ({func_names}) found in '{tool}'. "
                "Could not determine a unique entry point."
            )

    if func_obj is None:
        print(f"Error: No suitable callable function found in '{tool}'.")
        return {}, None

    if not schema.get("function"):
        schema = generate_schema_from_function(func_obj, tool_name)
    else:
        schema["function"]["name"] = tool_name

    return schema, func_obj


async def build_tool_suite(tools_dir: str = "tools", tools_list: list = None) -> tuple[list, dict]:
    schemas = []
    callable_tools = {}
    loaded_module_names = set()

    if not os.path.isdir(tools_dir) or not tools_list:
        return [], {}
    try:
        for tool in tools_list:
            if tool.endswith(".py"):
                tool = tool[:-3]
            schema, callable_tool = _callable_tool_schema(tools_dir, tool)
            if tool in sys.modules:
                loaded_module_names.add(tool)
            if callable_tool:
                schemas.append(schema)
                callable_tools[tool] = callable_tool
            else:
                print(f"Warning: Failed to build callable tool '{tool}' from '{tools_dir}'.")
    finally:
        for name in loaded_module_names:
            if name in sys.modules:
                del sys.modules[name]

    return schemas, callable_tools
