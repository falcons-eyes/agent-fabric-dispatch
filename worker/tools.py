"""Tool set v1 — refactor_bulk, summarize_files, classify.

Each tool defines a prompt template and MCP schema.
Inference is delegated to the caller (ollama_generate / vllm_generate).
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    prompt_template: str

    def to_mcp_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    def build_prompt(self, args: dict) -> str:
        return self.prompt_template.format(**args)


TOOLS: dict[str, Tool] = {
    "refactor_bulk": Tool(
        name="refactor_bulk",
        description="Refactor code according to specified instructions. Handles bulk refactoring tasks like renaming, restructuring, and pattern changes.",
        parameters={
            "code": {
                "type": "string",
                "description": "The source code to refactor",
            },
            "instructions": {
                "type": "string",
                "description": "Refactoring instructions (e.g., 'rename function foo to bar', 'extract method', 'convert to async')",
            },
            "language": {
                "type": "string",
                "description": "Programming language of the code",
            },
        },
        prompt_template=(
            "You are a code refactoring assistant. Refactor the following {language} code "
            "according to the instructions. Return ONLY the refactored code, no explanations.\n\n"
            "Instructions: {instructions}\n\n"
            "Code:\n```{language}\n{code}\n```\n\n"
            "Refactored code:"
        ),
    ),

    "summarize_files": Tool(
        name="summarize_files",
        description="Summarize the contents and purpose of source code files. Produces concise descriptions of what the code does.",
        parameters={
            "code": {
                "type": "string",
                "description": "The source code to summarize",
            },
            "file_path": {
                "type": "string",
                "description": "The file path (for context)",
            },
        },
        prompt_template=(
            "Summarize the following source file concisely. Include:\n"
            "1. Purpose of the file\n"
            "2. Key functions/classes and what they do\n"
            "3. Dependencies and imports\n"
            "4. Any notable patterns or concerns\n\n"
            "File: {file_path}\n\n"
            "```\n{code}\n```\n\n"
            "Summary:"
        ),
    ),

    "classify": Tool(
        name="classify",
        description="Classify code or text into categories. Useful for sorting, tagging, and organizing codebases.",
        parameters={
            "content": {
                "type": "string",
                "description": "The content to classify",
            },
            "categories": {
                "type": "string",
                "description": "Comma-separated list of possible categories",
            },
        },
        prompt_template=(
            "Classify the following content into one of these categories: {categories}\n\n"
            "Content:\n{content}\n\n"
            "Respond with ONLY the category name, nothing else.\n\n"
            "Category:"
        ),
    ),
}


def execute_tool(
    tool_name: str,
    args: dict,
    model: str,
    generate_fn: Callable[[str, str], str],
) -> str:
    """Execute a tool with the given arguments and inference function.

    Args:
        tool_name: Name of the tool to execute.
        args: Tool arguments.
        model: Model identifier for inference.
        generate_fn: Callable(model, prompt) -> str for inference.

    Returns:
        The tool's output text.
    """
    tool = TOOLS.get(tool_name)
    if not tool:
        return f"Error: unknown tool '{tool_name}'"

    try:
        prompt = tool.build_prompt(args)
    except KeyError as e:
        return f"Error: missing required argument {e}"

    return generate_fn(model, prompt)
