import json
from typing import Dict, Callable, Any, List
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        
    def register_tool(self, name: str, description: str, 
                     parameters: Dict[str, Any], function: Callable):
        """Register a new tool"""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        
    def get_tool_definitions(self) -> str:
        """Get tool definitions in format for LLM"""
        tools_desc = []
        for tool in self.tools.values():
            tools_desc.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        return json.dumps(tools_desc, indent=2)
        
    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments"""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
            
        tool = self.tools[name]
        return tool.function(**args)
        
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())