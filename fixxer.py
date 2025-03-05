import os
import re
import json
from pathlib import Path
from collections import defaultdict

def parse_python_file(file_path):
    """Parse a Python file to extract agent and tool information."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract agent name if it's an agent file
    agent_name = None
    agent_pattern = re.compile(r'class\s+(\w+)(?:\([\w\s,]+\))?\s*:.*?Agent', re.DOTALL)
    agent_match = agent_pattern.search(content)
    if agent_match:
        agent_name = agent_match.group(1)
    
    # Extract tool names defined in the file
    tool_pattern = re.compile(r'class\s+(\w+)(?:\([\w\s,]+\))?\s*:.*?Tool', re.DOTALL)
    tools_defined = [match.group(1) for match in tool_pattern.finditer(content)]
    
    # Extract tools referenced by agents
    tools_referenced = []
    if agent_name:
        # Look for tool references in initialize_tools or similar methods
        tools_ref_pattern = re.compile(r'tools\s*=\s*\[(.*?)\]', re.DOTALL)
        tools_ref_match = tools_ref_pattern.search(content)
        if tools_ref_match:
            tools_str = tools_ref_match.group(1)
            # Extract tool class names from the tools list
            tool_refs = re.findall(r'(\w+)\(', tools_str)
            tools_referenced.extend(tool_refs)
    
    return agent_name, tools_defined, tools_referenced

def scan_directory(src_dir):
    """Recursively scan a directory for Python files and extract agent/tool information."""
    agents = {}
    all_tools = set()
    
    # Walk through all Python files in the src directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                agent_name, tools_defined, tools_referenced = parse_python_file(file_path)
                
                # Add defined tools to the set of all tools
                all_tools.update(tools_defined)
                
                # If this is an agent file, store its information
                if agent_name and tools_referenced:
                    agents[agent_name] = tools_referenced
    
    return agents, all_tools

def generate_report(agents, all_tools):
    """Generate a report of verified and missing tools for each agent."""
    report = []
    
    for agent_name, referenced_tools in agents.items():
        verified_tools = [tool for tool in referenced_tools if tool in all_tools]
        missing_tools = [tool for tool in referenced_tools if tool not in all_tools]
        
        agent_report = {
            "agent_name": agent_name,
            "tools_verified": verified_tools,
            "tools_defined_but_missing": missing_tools
        }
        
        report.append(agent_report)
    
    return report

def main():
    src_dir = "src"  # Adjust this path if needed
    
    # Check if src directory exists
    if not os.path.isdir(src_dir):
        print(f"Error: Directory '{src_dir}' not found.")
        return
    
    print(f"Scanning directory: {src_dir}")
    agents, all_tools = scan_directory(src_dir)
    
    if not agents:
        print("No agents found in the source directory.")
        return
    
    report = generate_report(agents, all_tools)
    
    # Display the report
    print("\n=== Agent Tool Verification Report ===\n")
    for agent_data in report:
        print(f"Agent: {agent_data['agent_name']}")
        
        print("Tools verified:")
        if agent_data['tools_verified']:
            for tool in agent_data['tools_verified']:
                print(f"  - {tool}")
        else:
            print("  None")
        
        print("Tools defined but missing:")
        if agent_data['tools_defined_but_missing']:
            for tool in agent_data['tools_defined_but_missing']:
                print(f"  - {tool}")
        else:
            print("  None")
        
        print()
    
    # Save the report to a JSON file
    with open('agent_tool_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to agent_tool_report.json")

if __name__ == "__main__":
    main()