import os
import re
import json
from pathlib import Path
import argparse
from collections import defaultdict
import glob

def parse_python_file(file_path):
    """Parse a Python file to extract agent and tool information with thorough pattern matching."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        # Extract agent name if it's an agent file (more comprehensive patterns)
        agent_name = None
        agent_patterns = [
            # Standard class definition
            re.compile(r'class\s+(\w+)(?:\([\w\s,]+\))?\s*:.*?Agent', re.DOTALL),
            # Agent class with inheritance
            re.compile(r'class\s+(\w+)\s*\(\s*.*?Agent\s*[,)]', re.DOTALL),
            # Agent typed variable
            re.compile(r'(\w+)\s*[:=]\s*[\w.]*Agent\(', re.DOTALL)
        ]
        
        for pattern in agent_patterns:
            agent_match = pattern.search(content)
            if agent_match:
                agent_name = agent_match.group(1)
                break
        
        # Extract tool classes defined in the file (more comprehensive)
        tools_defined = []
        tool_class_patterns = [
            # Standard tool class definition
            re.compile(r'class\s+(\w+)(?:\([\w\s.,]+\))?\s*:.*?Tool', re.DOTALL),
            # Tool with inheritance
            re.compile(r'class\s+(\w+)\s*\(\s*.*?Tool\s*[,)]', re.DOTALL)
        ]
        
        for pattern in tool_class_patterns:
            for match in pattern.finditer(content):
                tool_name = match.group(1)
                if tool_name not in tools_defined:
                    tools_defined.append(tool_name)
        
        # Extract tools referenced in the file (more comprehensive)
        tools_referenced = []
        
        # Look for tool instantiations
        tool_instantiation_pattern = re.compile(r'(\w+)\s*\(\s*[^)]*?\s*\)\s*(?:#.*?Tool|as\s+tool)', re.DOTALL)
        for match in tool_instantiation_pattern.finditer(content):
            tool_ref = match.group(1)
            if tool_ref not in tools_referenced:
                tools_referenced.append(tool_ref)
        
        # Look for tool references in tool lists or initialization
        tools_list_patterns = [
            # Standard tools list assignment
            re.compile(r'tools\s*=\s*\[(.*?)\]', re.DOTALL),
            # Tools append or extend
            re.compile(r'tools\s*\.\s*(?:append|extend)\s*\(\s*(.*?)\s*\)', re.DOTALL),
            # Tools initialization in constructor
            re.compile(r'self\s*\.\s*tools\s*=\s*\[(.*?)\]', re.DOTALL)
        ]
        
        for pattern in tools_list_patterns:
            for match in pattern.finditer(content):
                tools_str = match.group(1)
                # Extract tool class names
                tool_refs = re.findall(r'(\w+)\s*\(', tools_str)
                for tool_ref in tool_refs:
                    if tool_ref not in tools_referenced:
                        tools_referenced.append(tool_ref)
        
        # Check for imports of tools
        import_pattern = re.compile(r'from\s+[\w.]+\s+import\s+(.*?)(?:$|#|\n)', re.MULTILINE)
        for match in import_pattern.finditer(content):
            imports = match.group(1)
            # Look for Tool suffix in imports
            for imported in re.finditer(r'(\w+Tool)\b', imports):
                tool_ref = imported.group(1)
                if tool_ref not in tools_referenced:
                    tools_referenced.append(tool_ref)
        
        return agent_name, tools_defined, tools_referenced
    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None, [], []

def scan_directory(src_dir):
    """Recursively scan a directory for Python files and extract agent/tool information."""
    agents = {}
    all_tools = set()
    agent_files = {}
    
    # Walk through all Python files in the directory recursively
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, src_dir)
                
                agent_name, tools_defined, tools_referenced = parse_python_file(file_path)
                
                # Add defined tools to the set of all tools
                all_tools.update(tools_defined)
                
                # If this is an agent file, store its information
                if agent_name:
                    agents[agent_name] = {
                        "tools_referenced": tools_referenced,
                        "file_path": relative_path
                    }
                    agent_files[agent_name] = file_path
                
                # Store tools defined in this file
                for tool in tools_defined:
                    all_tools.add(tool)
    
    # Second pass to find more tool references in agent files
    for agent_name, file_path in agent_files.items():
        if agent_name in agents:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                # Look for any tool in all_tools being referenced in the file
                for tool in all_tools:
                    if re.search(r'\b' + re.escape(tool) + r'\b', content) and tool not in agents[agent_name]["tools_referenced"]:
                        agents[agent_name]["tools_referenced"].append(tool)
    
    return agents, all_tools

def generate_report(agents, all_tools):
    """Generate a detailed report of verified and missing tools for each agent."""
    report = []
    
    for agent_name, agent_data in agents.items():
        referenced_tools = agent_data["tools_referenced"]
        file_path = agent_data["file_path"]
        
        verified_tools = [tool for tool in referenced_tools if tool in all_tools]
        missing_tools = [tool for tool in referenced_tools if tool not in all_tools]
        
        agent_report = {
            "agent_name": agent_name,
            "file_path": file_path,
            "tools_verified": verified_tools,
            "tools_defined_but_missing": missing_tools,
            "total_tools_referenced": len(referenced_tools),
            "total_tools_verified": len(verified_tools),
            "total_tools_missing": len(missing_tools)
        }
        
        report.append(agent_report)
    
    return report

def find_agent_folders(src_dir):
    """Find all agent folders in the source directory."""
    agent_folders = []
    
    # Find all folders that start with "agent"
    for root, dirs, _ in os.walk(src_dir):
        for dir_name in dirs:
            if dir_name.startswith("agent"):
                agent_folders.append(os.path.join(root, dir_name))
    
    return agent_folders

def get_implemented_agents(src_dir):
    """Get all implemented agents from agent folders."""
    agent_folders = find_agent_folders(src_dir)
    implemented_agents = set()
    
    for folder in agent_folders:
        for py_file in glob.glob(f"{folder}/**/*.py", recursive=True):
            agent_name, _, _ = parse_python_file(py_file)
            if agent_name:
                implemented_agents.add(agent_name)
    
    return implemented_agents

def parse_graph_file(src_dir):
    """Parse the graph.py file to extract agent names."""
    graph_agents = set()
    graph_file_path = None
    
    # Try to find graph.py file
    for root, _, files in os.walk(src_dir):
        if "graph.py" in files:
            graph_file_path = os.path.join(root, "graph.py")
            break
    
    if not graph_file_path:
        print("Error: graph.py file not found.")
        return graph_agents
    
    try:
        with open(graph_file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
            
        # Extract agent names from the graph
        # This is a simple approach, might need refinement based on the actual graph.py structure
        agent_pattern = re.compile(r'[\'"]agent[\'"]:\s*[\'"](\w+)[\'"]', re.IGNORECASE)
        for match in agent_pattern.finditer(content):
            graph_agents.add(match.group(1))
            
        # Try to catch any other agent references
        agent_pattern2 = re.compile(r'Agent\([\'"](\w+)[\'"]', re.IGNORECASE)
        for match in agent_pattern2.finditer(content):
            graph_agents.add(match.group(1))
        
    except Exception as e:
        print(f"Error parsing graph file: {e}")
    
    return graph_agents

def check_graph_inclusion(src_dir):
    """Check for agents in the graph vs. implemented agents."""
    print("\nChecking graph inclusion...")
    
    graph_agents = parse_graph_file(src_dir)
    implemented_agents = get_implemented_agents(src_dir)
    
    # Agents in graph but not implemented
    missing_agents = graph_agents - implemented_agents
    
    # Agents implemented but not in graph
    unused_agents = implemented_agents - graph_agents
    
    print("\n===== Graph Inclusion Report =====\n")
    print(f"Agents in graph: {len(graph_agents)}")
    print(f"Implemented agents: {len(implemented_agents)}")
    
    print("\n1) Agents named in the graph but not implemented:")
    if missing_agents:
        for agent in sorted(missing_agents):
            print(f"  - {agent}")
    else:
        print("  None")
    
    print("\n2) Agents implemented but not present in the graph:")
    if unused_agents:
        for agent in sorted(unused_agents):
            print(f"  - {agent}")
    else:
        print("  None")

def display_menu():
    """Display the menu options."""
    print("\n===== Agent Analysis Tool =====")
    print("1: Agents & Tools Analysis")
    print("2: Graph Inclusion Check")
    print("0: Exit")
    return input("Enter your choice (0-2): ")

def agents_and_tools_analysis(src_dir):
    """Run the agents and tools analysis."""
    print(f"\nScanning directory: {src_dir}")
    print("This may take a while for large codebases...")
    
    agents, all_tools = scan_directory(src_dir)
    
    if not agents:
        print("No agents found in the source directory.")
        return
    
    report = generate_report(agents, all_tools)
    
    # Display the report
    print("\n===== Agent Tool Verification Report =====\n")
    print(f"Total number of agents found: {len(agents)}")
    print(f"Total number of tools defined: {len(all_tools)}\n")
    
    for agent_data in report:
        print(f"Agent: {agent_data['agent_name']} (in {agent_data['file_path']})")
        print(f"Total tools referenced: {agent_data['total_tools_referenced']}")
        print(f"Tools verified: {agent_data['total_tools_verified']}")
        print(f"Tools missing: {agent_data['total_tools_missing']}")
        
        print("\nVerified tools:")
        if agent_data['tools_verified']:
            for tool in sorted(agent_data['tools_verified']):
                print(f"  - {tool}")
        else:
            print("  None")
        
        print("\nTools referenced but not found in codebase:")
        if agent_data['tools_defined_but_missing']:
            for tool in sorted(agent_data['tools_defined_but_missing']):
                print(f"  - {tool}")
        else:
            print("  None")
        
        print("\n" + "-"*50 + "\n")
    
    # Save the report to a JSON file
    timestamp = os.path.basename(os.path.normpath(src_dir))
    report_filename = f'agent_tool_report_{timestamp}.json'
    
    with open(report_filename, 'w') as f:
        json.dump({
            "summary": {
                "total_agents": len(agents),
                "total_tools": len(all_tools),
                "scan_directory": src_dir
            },
            "agents": report,
            "all_tools": list(all_tools)
        }, f, indent=2)
    
    print(f"Report saved to {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='Agent and Tool Analysis')
    parser.add_argument('--dir', dest='src_dir', default="src", 
                        help='Directory to scan (default: src)')
    args = parser.parse_args()
    
    src_dir = args.src_dir
    
    # Check if directory exists
    if not os.path.isdir(src_dir):
        print(f"Error: Directory '{src_dir}' not found.")
        return
    
    while True:
        choice = display_menu()
        
        if choice == '0':
            print("Exiting the program.")
            break
        elif choice == '1':
            agents_and_tools_analysis(src_dir)
        elif choice == '2':
            check_graph_inclusion(src_dir)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()