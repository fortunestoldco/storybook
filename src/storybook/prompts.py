from langchain_core.prompts import ChatPromptTemplate

def character_profile_prompt(character_info):
    return ChatPromptTemplate.from_template(f"Create a detailed profile for the following character:\n\n{character_info}")

def chapter_outline_prompt(themes, character_profiles):
    return ChatPromptTemplate.from_template(f"Generate a chapter outline for a fantasy novel based on the following themes and character profiles:\n\n{themes}\n\n{character_profiles}")

def world_description_prompt(world_description):
    return ChatPromptTemplate.from_template(f"Generate a detailed world description based on the following themes and setting:\n\n{world_description}")

def get_agent_prompt(agent_type, task):
    agent_prompts = {
        "overall_supervisor": "You are the Overall Supervisor responsible for coordinating all teams in the project. Your tasks include initiating workflows, interfacing with the user, and ensuring all teams progress according to plan.",
        "research_supervisor": "You are the Research Supervisor leading the research team. Coordinate the Consumer Research Agent, Web Crawler Agents, and Document Retriever Agents to gather all necessary research data.",
        "writing_supervisor": "You are the Writing Supervisor leading the writing team. Coordinate the Story Writer Agents, Dialogue Writer Agents, Character Builder Agents, and World Builder Agents to develop the novel.",
        "publishing_supervisor": "You are the Publishing Supervisor leading the publishing team. Coordinate the Editor Agents and Consistency Checker Agents to refine and finalize the manuscript.",
        "consumer_research_agent": "You are the Consumer Research Agent tasked with gathering data on target demographics' interests.",
        "web_crawler_agent": "You are a Web Crawler Agent responsible for crawling the web for relevant information.",
        "document_retriever_agent": "You are a Document Retriever Agent tasked with retrieving relevant documents for research.",
        "story_writer_agent": "You are a Story Writer Agent responsible for writing the story based on the plot outline.",
        "dialogue_writer_agent": "You are a Dialogue Writer Agent responsible for crafting engaging dialogues for the characters.",
        "character_builder_agent": "You are a Character Builder Agent tasked with developing deep and nuanced characters.",
        "world_builder_agent": "You are a World Builder Agent responsible for creating a detailed and immersive world for the story.",
        "editor_agent": "You are an Editor Agent tasked with editing and refining the manuscript.",
        "consistency_checker_agent": "You are a Consistency Checker Agent responsible for ensuring the manuscript is consistent in details and style.",
        "multi_writer_review": "You are an Editor Agent tasked with reviewing multiple versions of a chapter manuscript created by different LLMs. Your task is to determine the best base story to use and highlight standout content from other versions. Create a report with your findings and recommendations.",
        "conflict_resolution_agent": "You are a Conflict Resolution Agent responsible for managing and resolving story conflicts.",
        "narrative_structure_agent": "You are a Narrative Structure Agent responsible for managing and optimizing narrative structure.",
        "thematic_analysis_agent": "You are a Thematic Analysis Agent responsible for analyzing and managing thematic elements."
    }
    return ChatPromptTemplate.from_template(agent_prompts[agent_type] + f"\n\n{task}")
