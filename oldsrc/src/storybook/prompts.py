from langchain_core.prompts import ChatPromptTemplate

def character_profile_prompt(character_info):
    return ChatPromptTemplate.from_template(f"Create a detailed profile for the following character:\n\n{character_info}")

def chapter_outline_prompt(themes, character_profiles):
    return ChatPromptTemplate.from_template(f"Generate a chapter outline for a fantasy novel based on the following themes and character profiles:\n\n{themes}\n\n{character_profiles}")

def world_description_prompt(world_description):
    return ChatPromptTemplate.from_template(f"Generate a detailed world description based on the following themes and setting:\n\n{world_description}")
