from typing import Dict, List, Callable, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models import Novel, AgentInput, AgentOutput, Character, Setting, Subplot

# Initialize the language model
model = ChatOpenAI(model="gpt-4", temperature=0.7)

def create_agent(system_prompt: str) -> Callable:
    """Create an agent with the given system prompt"""
    def agent_function(input_data: AgentInput, config: RunnableConfig = None) -> AgentOutput:
        novel = input_data.novel
        instructions = input_data.instructions or ""
        focus_areas = input_data.focus_areas or []
        
        # Prepare the content for the agent
        content = f"""
        Title: {novel.title}
        Author: {novel.author}
        
        Instructions: {instructions}
        
        Focus Areas: {', '.join(focus_areas) if focus_areas else 'All areas'}
        
        Current Manuscript:
        {novel.manuscript[:8000]}...
        
        {prepare_context_for_agent(novel, system_prompt)}
        """
        
        # Get the response from the language model
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        response = model.invoke(messages, config=config)
        
        # Process the response and update the novel
        updated_novel = process_agent_response(novel, response.content, system_prompt)
        
        # Create and return the agent output
        return AgentOutput(
            novel=updated_novel,
            notes=extract_notes(response.content),
            changes_made=extract_changes(response.content),
            recommendations=extract_recommendations(response.content)
        )
    
    return agent_function

def prepare_context_for_agent(novel: Novel, agent_type: str) -> str:
    """Prepare the relevant context based on the agent type"""
    context = ""
    
    if "Character" in agent_type:
        context += "\nCurrent Characters:\n"
        for name, character in novel.characters.items():
            context += f"- {name}: {character.backstory[:200]}...\n"
            context += f"  Motivations: {', '.join(character.motivations)}\n"
            context += f"  Arc: {character.arc[:200]}...\n"
    
    elif "Dialogue" in agent_type:
        context += "\nCharacter Voices to Consider:\n"
        for name, character in novel.characters.items():
            context += f"- {name}: Traits: {', '.join(character.traits)}\n"
    
    elif "World-Building" in agent_type:
        context += "\nCurrent Settings:\n"
        for name, setting in novel.settings.items():
            context += f"- {name}: {setting.description[:200]}...\n"
            context += f"  History: {setting.history[:200]}...\n"
    
    elif "Subplot" in agent_type:
        context += "\nCurrent Subplots:\n"
        for subplot in novel.subplots:
            context += f"- {subplot.title}: {subplot.description[:200]}...\n"
            context += f"  Connected Characters: {', '.join(subplot.connected_characters)}\n"
    
    elif "Story Arc" in agent_type:
        context += f"\nMain Plot: {novel.main_plot[:500]}...\n"
        context += "\nSubplots:\n"
        for subplot in novel.subplots:
            context += f"- {subplot.title}: {subplot.description[:200]}...\n"
    
    return context

def process_agent_response(novel: Novel, response: str, agent_type: str) -> Novel:
    """Process the agent's response and update the novel accordingly"""
    # Create a copy of the novel to avoid modifying the original
    updated_novel = Novel(**novel.model_dump())
    
    # Add to revision history
    updated_novel.revision_history.append({
        "stage": updated_novel.current_stage,
        "agent": agent_type,
        "changes": response[:500] + "..." if len(response) > 500 else response
    })
    
    # Process based on agent type
    if "Character Development" in agent_type:
        updated_novel = process_character_development(updated_novel, response)
    elif "Dialogue Enhancer" in agent_type:
        updated_novel.manuscript = process_dialogue_enhancement(updated_novel.manuscript, response)
    elif "World-Building" in agent_type:
        updated_novel = process_world_building(updated_novel, response)
    elif "Subplot Weaver" in agent_type:
        updated_novel = process_subplot_weaving(updated_novel, response)
    elif "Story Arc Analyst" in agent_type:
        updated_novel = process_story_arc_analysis(updated_novel, response)
    elif "Continuity Editor" in agent_type:
        updated_novel.manuscript = process_continuity_editing(updated_novel.manuscript, response)
    elif "Language and Style" in agent_type:
        updated_novel.manuscript = process_language_style(updated_novel.manuscript, response)
    elif "Quality Assurance" in agent_type:
        updated_novel.latest_feedback = response
    
    # Update the current stage
    if "Character Development" in agent_type:
        updated_novel.current_stage = "character_development"
    elif "Dialogue Enhancer" in agent_type:
        updated_novel.current_stage = "dialogue_enhancement"
    elif "World-Building" in agent_type:
        updated_novel.current_stage = "world_building"
    elif "Subplot Weaver" in agent_type:
        updated_novel.current_stage = "subplot_weaving"
    elif "Story Arc Analyst" in agent_type:
        updated_novel.current_stage = "story_arc_analysis"
    elif "Continuity Editor" in agent_type:
        updated_novel.current_stage = "continuity_editing"
    elif "Language and Style" in agent_type:
        updated_novel.current_stage = "language_style"
    elif "Quality Assurance" in agent_type:
        updated_novel.current_stage = "final_review"
    
    return updated_novel

# Helper functions for processing agent responses
def process_character_development(novel: Novel, response: str) -> Novel:
    # Look for character sections in the response
    import re
    
    # Find character sections with pattern: ### Character: Name
    character_sections = re.split(r'###\s+Character:\s+([^\n]+)', response)
    
    if len(character_sections) > 1:
        # First element is preamble
        for i in range(1, len(character_sections), 2):
            if i + 1 < len(character_sections):
                name = character_sections[i].strip()
                details = character_sections[i + 1].strip()
                
                # Parse character details
                backstory_match = re.search(r'Backstory:(.*?)(?=Motivations:|Arc:|Traits:|$)', details, re.DOTALL)
                motivations_match = re.search(r'Motivations:(.*?)(?=Backstory:|Arc:|Traits:|$)', details, re.DOTALL)
                arc_match = re.search(r'Arc:(.*?)(?=Backstory:|Motivations:|Traits:|$)', details, re.DOTALL)
                traits_match = re.search(r'Traits:(.*?)(?=Backstory:|Motivations:|Arc:|$)', details, re.DOTALL)
                
                backstory = backstory_match.group(1).strip() if backstory_match else ""
                motivations = [m.strip() for m in motivations_match.group(1).strip().split(',')] if motivations_match else []
                arc = arc_match.group(1).strip() if arc_match else ""
                traits = [t.strip() for t in traits_match.group(1).strip().split(',')] if traits_match else []
                
                # Create or update character
                if name in novel.characters:
                    novel.characters[name].backstory = backstory or novel.characters[name].backstory
                    novel.characters[name].motivations = motivations or novel.characters[name].motivations
                    novel.characters[name].arc = arc or novel.characters[name].arc
                    novel.characters[name].traits = traits or novel.characters[name].traits
                else:
                    novel.characters[name] = Character(
                        name=name,
                        backstory=backstory,
                        motivations=motivations,
                        arc=arc,
                        traits=traits
                    )
    
    return novel

def process_dialogue_enhancement(manuscript: str, response: str) -> str:
    # Extract the enhanced manuscript if provided in full
    import re
    
    enhanced_manuscript_match = re.search(r'ENHANCED MANUSCRIPT:(.*)', response, re.DOTALL)
    if enhanced_manuscript_match:
        return enhanced_manuscript_match.group(1).strip()
    
    # Look for specific dialogue changes
    dialogue_changes = re.findall(r'ORIGINAL DIALOGUE:\s*"([^"]+)"\s*ENHANCED DIALOGUE:\s*"([^"]+)"', response)
    
    updated_manuscript = manuscript
    for original, enhanced in dialogue_changes:
        updated_manuscript = updated_manuscript.replace(f'"{original}"', f'"{enhanced}"')
    
    return updated_manuscript

def process_world_building(novel: Novel, response: str) -> Novel:
    import re
    
    # Find setting sections with pattern: ### Setting: Name
    setting_sections = re.split(r'###\s+Setting:\s+([^\n]+)', response)
    
    if len(setting_sections) > 1:
        # First element is preamble
        for i in range(1, len(setting_sections), 2):
            if i + 1 < len(setting_sections):
                name = setting_sections[i].strip()
                details = setting_sections[i + 1].strip()
                
                # Parse setting details
                description_match = re.search(r'Description:(.*?)(?=History:|Cultures:|Environment:|$)', details, re.DOTALL)
                history_match = re.search(r'History:(.*?)(?=Description:|Cultures:|Environment:|$)', details, re.DOTALL)
                cultures_match = re.search(r'Cultures:(.*?)(?=Description:|History:|Environment:|$)', details, re.DOTALL)
                environment_match = re.search(r'Environment:(.*?)(?=Description:|History:|Cultures:|$)', details, re.DOTALL)
                
                description = description_match.group(1).strip() if description_match else ""
                history = history_match.group(1).strip() if history_match else ""
                cultures = [c.strip() for c in cultures_match.group(1).strip().split(',')] if cultures_match else []
                environment = environment_match.group(1).strip() if environment_match else ""
                
                # Create or update setting
                if name in novel.settings:
                    novel.settings[name].description = description or novel.settings[name].description
                    novel.settings[name].history = history or novel.settings[name].history
                    novel.settings[name].cultures = cultures or novel.settings[name].cultures
                    novel.settings[name].environment = environment or novel.settings[name].environment
                else:
                    novel.settings[name] = Setting(
                        name=name,
                        description=description,
                        history=history,
                        cultures=cultures,
                        environment=environment
                    )
    
    return novel

def process_subplot_weaving(novel: Novel, response: str) -> Novel:
    import re
    
    # Find subplot sections with pattern: ### Subplot: Title
    subplot_sections = re.split(r'###\s+Subplot:\s+([^\n]+)', response)
    
    if len(subplot_sections) > 1:
        # First element is preamble
        for i in range(1, len(subplot_sections), 2):
            if i + 1 < len(subplot_sections):
                title = subplot_sections[i].strip()
                details = subplot_sections[i + 1].strip()
                
                # Parse subplot details
                description_match = re.search(r'Description:(.*?)(?=Connected Characters:|Resolution:|$)', details, re.DOTALL)
                characters_match = re.search(r'Connected Characters:(.*?)(?=Description:|Resolution:|$)', details, re.DOTALL)
                resolution_match = re.search(r'Resolution:(.*?)(?=Description:|Connected Characters:|$)', details, re.DOTALL)
                
                description = description_match.group(1).strip() if description_match else ""
                characters = [c.strip() for c in characters_match.group(1).strip().split(',')] if characters_match else []
                resolution = resolution_match.group(1).strip() if resolution_match else ""
                
                # Create new subplot (or update existing one by title)
                existing_subplot = next((s for s in novel.subplots if s.title == title), None)
                
                if existing_subplot:
                    existing_subplot.description = description or existing_subplot.description
                    existing_subplot.connected_characters = characters or existing_subplot.connected_characters
                    existing_subplot.resolution = resolution or existing_subplot.resolution
                else:
                    novel.subplots.append(Subplot(
                        title=title,
                        description=description,
                        connected_characters=characters,
                        resolution=resolution
                    ))
    
    # Check for enhanced manuscript
    enhanced_manuscript_match = re.search(r'ENHANCED MANUSCRIPT:(.*)', response, re.DOTALL)
    if enhanced_manuscript_match:
        novel.manuscript = enhanced_manuscript_match.group(1).strip()
    
    return novel

def process_story_arc_analysis(novel: Novel, response: str) -> Novel:
    import re
    
    # Extract main plot if provided
    main_plot_match = re.search(r'MAIN PLOT:(.*?)(?=SUBPLOTS:|$)', response, re.DOTALL)
    if main_plot_match:
        novel.main_plot = main_plot_match.group(1).strip()
    
    # Check for enhanced manuscript
    enhanced_manuscript_match = re.search(r'ENHANCED MANUSCRIPT:(.*)', response, re.DOTALL)
    if enhanced_manuscript_match:
        novel.manuscript = enhanced_manuscript_match.group(1).strip()
    
    # Look for subplot revisions
    subplot_revisions = re.findall(r'SUBPLOT REVISION: ([^\n]+)\n(.*?)(?=SUBPLOT REVISION:|$)', response, re.DOTALL)
    
    for title, details in subplot_revisions:
        title = title.strip()
        details = details.strip()
        
        # Find the subplot to update
        for subplot in novel.subplots:
            if subplot.title.lower() == title.lower():
                # Update the subplot
                description_match = re.search(r'Description:(.*?)(?=Connected Characters:|Resolution:|$)', details, re.DOTALL)
                if description_match:
                    subplot.description = description_match.group(1).strip()
                
                resolution_match = re.search(r'Resolution:(.*?)(?=Description:|Connected Characters:|$)', details, re.DOTALL)
                if resolution_match:
                    subplot.resolution = resolution_match.group(1).strip()
    
    return novel

def process_continuity_editing(manuscript: str, response: str) -> str:
    # Extract the corrected manuscript if provided in full
    import re
    
    corrected_manuscript_match = re.search(r'CORRECTED MANUSCRIPT:(.*)', response, re.DOTALL)
    if corrected_manuscript_match:
        return corrected_manuscript_match.group(1).strip()
    
    # Look for specific corrections
    corrections = re.findall(r'ISSUE:\s*(.*?)\s*CORRECTION:\s*(.*?)(?=ISSUE:|$)', response, re.DOTALL)
    
    updated_manuscript = manuscript
    for issue, correction in corrections:
        issue = issue.strip()
        correction = correction.strip()
        
        if issue in updated_manuscript:
            updated_manuscript = updated_manuscript.replace(issue, correction)
    
    return updated_manuscript

def process_language_style(manuscript: str, response: str) -> str:
    # Extract the polished manuscript if provided in full
    import re
    
    polished_manuscript_match = re.search(r'POLISHED MANUSCRIPT:(.*)', response, re.DOTALL)
    if polished_manuscript_match:
        return polished_manuscript_match.group(1).strip()
    
    return manuscript

def extract_notes(response: str) -> str:
    import re
    
    notes_match = re.search(r'NOTES:(.*?)(?=CHANGES MADE:|RECOMMENDATIONS:|$)', response, re.DOTALL)
    if notes_match:
        return notes_match.group(1).strip()
    return ""

def extract_changes(response: str) -> List[str]:
    import re
    
    changes_match = re.search(r'CHANGES MADE:(.*?)(?=NOTES:|RECOMMENDATIONS:|$)', response, re.DOTALL)
    if changes_match:
        changes_text = changes_match.group(1).strip()
        # Extract bullet points
        changes = re.findall(r'[-*]\s*(.*?)(?=[-*]|$)', changes_text, re.DOTALL)
        return [change.strip() for change in changes if change.strip()]
    return []

def extract_recommendations(response: str) -> List[str]:
    import re
    
    recommendations_match = re.search(r'RECOMMENDATIONS:(.*?)(?=NOTES:|CHANGES MADE:|$)', response, re.DOTALL)
    if recommendations_match:
        recommendations_text = recommendations_match.group(1).strip()
        # Extract bullet points
        recommendations = re.findall(r'[-*]\s*(.*?)(?=[-*]|$)', recommendations_text, re.DOTALL)
        return [recommendation.strip() for recommendation in recommendations if recommendation.strip()]
    return []

# Define the agents with their system prompts
character_development_agent = create_agent("""
You are a Character Development Specialist. Analyze the manuscript's characters and enhance them by adding detailed backstories, clear motivations, and dynamic character arcs that align with the overall plot.

For each major character, provide:
1. A rich backstory that explains their origins and formative experiences
2. Clear motivations that drive their actions throughout the story
3. A character arc showing how they change during the narrative
4. Defining traits that make them unique and memorable

Format your response as follows:
### Character: [Character Name]
Backstory: [Detailed backstory]
Motivations: [List of motivations]
Arc: [Character arc description]
Traits: [List of key traits]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further development]
""")

dialogue_enhancer_agent = create_agent("""
You are a Dialogue Enhancer. Improve the manuscript's dialogue to make it engaging and authentic, ensuring each character's voice is distinct and conversations drive the plot forward.

For each piece of dialogue you enhance:
1. Make it sound natural and true to the character's voice
2. Ensure it reveals character traits, advances the plot, or adds to worldbuilding
3. Eliminate unnecessary dialogue that doesn't serve the story
4. Balance dialogue with action and description

Format your response with:
ENHANCED MANUSCRIPT: [Provide the full manuscript with enhanced dialogue]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further improvements]
""")

worldbuilding_agent = create_agent("""
You are a World-Building Architect. Develop the manuscript's setting into a rich and immersive world, detailing its geography, history, cultures, and societal structures that influence the characters and plot.

For each major setting, provide:
1. A vivid description that brings the location to life
2. Historical context that explains how the setting evolved
3. Cultural elements that influence character behavior and plot
4. Environmental details that contribute to atmosphere and tone

Format your response as follows:
### Setting: [Setting Name]
Description: [Detailed description]
History: [Historical context]
Cultures: [Cultural elements]
Environment: [Environmental details]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further development]
""")

subplot_weaver_agent = create_agent("""
You are a Subplot Weaver. Identify opportunities to introduce subplots that add depth to characters and themes, weaving them seamlessly into the main narrative.

For each subplot you create or enhance:
1. Make it relevant to the main plot or deepen character development
2. Ensure it has its own beginning, middle, and resolution
3. Connect it to appropriate characters
4. Weave it naturally into the main narrative

Format your response as follows:
### Subplot: [Subplot Title]
Description: [Detailed description]
Connected Characters: [List of characters involved]
Resolution: [How the subplot concludes]

ENHANCED MANUSCRIPT: [Provide the full manuscript with subplots integrated]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further development]
""")

story_arc_analyst_agent = create_agent("""
You are a Story Arc Analyst. Evaluate the manuscript's plot structure, ensuring that the main plot and subplots progress logically with impactful climaxes and resolutions.

Analyze and improve:
1. The main plot's structure, pacing, and progression
2. Each subplot's connection to the main narrative
3. The overall dramatic arc (exposition, rising action, climax, falling action, resolution)
4. The effectiveness of plot twists and revelations

Format your response as follows:
MAIN PLOT: [Analysis and improvements for the main plot]

For each subplot that needs revision:
SUBPLOT REVISION: [Subplot Title]
Description: [Revised description]
Resolution: [Revised resolution]

ENHANCED MANUSCRIPT: [Provide the full manuscript with revised plot structure]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further improvements]
""")

continuity_editor_agent = create_agent("""
You are a Continuity Editor. Review the manuscript for consistency in all aspects, correcting any discrepancies in character traits, plot details, and world-building elements.

Look for and fix:
1. Character inconsistencies (traits, behavior, appearance)
2. Timeline issues and chronological errors
3. Setting and world-building contradictions
4. Plot holes and logical inconsistencies

Format your response as follows:
CORRECTED MANUSCRIPT: [Provide the full manuscript with corrections]

For each issue found:
ISSUE: [Description of the continuity issue]
CORRECTION: [How you fixed it]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further improvements]
""")

language_style_polisher_agent = create_agent("""
You are a Language and Style Polisher. Enhance the manuscript's prose by improving sentence structure, word choice, and narrative flow while maintaining the author's voice.

Focus on improving:
1. Sentence variety and rhythm
2. Word choice and precision
3. Show-don't-tell narrative techniques
4. Elimination of cliches and redundancies
5. Balanced use of description, action, and dialogue

Format your response as follows:
POLISHED MANUSCRIPT: [Provide the full manuscript with polished language]

Then include:
NOTES: [Your analysis and approach]
CHANGES MADE: [List of significant changes]
RECOMMENDATIONS: [Suggestions for further improvements]
""")

quality_assurance_agent = create_agent("""
You are a Quality Assurance Reviewer. Conduct a comprehensive review of the manuscript to ensure it meets high standards for publication, focusing on overall quality and reader impact.

Evaluate:
1. Overall narrative coherence and engagement
2. Character development and relatability
3. Setting and world-building effectiveness
4. Plot strength and pacing
5. Thematic depth and resonance
6. Potential reader satisfaction

Format your response with:
FINAL REVIEW: [Comprehensive evaluation of the manuscript]
STRENGTHS: [List of the manuscript's strongest elements]
AREAS FOR IMPROVEMENT: [List of aspects that could still be enhanced]
PUBLICATION READINESS: [Assessment of how ready the manuscript is for publication]

Then include:
NOTES: [Your analysis and approach]
RECOMMENDATIONS: [Final suggestions]
""")
