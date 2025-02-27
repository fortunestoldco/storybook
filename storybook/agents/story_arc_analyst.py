from typing import Dict, List, Any, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from storybook.config import get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)

class StoryArcAnalyst:
    """Agent responsible for analyzing and refining story arcs."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)
        self.document_store = DocumentStore()
    
    def refine_story_arcs(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None,
                         research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze and refine the story arcs in the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}
        
        # Analyze story structure
        structure_analysis = self._analyze_story_structure(
            manuscript["content"],
            target_audience
        )
        
        # Analyze character arcs
        character_arcs = self._analyze_character_arcs(
            manuscript_id,
            manuscript["content"],
            target_audience
        )
        
        # Evaluate pacing
        pacing_analysis = self._analyze_pacing(
            manuscript["content"],
            structure_analysis,
            target_audience
        )
        
        # Generate improvement recommendations
        improvement_plan = self._generate_improvement_plan(
            structure_analysis,
            character_arcs,
            pacing_analysis,
            target_audience,
            research_insights
        )
        
        # Apply story arc refinements
        updated_content = self._apply_story_arc_refinements(
            manuscript["content"],
            improvement_plan,
            target_audience
        )
        
        # Store the updated manuscript
        self.document_store.update_manuscript(
            manuscript_id,
            {"content": updated_content}
        )
        
        # Compile complete analysis
        analysis = {
            "structure_analysis": structure_analysis,
            "character_arcs": character_arcs,
            "pacing_analysis": pacing_analysis,
            "improvement_plan": improvement_plan
        }
        
        return {
            "manuscript_id": manuscript_id,
            "message": "Completed story arc analysis and refinement.",
            "analysis": analysis
        }
    
    def _analyze_story_structure(self, content: str, target_audience: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze the story structure and identify key plot points."""
        # Create sampling for analysis
        sample_size = min(8000, len(content) // 3)
        beginning = content[:sample_size]
        middle_start = max(0, (len(content) // 2) - (sample_size // 2))
        middle = content[middle_start:middle_start + sample_size]
        end_start = max(0, len(content) - sample_size)
        end = content[end_start:]
        
        # Combine samples
        sample = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Consider whether the story structure would resonate with this audience.
            """
        
        # Create prompt for structure analysis
        prompt = ChatPromptTemplate.from_template("""
        You are a Story Structure Analyst. Analyze the narrative structure of this manuscript.
        
        Manuscript Sample:
        {manuscript_sample}
        
        {audience_context}
        
        Analyze the following aspects:
        
        1. Overall Structure: Identify the story structure (3-act, 5-act, Hero's Journey, etc.)
        2. Key Plot Points: Identify the major plot points/turning points in the story
        3. Inciting Incident: What event sets the story in motion?
        4. Midpoint: What significant event happens in the middle that shifts the direction?
        5. Climax: What is the culminating moment of tension/conflict?
        6. Resolution: How does the story wrap up?
        7. Narrative Cohesion: How well do the plot elements connect and flow?
        8. Structural Strengths: What works well in the current structure?
        9. Structural Weaknesses: What structural issues might need attention?
        
        Provide a detailed analysis with specific examples from the text.
        """)
        
        # Create the chain
        chain = (
            {
                "manuscript_sample": lambda _: sample,
                "audience_context": lambda _: audience_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        structure_analysis = chain.invoke("Analyze story structure")
        
        # Parse the analysis into sections
        import re
        
        # Extract overall structure
        structure_match = re.search(r'Overall Structure:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        overall_structure = structure_match.group(1).strip() if structure_match else ""
        
        # Extract other key elements
        plot_points_match = re.search(r'Key Plot Points:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        inciting_match = re.search(r'Inciting Incident:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        midpoint_match = re.search(r'Midpoint:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        climax_match = re.search(r'Climax:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        resolution_match = re.search(r'Resolution:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        
        strengths_match = re.search(r'Structural Strengths:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        weaknesses_match = re.search(r'Structural Weaknesses:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', structure_analysis, re.DOTALL)
        
        # Compile the structured analysis
        return {
            "full_analysis": structure_analysis,
            "overall_structure": overall_structure,
            "key_plot_points": plot_points_match.group(1).strip() if plot_points_match else "",
            "inciting_incident": inciting_match.group(1).strip() if inciting_match else "",
            "midpoint": midpoint_match.group(1).strip() if midpoint_match else "",
            "climax": climax_match.group(1).strip() if climax_match else "",
            "resolution": resolution_match.group(1).strip() if resolution_match else "",
            "strengths": strengths_match.group(1).strip() if strengths_match else "",
            "weaknesses": weaknesses_match.group(1).strip() if weaknesses_match else ""
        }
    
    def _analyze_character_arcs(self, manuscript_id: str, content: str, 
                           target_audience: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyze the arcs of main characters throughout the story."""
        # Get character information
        characters = []
        character_docs = self.document_store.db.query_documents("characters", {"manuscript_id": manuscript_id})
        
        # If we have stored character information, use it
        if character_docs:
            characters = [
                {
                    "name": doc.get("character_name", ""),
                    "description": doc.get("physical_description", ""),
                    "personality": doc.get("personality", ""),
                    "motivations": doc.get("motivations", "")
                }
                for doc in character_docs
            ]
        
        # If no stored characters, extract from content
        if not characters:
            # Create a simple prompt to extract character names
            prompt = ChatPromptTemplate.from_template("""
            Identify the main characters in this manuscript based on the sample provided.
            For each character, provide their name only.
            
            Sample:
            {content_sample}
            
            List the names of the 3-5 most important characters.
            """)
            
            # Sample the content
            sample = content[:10000]  # First 10k chars
            
            # Create the chain
            chain = (
                {"content_sample": lambda _: sample}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            character_names = chain.invoke("Extract characters")
            
            # Parse the character names
            import re
            name_matches = re.findall(r'(?:^|\n)\s*-?\s*(\w+(?:\s+\w+){0,2})', character_names)
            characters = [{"name": name.strip()} for name in name_matches if len(name.strip()) > 0]
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Consider whether these character arcs would resonate with the target audience.
            """
        
        # Analyze arc for each character
        character_arcs = []
        for character in characters[:5]:  # Limit to top 5 characters
            # Create the prompt for character arc analysis
            arc_prompt = ChatPromptTemplate.from_template("""
            You are a Character Arc Analyst. Analyze the arc of the following character throughout the manuscript.
            
            Character Name: {character_name}
            Character Description: {character_description}
            Character Personality: {character_personality}
            Character Motivations: {character_motivations}
            
            Manuscript Sample:
            {manuscript_sample}
            
            {audience_context}
            
            Analyze this character's arc by addressing:
            
            1. Starting State: The character's condition/mindset at the beginning
            2. Key Moments: Major moments that change or challenge this character
            3. Internal Journey: How the character grows or changes emotionally/psychologically
            4. External Journey: How the character's circumstances or status changes
            5. Resolution: Where the character ends up at the conclusion
            6. Arc Type: Identify the type of character arc (e.g., positive, negative, flat, circular)
            7. Arc Effectiveness: How compelling and believable is this character's journey?
            8. Improvement Opportunities: How could this character's arc be strengthened?
            
            Provide specific examples from the text where possible.
            """)
            
            # Get a sample that contains mentions of this character
            query = character["name"]
            relevant_parts = self.document_store.get_manuscript_relevant_parts(manuscript_id, query, k=5)
            sample_text = "\n\n".join([doc.page_content for doc in relevant_parts])
            
            # If we don't get enough relevant parts, use general samples
            if len(sample_text) < 1000:
                sample_size = min(5000, len(content) // 3)
                beginning = content[:sample_size]
                middle_start = max(0, (len(content) // 2) - (sample_size // 2))
                middle = content[middle_start:middle_start + sample_size]
                end_start = max(0, len(content) - sample_size)
                end = content[end_start:]
                sample_text = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"
            
            # Create the chain
            arc_chain = (
                {
                    "character_name": lambda _: character["name"],
                    "character_description": lambda _: character.get("description", "Not available"),
                    "character_personality": lambda _: character.get("personality", "Not available"),
                    "character_motivations": lambda _: character.get("motivations", "Not available"),
                    "manuscript_sample": lambda _: sample_text,
                    "audience_context": lambda _: audience_context
                }
                | arc_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            arc_analysis = arc_chain.invoke(f"Analyze {character['name']}'s arc")
            
            # Parse the analysis into sections
            import re
            
            # Extract key sections
            start_match = re.search(r'Starting State:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            key_moments_match = re.search(r'Key Moments:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            internal_match = re.search(r'Internal Journey:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            external_match = re.search(r'External Journey:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            resolution_match = re.search(r'Resolution:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            arc_type_match = re.search(r'Arc Type:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            effectiveness_match = re.search(r'Arc Effectiveness:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            improvement_match = re.search(r'Improvement Opportunities:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', arc_analysis, re.DOTALL)
            
            # Compile the arc analysis
            character_arc = {
                "character_name": character["name"],
                "full_analysis": arc_analysis,
                "starting_state": start_match.group(1).strip() if start_match else "",
                "key_moments": key_moments_match.group(1).strip() if key_moments_match else "",
                "internal_journey": internal_match.group(1).strip() if internal_match else "",
                "external_journey": external_match.group(1).strip() if external_match else "",
                "resolution": resolution_match.group(1).strip() if resolution_match else "",
                "arc_type": arc_type_match.group(1).strip() if arc_type_match else "",
                "effectiveness": effectiveness_match.group(1).strip() if effectiveness_match else "",
                "improvement_opportunities": improvement_match.group(1).strip() if improvement_match else ""
            }
            
            character_arcs.append(character_arc)
        
        return character_arcs
    
    def _analyze_pacing(self, content: str, structure_analysis: Dict[str, Any],
                    target_audience: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze the pacing of the story."""
        # Create sampling for analysis
        # We'll use larger chunks to better analyze pacing
        chunk_size = min(10000, len(content) // 5)
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            chunks.append(chunk)
        
        # Limit to 5 chunks
        chunks = chunks[:5]
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Consider whether the pacing would be appropriate for this audience.
            """
        
        # Create prompt for pacing analysis
        prompt = ChatPromptTemplate.from_template("""
        You are a Narrative Pacing Analyst. Analyze the pacing of this manuscript based on the provided samples.
        
        Manuscript Structure Overview:
        Overall Structure: {structure_overview}
        Key Plot Points: {plot_points}
        
        Content Samples:
        {content_samples}
        
        {audience_context}
        
        Analyze the following aspects of pacing:
        
        1. Overall Pacing: Is the story generally fast-paced, medium-paced, or slow-paced?
        2. Pacing Variations: How does the pacing change throughout the story?
        3. High-Intensity Sections: Which parts move quickly or contain high tension?
        4. Low-Intensity Sections: Which parts move slowly or contain lower tension?
        5. Pacing Balance: Is there a good balance between high and low intensity?
        6. Tension Graph: Describe how tension rises and falls throughout the narrative
        7. Pacing Strengths: What works well with the current pacing?
        8. Pacing Issues: What pacing problems might undermine reader engagement?
        
        Provide a detailed analysis with specific examples from the text.
        """)
        
        # Format content samples
        content_samples = []
        for i, chunk in enumerate(chunks):
            content_samples.append(f"SAMPLE {i+1} (approximately {i+1}/{len(chunks)} of the way through):\n{chunk[:1000]}...")
        
        samples_text = "\n\n".join(content_samples)
        
        # Create the chain
        chain = (
            {
                "structure_overview": lambda _: structure_analysis.get("overall_structure", ""),
                "plot_points": lambda _: structure_analysis.get("key_plot_points", ""),
                "content_samples": lambda _: samples_text,
                "audience_context": lambda _: audience_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        pacing_analysis = chain.invoke("Analyze pacing")
        
        # Parse the analysis into sections
        import re
        
        # Extract key sections
        overall_pacing_match = re.search(r'Overall Pacing:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        variations_match = re.search(r'Pacing Variations:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        high_intensity_match = re.search(r'High-Intensity Sections:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        low_intensity_match = re.search(r'Low-Intensity Sections:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        balance_match = re.search(r'Pacing Balance:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        tension_match = re.search(r'Tension Graph:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        strengths_match = re.search(r'Pacing Strengths:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        issues_match = re.search(r'Pacing Issues:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', pacing_analysis, re.DOTALL)
        
        # Compile the pacing analysis
        return {
            "full_analysis": pacing_analysis,
            "overall_pacing": overall_pacing_match.group(1).strip() if overall_pacing_match else "",
            "pacing_variations": variations_match.group(1).strip() if variations_match else "",
            "high_intensity_sections": high_intensity_match.group(1).strip() if high_intensity_match else "",
            "low_intensity_sections": low_intensity_match.group(1).strip() if low_intensity_match else "",
            "pacing_balance": balance_match.group(1).strip() if balance_match else "",
            "tension_graph": tension_match.group(1).strip() if tension_match else "",
            "strengths": strengths_match.group(1).strip() if strengths_match else "",
            "issues": issues_match.group(1).strip() if issues_match else ""
        }
    
    def _generate_improvement_plan(self, structure_analysis: Dict[str, Any], 
                                character_arcs: List[Dict[str, Any]],
                                pacing_analysis: Dict[str, Any],
                                target_audience: Optional[Dict[str, Any]] = None,
                                research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive plan for improving story arcs and structure."""
        # Format the analyses
        structure_summary = f"""
        Overall Structure: {structure_analysis.get('overall_structure', '')}
        Key Plot Points: {structure_analysis.get('key_plot_points', '')}
        Inciting Incident: {structure_analysis.get('inciting_incident', '')}
        Midpoint: {structure_analysis.get('midpoint', '')}
        Climax: {structure_analysis.get('climax', '')}
        Resolution: {structure_analysis.get('resolution', '')}
        Structural Weaknesses: {structure_analysis.get('weaknesses', '')}
        """
        
        # Summarize character arcs
        character_summary = []
        for arc in character_arcs:
            summary = f"Character: {arc.get('character_name', '')}\n"
            summary += f"Arc Type: {arc.get('arc_type', '')}\n"
            summary += f"Effectiveness: {arc.get('effectiveness', '')}\n"
            summary += f"Improvement Opportunities: {arc.get('improvement_opportunities', '')}\n"
            character_summary.append(summary)
        
        character_summary_text = "\n\n".join(character_summary)
        
        # Summarize pacing
        pacing_summary = f"""
        Overall Pacing: {pacing_analysis.get('overall_pacing', '')}
        Pacing Balance: {pacing_analysis.get('pacing_balance', '')}
        Pacing Issues: {pacing_analysis.get('issues', '')}
        """
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Reading Preferences: {target_audience.get('reading_preferences', {}).get('reading', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Tailor your improvement recommendations to ensure they will resonate with this audience.
            """
        
        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider these market trends when recommending story arc improvements.
            """
        
        # Create prompt for improvement plan
        prompt = ChatPromptTemplate.from_template("""
        You are a Story Arc Improvement Specialist. Create a comprehensive plan to improve the story arcs
        and structure of this manuscript based on the analyses provided.
        
        Structure Analysis:
        {structure_summary}
        
        Character Arc Analysis:
        {character_summary}
        
        Pacing Analysis:
        {pacing_summary}
        
        {audience_context}
        
        {research_context}
        
        Create a detailed improvement plan addressing:
        
        1. Structure Improvements: How to strengthen the overall narrative structure
        2. Plot Enhancement: How to make the plot more compelling and cohesive
        3. Character Arc Refinements: Specific ways to strengthen character journeys
        4. Pacing Adjustments: How to improve rhythm and tension
        5. Key Scene Additions or Modifications: Specific scenes to add, remove, or change
        6. Thematic Reinforcement: How to better communicate core themes
        7. Target Audience Appeal: How to make the story more appealing to the target readers
        
        For each recommendation, provide:
        - A clear explanation of the issue
        - A specific suggestion for improvement
        - An example of how to implement the change
        
        Prioritize improvements that will have the greatest impact on reader engagement.
        """)
        
        # Create the chain
        chain = (
            {
                "structure_summary": lambda _: structure_summary,
                "character_summary": lambda _: character_summary_text,
                "pacing_summary": lambda _: pacing_summary,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        improvement_plan = chain.invoke("Generate improvement plan")
        
        # Parse the plan into sections
        import re
        
        # Extract key sections
        structure_improvements_match = re.search(r'Structure Improvements:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        plot_match = re.search(r'Plot Enhancement:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        character_match = re.search(r'Character Arc Refinements:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        pacing_match = re.search(r'Pacing Adjustments:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        scenes_match = re.search(r'Key Scene Additions or Modifications:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        themes_match = re.search(r'Thematic Reinforcement:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        audience_match = re.search(r'Target Audience Appeal:?\s*(.*?)(?=\n\n|\n\d\.|\Z)', improvement_plan, re.DOTALL)
        
        # Compile the improvement plan
        return {
            "full_plan": improvement_plan,
            "structure_improvements": structure_improvements_match.group(1).strip() if structure_improvements_match else "",
            "plot_enhancement": plot_match.group(1).strip() if plot_match else "",
            "character_arc_refinements": character_match.group(1).strip() if character_match else "",
            "pacing_adjustments": pacing_match.group(1).strip() if pacing_match else "",
            "key_scene_changes": scenes_match.group(1).strip() if scenes_match else "",
            "thematic_reinforcement": themes_match.group(1).strip() if themes_match else "",
            "target_audience_appeal": audience_match.group(1).strip() if audience_match else ""
        }
    
    def _apply_story_arc_refinements(self, content: str, improvement_plan: Dict[str, Any],
                                 target_audience: Optional[Dict[str, Any]] = None) -> str:
        """Apply story arc refinements to the manuscript."""
        # Extract key scene changes from the improvement plan
        key_scene_changes = improvement_plan.get("key_scene_changes", "")
        
        # If no specific scene changes, return content unchanged
        if not key_scene_changes:
            return content
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Ensure the refinements will appeal to this audience.
            """
        
        # Create prompt for applying refinements
        prompt = ChatPromptTemplate.from_template("""
        You are a Story Arc Refinement Specialist. Apply key improvements to enhance the story arcs in this manuscript.
        
        Key Scene Recommendations:
        {key_scene_changes}
        
        Plot Enhancement Recommendations:
        {plot_enhancement}
        
        {audience_context}
        
        Create 2-3 new or modified scenes (1-3 paragraphs each) that could be added to the manuscript to strengthen the story arcs.
        Each scene should:
        1. Address a specific issue identified in the recommendations
        2. Enhance character development, plot progression, or thematic depth
        3. Match the style and voice of the existing manuscript
        
        Format each scene as a separate, complete section that could be inserted into the manuscript.
        Indicate where in the narrative flow each scene would be placed (beginning, middle, or end).
        """)
        
        # Create the chain
        chain = (
            {
                "key_scene_changes": lambda _: key_scene_changes,
                "plot_enhancement": lambda _: improvement_plan.get("plot_enhancement", ""),
                "audience_context": lambda _: audience_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        new_scenes = chain.invoke("Generate new scenes")
        
        # Parse the scenes
        import re
        
        # Try to identify scene sections
        scene_blocks = re.split(r'(?:Scene\s+\d+:|New Scene\s+\d+:)', new_scenes)
        if scene_blocks and not scene_blocks[0].strip():
            scene_blocks = scene_blocks[1:]
        
        if not scene_blocks:
            return content
        
        # Process each scene
        updated_content = content
        
        for scene_block in scene_blocks:
            # Try to determine where to place this scene
            placement = "middle"  # Default
            if re.search(r'beginning|start|open|first', scene_block, re.IGNORECASE):
                placement = "beginning"
            elif re.search(r'end|final|conclusion|climax', scene_block, re.IGNORECASE):
                placement = "end"
            
            # Extract the scene content
            # Remove any placement instructions or headers
            scene_content = re.sub(r'^.*?(Placement|Location|Insert).*?\n', '', scene_block, flags=re.IGNORECASE)
            scene_content = scene_content.strip()
            
            # Find an insertion point based on placement
            if placement == "beginning":
                # Insert after the first few paragraphs
                paragraphs = updated_content.split("\n\n")
                insert_idx = min(5, len(paragraphs) // 10)  # After about 10% of paragraphs
                paragraphs.insert(insert_idx, scene_content)
                updated_content = "\n\n".join(paragraphs)
                
            elif placement == "end":
                # Insert before the last few paragraphs
                paragraphs = updated_content.split("\n\n")
                insert_idx = max(len(paragraphs) - 5, int(len(paragraphs) * 0.9))  # Before last 10% of paragraphs
                paragraphs.insert(insert_idx, scene_content)
                updated_content = "\n\n".join(paragraphs)
                
            else:  # middle
                # Insert near the middle
                mid_point = len(updated_content) // 2
                # Find the nearest paragraph break
                before_break = updated_content.rfind("\n\n", 0, mid_point)
                after_break = updated_content.find("\n\n", mid_point)
                
                if before_break != -1:
                    # Insert after this paragraph
                    updated_content = (
                        updated_content[:before_break] + 
                        "\n\n" + scene_content + 
                        "\n\n" + updated_content[before_break:]
                    )
                elif after_break != -1:
                    # Insert before this paragraph
                    updated_content = (
                        updated_content[:after_break] + 
                        "\n\n" + scene_content + 
                        "\n\n" + updated_content[after_break:]
                    )
        
        return updated_content
