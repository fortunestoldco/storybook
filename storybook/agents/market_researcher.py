from __future__ import annotations
from typing import Dict, List, Any, Optional
# Standard library imports
from typing import Dict, List, Any, Optional
import logging
import jsonain_core.prompts import ChatPromptTemplate
import rechain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthroughr
# Third-party importsocuments import DocumentPassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParseravilySearchAPIWrapper
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import DocumententStore
from storybook.config import create_llm, get_llm, TAVILY_API_KEY
# Local importsg.getLogger(__name__)ort DocumentStore
from storybook.agents.base import BaseAgentResearchTools
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore
    """Agent responsible for enhancing dialogue in the manuscript."""
logger = logging.getLogger(__name__)# Add inheritance
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):""
        """Initialize with optional LLM configuration."""
class LanguagePolisher(BaseAgent):g)ict[str, Any]] = None):
    """Agent responsible for polishing language and style."""alize with optional LLM configuration."""
llm_config)  # Add super call
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)y_api_key=TAVILY_API_KEY)
        self.document_store = DocumentStore()
        target_audience: Optional[Dict[str, Any]] = None,
    def polish_language(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None, research_insights: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Polish language and style in the manuscript."""
        try:t[str, Any]:
            # Update LLM if new config providedscript."""
            if llm_config:document_store.get_manuscript(manuscript_id)tools.get_research_tool(),
                self.llm = create_llm(llm_config)
            return {"error": f"Manuscript {manuscript_id} not found"}        ]
            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:tionsct[str, Any]] = None) -> Dict[str, Any]:
                return {"error": f"Manuscript {manuscript_id} not found"}ontent"])        """Infer the genre and target audience of the manuscript."""
t_store.get_manuscript(manuscript_id)
            # Add audience context if available
            audience_context = ""und")
            if target_audience:: manuscript_id,
                audience_context = f"""ant dialogue sections found to enhance.",
                Target Audience: {target_audience.get('demographic', 'General readers')}
                Reading Level: {target_audience.get('reading_level', 'Standard')}
                Style Preferences: {target_audience.get('style_preferences', 'Not specified')}
                """alysis = self._analyze_dialogue_quality(
            dialogue_sections, characters, target_audience Take beginning, middle samples for analysis
            # Create prompt for language polishingnt) // 3)
            prompt = ChatPromptTemplate.from_template(
                """ improvement strategies
                You are an expert Language Polisher. Review and enhance the following manuscript text,
                focusing on clarity, style, and engagement. research_insights
        )        # Combine samples
                {audience_context}middle}"
        # Update the manuscript with enhanced dialogue
                Original Text:f._enhance_dialogue_sections(audience inference
                {manuscript_text},plate.from_template(
            dialogue_sections,
                Please polish the language focusing on:ying the genre, themes, and target audience of manuscripts.
                1. Clarity and readability
                2. Sentence structure and flowing manuscript sample, determine:
                3. Word choice and vocabulary
                4. Style consistency
                5. Grammar and punctuationiptnder if specifically targeted, interests)
                6. Voice and tonepply_dialogue_updates(
            manuscript["content"], updated_sections. Similar published books this manuscript reminds you of
                Provide:potential market positioning
                1. The polished text
                2. A summary of changes made
                3. Style recommendationscript(
                """ipt_id, {"content": updated_content}
            )    Format your response as a detailed analysis with specific sections for each element.
ecise as possible about the target demographic, as this will guide market research.
            # Create the chain
            chain = (pt_id": manuscript_id,
                {age": f"Enhanced {len(updated_sections)} dialogue sections based on character profiles and target audience.",
                    "manuscript_text": lambda _: manuscript["content"],
                    "audience_context": lambda _: audience_context, (
                }    {"manuscript_sample": lambda _: sample}
                | prompt
                | self.llmsections(self, content: str) -> List[Dict[str, Any]]:
                | StrOutputParser()gnificant dialogue."""
            )it the content into paragraphs
        paragraphs = content.split("\n\n")
            # Run the chain
            result = chain.invoke("Polish manuscript language")
        current_section = {"start": 0, "end": 0, "content": "", "dialogue_count": 0}
            # Update manuscript with polished content
            self.document_store.update_manuscript("]
                manuscript_id,
                {"content": result}
            )gue_count = 0            "key themes",

            return {in enumerate(paragraphs):
                "manuscript_id": manuscript_id,gue
                "message": "Language polishing complete",
                "polished_content": result: analysis}
            }   if marker in para:
                    has_dialogue = True
        except Exception as e:ount += (
            logger.error(f"Error in polish_language: {str(e)}")
            return self.handle_error(e)alogue turnsn}s?:?(.+?)(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL
                    break            )
    def _analyze_language_style(
        self, content: str, target_audience: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:gue and not in_section:er().replace(" ", "_")
        """Analyze the language and writing style of the manuscript.""" 
        # Sample representative sections
        sample_size = min(5000, len(content) // 4)/ 2        return result
        beginning = content[:sample_size]
        middle_start = max(0, (len(content) // 2) - (sample_size // 2))ogue, end it, themes: List[str]) -> Dict[str, Any]:
        middle = content[middle_start : middle_start + sample_size]enre and themes."""
        end_start = max(0, len(content) - sample_size)
        end = content[end_start:] 3):
                    if i + j < len(paragraphs) and not any(
        # Combine samples in paragraphs[i + j] for m in dialogue_markers{genre} books with themes of {', '.join(themes[:3])}",
        sample = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"
                        no_dialogue_streak += 1            f"reader demographics for {genre} fiction",
        # Add audience context if available
        audience_context = ""e_streak >= 3:
        if target_audience:he section
            audience_context = f""" = "\n\n".join(paragraphs[section_start:i])
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
                        dialogue_sections.append(_tool.invoke(query)
            Consider whether the language style is appropriate for this audience.
            """                 "start": section_start,
                                "end": i - 1,ts": result})
        # Create prompt for style analysis section_content,
        prompt = ChatPromptTemplate.from_template(dialogue_count,, try to crawl one main URL for more in-depth information
            """             }f.research_tools.get_web_crawl_tool()
        You are a Literary Style Analyst specializing in prose evaluation. Analyze the writing style and language
        of this manuscript based on the provided excerpts.
                    in_section = False://\S+", result)
        Manuscript Excerpts:_count = 0            if urls:
        {manuscript_excerpts}
        # Handle any ongoing section at the end
        {audience_context}dialogue_count >= 3:
            section_content = "\n\n".join(paragraphs[section_start:])append(
        Analyze the following aspects of writing style:"Crawling {urls[0]}", "results": crawl_result}
                {
        1. Voice and Point of View: First person, third person, etc. and consistency
        2. Tone: Formal/informal, serious/humorous, emotional/detached
        3. Sentence Structure: Variety, complexity, flow
        4. Vocabulary Level: Simple/complex, specific/general, contemporary/archaicdings
        5. Literary Devices: Use of metaphor, simile, imagery, etc.
        6. Dialogue Style: How characters speak, dialogue tags, etc.
        7. Description Quality: Sensory details, immersion, showing vs. telling
        8. Prose Rhythm: Pacing of sentences, paragraph structure a comprehensive analysis of the market landscape.
        9. Strengths: What aspects of the writing style are most effective
        10. Weaknesses: What aspects could be improved
        11. Audience Appropriateness: How well the style suits the target audience
        dialogue_sections: List[Dict[str, Any]],
        Provide a detailed analysis with specific examples from the text.
        """get_audience: Optional[Dict[str, Any]] = None,g similar titles
        )Dict[str, Any]:
        """Analyze the quality of dialogue in the manuscript."""
        # Create the chaindialogue-rich sections (up to 3)in the market
        chain = (ctions = sorted(
            {ialogue_sections, key=lambda x: x["dialogue_count"], reverse=True
                "manuscript_excerpts": lambda _: sample,
                "audience_context": lambda _: audience_context,on in sample_sections])        """
            }
            | promptaracter voice information
            | self.llmes = []
            | StrOutputParser()ters:
        )   if "voice" in character:s]
                character_voices.append(f"{character['name']}: {character['voice']}")        )
        # Run the chain
        style_analysis = chain.invoke("Analyze writing style")
            "\n".join(character_voices)
        # Parse the analysis into sections
        import re"No character voice information available."   | prompt
        )            | self.llm
        # Extract key sections
        voice_match = re.search(f available
            r"Voice and Point of View:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,ontext = f"""Analyze market")
        )   Target Audience Information:
        tone_match = re.search(get_audience.get('demographic', 'General readers')}
            r"Tone:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALLes', {}).get('language', 'Various preferences')}rn {"raw_research": research_results, "market_analysis": market_analysis}
        )   
        sentence_match = re.search(alogue would resonate with this audience._target_demographic(
            r"Sentence Structure:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,nalysis prompt."""
        )rompt = ChatPromptTemplate.from_template(t the research query for the target demographic
        vocab_match = re.search(
            r"Vocabulary Level:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALLlowing manuscript sections.
        )
        lit_devices_match = re.search(earch_tools.get_research_tool()
            r"Literary Devices:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        dialogue_match = re.search(nal queries to understand the demographic better
            r"Dialogue Style:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )ialogue Samples:t preferences of {demographic_info}",
        description_match = re.search( media usage among {demographic_info}",
            r"Description Quality:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,e for:
            re.DOTALL,y and natural flow
        ). Character voice consistency 
        rhythm_match = re.search(ution clarity
            r"Prose Rhythm:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        ). Purpose (how dialogue advances plot or reveals character)uery": query, "results": result})
        strengths_match = re.search(ticity
            r"Strengths:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALLfor more detailed information
        )
        weaknesses_match = re.search(ailed analysis with specific examples. if urls:
            r"Weaknesses:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )                try:
        audience_match = re.search(rawling
            r"Audience Appropriateness:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,itional_research.append(
            re.DOTALL,
        )       "character_voice_info": lambda _: character_voice_info,rom {urls[0]}",
                "dialogue_samples": lambda _: sample_text,
        # Compile the structured analysisa _: audience_context,           }
        return {
            "full_analysis": style_analysis,
            "voice_and_pov": voice_match.group(1).strip() if voice_match else "",
            "tone": tone_match.group(1).strip() if tone_match else "",
            "sentence_structure": ( analyze demographic research
                sentence_match.group(1).strip() if sentence_match else ""
            ),the chain
            "vocabulary_level": vocab_match.group(1).strip() if vocab_match else "",ts Specialist. Based on the following research about a target demographic,
            "literary_devices": ( help an author connect with this audience.
                lit_devices_match.group(1).strip() if lit_devices_match else ""
            ),nalysis": analysis,
            "dialogue_style": dialogue_match.group(1).strip() if dialogue_match else "",
            "description_quality": (alogue_sections),
                description_match.group(1).strip() if description_match else ""
            ),
            "prose_rhythm": rhythm_match.group(1).strip() if rhythm_match else "",
            "strengths": strengths_match.group(1).strip() if strengths_match else "",
            "weaknesses": weaknesses_match.group(1).strip() if weaknesses_match else "",
            "audience_appropriateness": (
                audience_match.group(1).strip() if audience_match else ""
            ),ch_insights: Optional[Dict[str, Any]] = None,rences (formats, length, pacing, content preferences)
        }Dict[str, Any]:ase channels)
        """Determine strategies for improving dialogue."""
    def _identify_improvement_areas(ent strategiesected in the book
        self,t = ChatPromptTemplate.from_template(uage style and complexity expectations
        content: str,
        style_analysis: Dict[str, Any],Specialist. Based on the following dialogue analysis, 
        target_audience: Optional[Dict[str, Any]] = None,ue in this manuscript.
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:s:
        """Identify specific areas for language improvement.""" 
        # Extract weaknesses from the style analysis
        weaknesses = style_analysis.get("weaknesses", "")
        {character_info}additional_text = "\n\n".join(
        # Add audience context if available
        audience_context = ""     f"Query: {r['query']}\nResults: {r['results']}"
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            - Reading Level: {style_analysis.get('audience_appropriateness', 'Unknown')}
            nhancing subtext and tension
            Focus on improvements that would make the language more appealing to this audience.
            """ngthening dialogue's contribution to plot advancement
        6. Enhancing cultural/period authenticity if relevantsearch,
        # Add research context if availableo the target audience        "additional_research": lambda _: additional_text,
        research_context = ""
        if research_insights:ovide specific examples of before/after dialogue improvements. | prompt
            research_context = f"""
            Market Research Insights:r()
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            epare character info summary
            Consider how language improvements could align with successful books in this market.
            """racter in characters:aphic profile")
            char_summary = f"Name: {character['name']}\n"
        # Create prompt for identifying improvement areas
        prompt = ChatPromptTemplate.from_template(aracter['personality'][:150]}...\n"c_info,
            """"motivations" in character:
        You are a Literary Style Enhancement Specialist. Based on the style analysis, identify specific 
        aspects of the manuscript's language that could be improved.
                char_summary += f"Voice: {character['voice'][:150]}...\n"ch,
        Style Analysis Strengths:(char_summary)            },
        {style_strengths}
        character_info_text = "\n\n".join(character_info)        }
        Style Analysis Weaknesses:
        {style_weaknesses}text if available manuscript_id: str, title: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        audience_context = ""
        {audience_context}:nd target audience
            audience_context = f"""genre_and_audience(manuscript_id, llm_config)
        {research_context}e Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
        Identify 5-7 specific improvement areas, such as:'reading_preferences', {}).get('language', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}        "manuscript_id": manuscript_id,
        1. Sentence variety (e.g., too many sentences with similar structure)
        2. Overused words or phraseslogue especially appealing to this audience.     "message": "Failed to infer genre and target audience.",
        3. Weak descriptions lacking sensory details
        4. Telling instead of showing
        5. Passive voice overusef availableormation
        6. Dialogue that sounds unnaturalrimary_genre", "fiction")
        7. Inconsistent tone:ience.get("target_demographic", "general readers")
        8. Too much/little exposition
            Market Research Insights:
        For each improvement area:('market_analysis_summary', 'No market analysis available')}# Extract themes
        - Clearly describe the issue
        - Explain its impact on reader engagementng the dialogue enhancement strategy.me_matches = re.findall(
        - Provide a specific example from the style analysis\s+\w+){0,2})', themes_text
        - Suggest an approach for addressing it
        # Create the chainatch in theme_matches:
        Focus on improvements that would have the greatest impact on reader experience,
        especially for the target audience.
        """     "dialogue_analysis": lambda _: dialogue_analysis["analysis"],
        )       "character_info": lambda _: character_info_text,
                "audience_context": lambda _: audience_context,
        # Create the chaincontext": lambda _: research_context,   themes = ["adventure", "relationships", "personal growth"]  # Default themes
        chain = (
            { promptResearch similar books
                "style_strengths": lambda _: style_analysis.get(
                    "strengths", "Not specified"
                ),   # Step 3: Analyze target demographic
                "style_weaknesses": lambda _: weaknesses,genre)
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,ies")            # Step 4: Generate comprehensive insights
            }
            | promptategies": strategies}                """
            | self.llmomprehensive market research report for an author.
            | StrOutputParser()ons(
        )elf,t Title: {title}
        content: str,
        # Run the chainns: List[Dict[str, Any]],hic}
        improvement_areas = chain.invoke("Identify improvement areas")
        improvement_strategies: Dict[str, Any],
        # Parse the areas into sectionsstr, Any]] = None,
        import ret[str, Any]]:
        """Enhance each dialogue section based on the improvement strategies."""
        # Try to extract individual improvement areas
        area_blocks = re.split(r"(?:\n|^)(\d+\.\s+)", improvement_areas)
        for character in characters:
        # Process the blocks a comprehensive report covering:
        structured_areas = []cter["name"],
        current_area = {}character.get("voice", ""),ket
                "personality": character.get("personality", ""), positioning
        for i, block in enumerate(area_blocks):otivations", ""),. Target Audience: Detailed profile of the ideal reader
            # If this is a number marker, start a new area
            if re.match(r"\d+\.\s+", block):ame"]] = summary            5. Positioning Recommendations: How to position this book for success
                if current_area and "name" in current_area:o the target audience
                    structured_areas.append(current_area)d approaches likely to reach this audience
                current_area = {"number": block.strip()}
            # If we have a current area and this isn't a number marker, add the contentacter development,
            elif current_area: f"""to the target audience.
                # First block after number is the title/nameic', 'General readers')}
                if "name" not in current_area:'reading_preferences', {}).get('language', 'No specific preferences')}
                    current_area["name"] = block.strip().split("\n")[0].strip() if block.strip() else ""
                    # Extract the rest of the content
                    content_lines = block.strip().split("\n")[1:] if len(block.strip().split("\n")) > 1 else []  # Fixed parentheses
                    current_area["description"] = "\n".join(content_lines).strip()
                else:ions = sorted(
                    # Append to existing descriptionogue_count"], reverse=True      "genre": lambda _: genre,
                    if "description" in current_area:da _: demographic,
                        current_area["description"] += "\n" + block.strip()
                    else:iority_sections:a _: market_research.get("market_analysis", ""),
                        current_area["description"] = block.strip()(
            prompt = ChatPromptTemplate.from_template(     "demographic_profile", ""
        # Add the last area if it exists
        if current_area and "name" in current_area:st. Improve the dialogue in this manuscript section
            structured_areas.append(current_area) strategies.    | prompt
            
        return {"full_analysis": improvement_areas, "areas": structured_areas}
            {original_section})
    def _polish_sections(
        self,haracter Information:
        content: str,r_info}comprehensive_report = chain.invoke("Generate market report")
        improvement_areas: Dict[str, Any],
        style_analysis: Dict[str, Any],database
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> tuple: Research Report for '{title}'",
        """Polish selected sections of the manuscript based on identified improvement areas.""" 
        # Break the content into paragraphs
        paragraphs = content.split("\n\n")ection while maintaining the story and meaning.
            Make each character's voice more distinctive and authentic based on their profile.
        # We'll enhance a selection of paragraphs rather than the entire manuscript
        # Select paragraphs for each improvement areaces the plot or reveals character.
        improvement_types = [
            area.get("name", "") for area in improvement_areas.get("areas", [])ith improved dialogue.
        ]   Preserve the structure and non-dialogue elements exactly as they are.
            """
        # Map improvement types to paragraph selection criteriafor future reference and search
        selection_criteria = {
            "sentence variety": lambda p: len(p.split(".")) > 4 and len(p) > 200,
            "overused words": lambda p: len(p) > 200,_names(
            "weak descriptions": lambda p: "saw" in p.lower()keys())       "type": "market_research",
            or "looked" in p.lower()nuscript_id": manuscript_id,
            or "seemed" in p.lower(),
            "telling": lambda p: "felt" in p.lower()
            or "thought" in p.lower()aphic,
            or "realized" in p.lower(),:,
            "passive voice": lambda p: " was " in p.lower() and " by " in p.lower(),
            "dialogue": lambda p: '"' in p or "'" in p,er_summaries[name]            )
            "tone": lambda p: len(p) > 200,
            "exposition": lambda p: len(p) > 300,_documents_with_embeddings("research", [doc])
        }   char_info = []
            for name, info in relevant_characters.items():
        # Select paragraphs for improvementr: {name}")
        paragraphs_to_improve = []"Voice: {info['voice'][:200]}")
                char_info.append(f"Personality: {info['personality'][:200]}")
        for i, para in enumerate(paragraphs):s: {info['motivations'][:200]}")d comprehensive market research.",
            # Skip very short paragraphs            "research_insights": {
            if len(para) < 50:
                continuefo_text = (hic": demographic,
                "\n".join(char_info)mes": themes,
            # Check against our criteriaysis", "")[
            for imp_type, criterion in selection_criteria.items():"           :500
                if any(          ]
                    imp_type.lower() in area.lower() for area in improvement_types
                ) and criterion(para):nsive_report": comprehensive_report,
                    paragraphs_to_improve.append((i, para, imp_type))
                    break
                    "original_section": lambda _: section["content"],
        # Limit the number of paragraphs to improveracter_info_text,""),
        max_improvements = min(10, len(paragraphs_to_improve))ement_strategies[nces": self._extract_preferences(
        # Prioritize by spreading throughout the manuscriptget("demographic_profile", "")
        step = (    ],
            len(paragraphs_to_improve) // max_improvements_context,   "content_expectations": self._extract_expectations(
            if len(paragraphs_to_improve) > max_improvementsle", "")
            else 1prompt
        )       | self.llm
        selected_paragraphs = [er()   },
            paragraphs_to_improve[i] for i in range(0, len(paragraphs_to_improve), step)
        ][:max_improvements]
            # Run the chain
        # Add audience context if available("Enhance dialogue")            return self.handle_error(e)
        audience_context = ""
        if target_audience:ons.append(Any]:
            audience_context = f"""n, "enhanced": enhanced_section}tract reading preferences from the demographic profile."""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'No specific preferences')}
            rn processed_sections        # Look for reading preferences
            Ensure the enhancements will appeal to this audience.
            """dialogue_updates(| re.DOTALL
        self, original_content: str, updated_sections: List[Dict[str, Any]]
        # Improve each selected paragraph
        improved_sections = []tes to the original content."""
        # Split the content into paragraphs.strip()
        for idx, para, imp_type in selected_paragraphs:
            # Create prompt for polishing this paragraph
            prompt = ChatPromptTemplate.from_template(le(
                """ices = set()            r"language style:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
            You are a Literary Style Enhancer. Improve the following paragraph from a manuscript,
            focusing particularly on {improvement_type}.
            update in updated_sections:
            Original Paragraph:original"]ch.group(1).strip()
            {original_paragraph}["start"]
            end_idx = original["end"]        # Look for character preferences
            Writing Style Notes:
            Voice: {voice}aragraphs as updated|\Z)", re.IGNORECASE | re.DOTALL
            Tone: {tone}ge(start_idx, end_idx + 1):
            Vocabulary Level: {vocabulary}    match = character_pattern.search(profile_text)
            
            {audience_context}ed content into paragraphs
            enhanced_paragraphs = update["enhanced"].split("\n\n")
            Enhance this paragraph to address the {improvement_type} issue while:
            1. Maintaining the same story information and meaningy, use a different approach
            2. Preserving the author's voice and overall tonet_idx + 1)) > 2:str, Any]:
            3. Keeping approximately the same lengthnal section
            4. Making the language more engaging and vividstart_idx : end_idx + 1])
                original_sentences = self._extract_sentences(original_section)
            Return ONLY the improved paragraph, with no additional comments.
            """ # Replace the entire section using sentence matching
            )   new_content = self._replace_section_by_sentences( | re.DOTALL
                    original_content, original_sentences, update["enhanced"]
            # Create the chain = sensitivity_pattern.search(profile_text)
            chain = (
                {f new_content:s"] = match.group(1).strip()
                    "improvement_type": lambda _: imp_type,
                    "original_paragraph": lambda _: para,
                    "voice": lambda _: style_analysis.get(tence matching fails        values_pattern = re.compile(
                        "voice_and_pov", "Not specified"
                    ),the paragraphs
                    "tone": lambda _: style_analysis.get("tone", "Not specified"),
                    "vocabulary": lambda _: style_analysis.get(
                        "vocabulary_level", "Not specified" start_idxxpectations["values"] = match.group(1).strip()
                    ),
                    "audience_context": lambda _: audience_context,= (
                }nhanced_paragraphs[:replacement_length]mendations_pattern = re.compile(
                | prompt r"recommendations:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
                | self.llm
                | StrOutputParser()into contentn.search(profile_text)
            )n "\n\n".join(paragraphs)        if match:

            # Run the chainnames(self, text: str, known_characters: List[str]) -> List[str]:
            improved_para = chain.invoke(f"Improve paragraph with {imp_type} issue")
        # Check for exact matches of known characters        mentioned_characters = [char for char in known_characters if char.lower() in text.lower()]        return mentioned_characters    def _extract_sentences(self, text: str) -> List[str]:        """Extract individual sentences from text."""        # Simple sentence splitting        sentences = re.split(r"(?<=[.!?])\s+", text)        # Filter out empty strings        return [s for s in sentences if s]    def _replace_section_by_sentences(        self,        original_content: str,        original_sentences: List[str],        enhanced_content: str,    ) -> Optional[str]:        """Replace a section by finding sentence boundaries in the original content."""        if not original_sentences:            return None        # Find the start of the first sentence        first_sentence = re.escape(original_sentences[0])        start_match = re.search(first_sentence, original_content)        if not start_match:            return None        # Find the end of the last sentence        last_sentence = re.escape(original_sentences[-1])        end_match = re.search(last_sentence, original_content)        if not end_match:            return None        # Calculate the replacement indices        start_idx = start_match.start()        end_idx = end_match.end()        # Replace the content        return (            original_content[:start_idx] + enhanced_content + original_content[end_idx:]        )
            # Store the improvement            improved_sections.append(                {                    "original": para,                    "improved": improved_para,                    "improvement_type": imp_type,                }            )            # Update the paragraph in the manuscript            paragraphs[idx] = improved_para        # Recombine the paragraphs        updated_content = "\n\n".join(paragraphs)        return updated_content, improved_sections    def _extract_style_patterns(self, content: str) -> Dict[str, List[str]]:        """Extract common style patterns from the text."""         # Implementation for style pattern extraction        pass    def _apply_style_rules(        self,        content: str,        style_rules: Dict[str, Any]    ) -> str:        """Apply predefined style rules to the text."""         # Implementation for applying style rules        pass    def _analyze_block(self, block: str) -> Dict[str, Any]:        """Analyze a block of text for style and language patterns."""        try:            if len(block.strip().split("\n")) > 1:  # Fixed syntax error here                # Process multi-line block                return {                    "length": len(block),                    "sentences": len(re.findall(r'[.!?]+', block)),                    "paragraphs": len(block.strip().split("\n\n')),                    "patterns": self._extract_patterns(block)                }            return {}        except Exception as e:            logger.error(f"Error analyzing block: {str(e)}")            return {}    def _extract_patterns(self, text: str) -> Dict[str, List[str]]:        """Extract common language patterns from text."""        patterns = {            "repetitive_words": [],            "complex_phrases": [],            "weak_constructions": []        }        try:            # Extract patterns using regex            words = re.findall(r'\b\w+\b', text.lower())            word_freq = {}            for word in words:                word_freq[word] = word_freq.get(word, 0) + 1                        # Find repetitive words            patterns["repetitive_words"] = [                word for word, count in word_freq.items()                if count > 3 and len(word) > 3            ]            return patterns        except Exception as e:            logger.error(f"Error extracting patterns: {str(e)}")            return patterns

class MarketResearcher(BaseAgent):
    """Agent responsible for market research."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.search_tool = TavilySearchAPIWrapper(api_key=TAVILY_API_KEY)