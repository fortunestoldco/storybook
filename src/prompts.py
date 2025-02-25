from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

RESEARCH_PROMPT = """You are researcher {researcher_name}.
Based on the information in the story bible: {story_bible}, research the topic: {topic}. Provide a concise and informative report."""

OUTLINE_PROMPT = """Based on the following research reports: {research_reports}, create a detailed story outline in three acts."""

WRITING_PROMPT = """You are writer {writer_name}.
Based on the following outline: {outline}, and using the information in the story bible: {story_bible}, write section {section_id} of the story.  Be creative and engaging.  Keep it concise."""

EDITING_PROMPT = """You are editor {editor_name}.
Based on the following story bible: {story_bible}, edit section {section_id} of the story, which currently reads: {section_content}.  Improve the grammar, style, and consistency.  Keep it concise."""

MARKET_RESEARCH_PROMPT = """You are the Market Researcher. Provide the overall state of the industry and trends on genre, commercial opportunities. What trends on genre/commercial opportunities exist in the novel industry?"""

CONTEXTUAL_RESEARCH_PROMPT = """You are the Contextual Researcher. Based on the user input, research similar books and come up with a top 20 list of similar titles based on thematics and narrative, and order them by popularity and commercial success.  For each of those 20 books - what did the audience like? what did they hate? did it win any awards? what are the overall reviews on amazon? how well did it sell? is it on any bestseller lists? did it get rave reviews or was it critically panned? Find out what the failings of that novel were, and why, and how can we counteract that with our novel."""

CONSUMER_RESEARCH_PROMPT = """You are the Consumer Researcher. Based on the market research report and the contextual research reports, determine the target market for the novel and provide contextual information on that audience. are they ABC1 / C2DE? do they watch TV? spend all day on tiktok? what are their buying habits? do they even really like books? what do they want to buy, what do they want. what dont they want."""

WORLD_BUILDING_PROMPT = """You are the World Building Writer.  Read the synopsis, brainstorm summary, and research report from the story bible. Craft the world in which the novel is set: location, rules, magic system, laws, maps, and the overall universe.  Write a detailed world specification document."""

CHARACTER_DEVELOPMENT_PROMPT = """You are the Character Development Writer.  Read the research, user input, and the world specification.  Write everything we need to know about the characters: their personalities, backstories, story arcs, friends, and overall definitions. Deliver a complete character development document."""

STORY_WRITER_PROMPT = """You are a Story Writer. Based on the story bible write the manuscript chapter in barebones narrative style.  Leave placeholders for the dialogue writer to complete. You will write chapter {chapter_number}."""

DIALOGUE_WRITER_PROMPT = """You are the Dialogue Writer. Using the story bible, write the dialogue for chapter {chapter_number}. Enhance and perfect the dialogue with your expert knowledge. Ensure that there are no plotholes or inconsistencies. Ensure that it relates to the world building document."""

CONTINUITY_CHECKER_PROMPT = """You are a Continuity Specialist. Based on chapters {chapter_1_number} and {chapter_2_number}, ensure that everything that happened in chapter 1 still applies in chapter 2. Check for plot holes, disappearing characters, or characters randomly written in with different names. Provide a report with all findings."""

COHESIVENESS_CHECKER_PROMPT = """You are the Cohesiveness Checker. Based on chapters {chapter_1_number} and {chapter_2_number}, ensure both chapters work together as one body of work and everything is cohesive. Provide a report with all findings."""

EDITORIAL_FEEDBACK_PROMPT = """You are the Editorial agent, who reads the story plan, and the two specifications and gives notes on originality, concept, creativity and marks it. Provide all findings"""

CHAPTER_EDITORIAL_FEEDBACK_PROMPT = """You are the Editorial agent. Based on the book bible, what feedback do you have on originality, tone, pacing, style and creativeness for chapters {chapter_1_number} and {chapter_2_number}. Provide all findings and notes."""
