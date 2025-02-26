# agents/publication/__init__.py
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from storybook.config import Config
from storybook.utils.state import NovelState


class BlurbGeneratorAgent:
    """Blurb Generator Agent that creates compelling book descriptions."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "BlurbGenerator"

    def generate_blurbs(
        self, state: NovelState, styles: List[str] = None
    ) -> Dict[str, str]:
        """Generate multiple book blurb options in different styles."""
        if styles is None:
            styles = [
                "standard",
                "intriguing",
                "character-focused",
                "action-focused",
                "thematic",
            ]

        blurbs = {}

        for style in styles:
            prompt = PromptTemplate(
                template="""You are a professional copywriter creating a compelling {style} book blurb for a novel.

Novel Details:
Title: {title}
Genre: {genre}
Target Audience: {target_audience}
Premise: {premise}
Themes: {themes}

Main Characters:
{characters}

Plot Overview:
{plot_overview}

Create a powerful {style} book blurb that would appear on the back cover or in online listings. Your blurb should:
1. Hook the reader immediately
2. Introduce the main conflict or premise in an intriguing way
3. Hint at the stakes or consequences
4. Appeal specifically to fans of {genre}
5. Leave the reader wanting more
6. Be approximately 150-200 words

For a {style} blurb specifically:
{style_guidance}

Write only the blurb text, not any analysis or explanation.
""",
                input_variables=[
                    "style",
                    "title",
                    "genre",
                    "target_audience",
                    "premise",
                    "themes",
                    "characters",
                    "plot_overview",
                    "style_guidance",
                ],
            )

            # Create character summary
            characters_text = ""
            for name, character in state.characters.items():
                characters_text += (
                    f"{name}: {character.role} - {character.background[:100]}...\n"
                )

            if not characters_text:
                characters_text = "Main protagonist and supporting characters."

            # Create plot overview
            plot_overview = ""
            for i, point in enumerate(
                state.plot_points[:5]
            ):  # Limit to first 5 major plot points
                plot_overview += f"- {point.title}: {point.description[:100]}...\n"

            if not plot_overview:
                plot_overview = state.premise

            # Define style-specific guidance
            style_guidance = {
                "standard": "Use classic blurb structure with setup, complication, and stake-raising question.",
                "intriguing": "Focus on mystery and unanswered questions. Use provocative language that creates curiosity.",
                "character-focused": "Center on character journey and internal conflicts. Make the reader care deeply about the protagonist.",
                "action-focused": "Emphasize exciting plot elements and external conflicts. Use dynamic, propulsive language.",
                "thematic": "Highlight the novel's themes and deeper meanings. Appeal to readers looking for thought-provoking content.",
            }

            response = self.llm.invoke(
                prompt.format(
                    style=style,
                    title=state.project_name,
                    genre=state.genre,
                    target_audience=state.target_audience,
                    premise=state.premise,
                    themes=", ".join(state.themes),
                    characters=characters_text,
                    plot_overview=plot_overview,
                    style_guidance=style_guidance.get(
                        style, "Create a compelling and marketable description."
                    ),
                )
            )

            blurbs[style] = response.content

        return blurbs

    def analyze_blurb_effectiveness(
        self, blurb: str, genre: str, target_audience: str
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of a book blurb for its intended audience."""
        prompt = PromptTemplate(
            template="""You are a publishing marketing expert analyzing the effectiveness of a book blurb.

Blurb:
{blurb}

Genre: {genre}
Target Audience: {target_audience}

Analyze this blurb for:

1. HOOK STRENGTH
   Evaluate how effectively it grabs attention in the first sentence
   
2. PREMISE CLARITY
   Assess whether the core premise is clear and compelling
   
3. GENRE ALIGNMENT
   Analyze how well it signals the genre and meets genre reader expectations
   
4. CHARACTER APPEAL
   Evaluate how effectively it makes readers care about the characters
   
5. STAKES COMMUNICATION
   Assess how clearly it communicates what's at stake in the story
   
6. MARKETABILITY
   Analyze its appeal to the target audience
   Evaluate its competitive positioning in the marketplace
   
7. CURIOSITY GENERATION
   Assess how effectively it creates questions and curiosity

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific analysis
- Suggestions for improvement

Also provide an overall effectiveness score and summary.
""",
            input_variables=["blurb", "genre", "target_audience"],
        )

        response = self.llm.invoke(
            prompt.format(blurb=blurb, genre=genre, target_audience=target_audience)
        )

        # Extract aspects and scores from the response
        # This is a simplified extraction - a real implementation would be more robust
        aspects = [
            "hook strength",
            "premise clarity",
            "genre alignment",
            "character appeal",
            "stakes communication",
            "marketability",
            "curiosity generation",
        ]

        analysis = {"full_analysis": response.content, "scores": {}}

        for aspect in aspects:
            # Look for score pattern like "Hook Strength: 0.8" or "Hook Strength - 0.8"
            pattern = f"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    analysis["scores"][aspect.replace(" ", "_")] = score
                except:
                    analysis["scores"][
                        aspect.replace(" ", "_")
                    ] = 0.5  # Default if parsing fails
            else:
                analysis["scores"][
                    aspect.replace(" ", "_")
                ] = 0.5  # Default if not found

        # Calculate overall score as average of individual scores
        analysis["overall_score"] = (
            sum(analysis["scores"].values()) / len(analysis["scores"])
            if analysis["scores"]
            else 0.5
        )

        return analysis


class BookTitleOptimizerAgent:
    """Book Title Optimizer Agent that tests potential titles for market appeal."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "BookTitleOptimizer"

    def generate_title_options(
        self, state: NovelState, count: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate multiple title options with different approaches."""
        prompt = PromptTemplate(
            template="""You are a title generation specialist for a publishing company, creating compelling book titles.

Novel Details:
Genre: {genre}
Premise: {premise}
Themes: {themes}
Target Audience: {target_audience}

Generate {count} diverse and marketable title options for this novel. For each title:
1. Create a main title (and optional subtitle if appropriate for the genre)
2. Explain the rationale behind the title
3. Identify what makes it commercially appealing
4. Note which audience segments would be most attracted
5. Rate its marketability from 0.0 to 1.0

Use diverse approaches for the titles, such as:
- Evocative imagery
- Character-focused
- Thematic resonance
- Intriguing questions or premises
- Genre-specific conventions
- Symbolic or metaphorical
- Action or plot-focused
- Emotional appeal

Format each title as:
TITLE: [Title]
SUBTITLE (if applicable): [Subtitle]
RATIONALE: [Explanation]
APPEAL: [Commercial appeal factors]
TARGET: [Specific audience segments]
MARKETABILITY: [0.0-1.0 score]
""",
            input_variables=["genre", "premise", "themes", "target_audience", "count"],
        )

        response = self.llm.invoke(
            prompt.format(
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
                target_audience=state.target_audience,
                count=count,
            )
        )

        # Parse the response into title options
        title_options = []
        current_title = {}
        current_field = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                # Empty line indicates end of current title
                if current_title:
                    title_options.append(current_title)
                    current_title = {}
                continue

            if line.startswith("TITLE:"):
                # Start of a new title
                if current_title:
                    title_options.append(current_title)
                    current_title = {}

                current_title["title"] = line.split("TITLE:")[1].strip()
                current_field = "title"
            elif line.startswith("SUBTITLE:"):
                current_title["subtitle"] = line.split("SUBTITLE:")[1].strip()
                current_field = "subtitle"
            elif line.startswith("RATIONALE:"):
                current_title["rationale"] = line.split("RATIONALE:")[1].strip()
                current_field = "rationale"
            elif line.startswith("APPEAL:"):
                current_title["appeal"] = line.split("APPEAL:")[1].strip()
                current_field = "appeal"
            elif line.startswith("TARGET:"):
                current_title["target"] = line.split("TARGET:")[1].strip()
                current_field = "target"
            elif line.startswith("MARKETABILITY:"):
                try:
                    score_text = line.split("MARKETABILITY:")[1].strip()
                    current_title["marketability"] = float(
                        re.search(r"(\d+\.\d+)", score_text).group(1)
                    )
                except:
                    current_title["marketability"] = 0.5  # Default if parsing fails
                current_field = "marketability"
            elif current_field and current_field in current_title:
                # Continue previous field
                current_title[current_field] += " " + line

        # Add the last title if there is one
        if current_title:
            title_options.append(current_title)

        return title_options

    def test_title_effectiveness(
        self, title: str, genre: str, target_audience: str
    ) -> Dict[str, Any]:
        """Test the market effectiveness of a book title."""
        prompt = PromptTemplate(
            template="""You are a publishing market analyst evaluating the effectiveness of a book title.

Title: {title}
Genre: {genre}
Target Audience: {target_audience}

Analyze this title for:

1. IMMEDIATE APPEAL
   Evaluate its ability to grab attention at first glance
   
2. GENRE SIGNALING
   Assess how clearly it signals the book's genre to target readers
   
3. MEMORABILITY
   Analyze how memorable and distinctive the title is
   
4. INTRIGUE FACTOR
   Evaluate how much curiosity or interest it generates
   
5. THEMATIC RELEVANCE
   Assess how well it suggests the book's themes or content
   
6. MARKETABILITY
   Analyze its commercial appeal and positioning
   Evaluate search-friendliness and social media sharing potential
   
7. COMPETITIVE DIFFERENTIATION
   Assess how it stands out from similar titles in the genre

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific analysis
- Comparison to successful titles in the genre if relevant

Also provide an overall effectiveness score and summary.
""",
            input_variables=["title", "genre", "target_audience"],
        )

        response = self.llm.invoke(
            prompt.format(title=title, genre=genre, target_audience=target_audience)
        )

        # Extract aspects and scores from the response
        aspects = [
            "immediate appeal",
            "genre signaling",
            "memorability",
            "intrigue factor",
            "thematic relevance",
            "marketability",
            "competitive differentiation",
        ]

        analysis = {"full_analysis": response.content, "scores": {}}

        for aspect in aspects:
            # Look for score pattern
            pattern = f"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    analysis["scores"][aspect.replace(" ", "_")] = score
                except:
                    analysis["scores"][
                        aspect.replace(" ", "_")
                    ] = 0.5  # Default if parsing fails
            else:
                analysis["scores"][
                    aspect.replace(" ", "_")
                ] = 0.5  # Default if not found

        # Calculate overall score
        analysis["overall_score"] = (
            sum(analysis["scores"].values()) / len(analysis["scores"])
            if analysis["scores"]
            else 0.5
        )

        return analysis


class ComparableTitleAnalystAgent:
    """Comparable Title Analyst Agent that identifies strategic comp titles."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ComparableTitleAnalyst"

    def identify_comp_titles(
        self, state: NovelState, count: int = 8
    ) -> List[Dict[str, Any]]:
        """Identify strategic comparable titles for marketing positioning."""
        prompt = PromptTemplate(
            template="""You are a literary agent or publishing marketing specialist identifying comparable titles for a novel.

Novel Details:
Title: {title}
Genre: {genre}
Premise: {premise}
Themes: {themes}
Target Audience: {target_audience}
Writing Style: {writing_style}

Identify {count} strategic comparable titles ("comps") that would help position this novel in the marketplace. For each comp:

1. Select published books from the last 5 years when possible (with some exceptions for classics or important genre touchstones)
2. Focus on books that achieved commercial success
3. Identify titles that would appeal to the same readership
4. Include books with similar themes, tone, style, or plot elements
5. Include a mix of well-known and up-and-coming authors
6. Consider titles that achieved the level of success this book aspires to

For each comp title, provide:
- Title and author
- Year published
- Brief description of the book
- Specific points of comparison to the novel
- Why readers of the comp would like this novel
- The comp's approximate commercial performance/recognition
- How to leverage this comp in marketing

Format each comp as:
TITLE: [Title]
AUTHOR: [Author]
PUBLISHED: [Year]
DESCRIPTION: [Brief description]
COMPARISON POINTS: [Specific similarities]
READER APPEAL: [Why readers of the comp would like this novel]
COMMERCIAL STATUS: [Sales level, awards, recognition]
MARKETING LEVERAGE: [How to use this comp in marketing]
""",
            input_variables=[
                "title",
                "genre",
                "premise",
                "themes",
                "target_audience",
                "writing_style",
                "count",
            ],
        )

        # Extract writing style information if available
        writing_style = "Not specified"
        if hasattr(state, "writing_plan") and state.writing_plan:
            if "stylistic_guidelines" in state.writing_plan:
                writing_style = state.writing_plan["stylistic_guidelines"]

        response = self.llm.invoke(
            prompt.format(
                title=state.project_name,
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
                target_audience=state.target_audience,
                writing_style=writing_style,
                count=count,
            )
        )

        # Parse the response into comp titles
        comp_titles = []
        current_comp = {}
        current_field = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                # Empty line indicates end of current comp
                if current_comp:
                    comp_titles.append(current_comp)
                    current_comp = {}
                continue

            if line.startswith("TITLE:"):
                # Start of a new comp
                if current_comp:
                    comp_titles.append(current_comp)
                    current_comp = {}

                current_comp["title"] = line.split("TITLE:")[1].strip()
                current_field = "title"
            elif line.startswith("AUTHOR:"):
                current_comp["author"] = line.split("AUTHOR:")[1].strip()
                current_field = "author"
            elif line.startswith("PUBLISHED:"):
                try:
                    year_text = line.split("PUBLISHED:")[1].strip()
                    current_comp["year"] = int(
                        re.search(r"(\d{4})", year_text).group(1)
                    )
                except:
                    current_comp["year"] = 0  # Default if parsing fails
                current_field = "year"
            elif line.startswith("DESCRIPTION:"):
                current_comp["description"] = line.split("DESCRIPTION:")[1].strip()
                current_field = "description"
            elif line.startswith("COMPARISON POINTS:"):
                current_comp["comparison_points"] = line.split("COMPARISON POINTS:")[
                    1
                ].strip()
                current_field = "comparison_points"
            elif line.startswith("READER APPEAL:"):
                current_comp["reader_appeal"] = line.split("READER APPEAL:")[1].strip()
                current_field = "reader_appeal"
            elif line.startswith("COMMERCIAL STATUS:"):
                current_comp["commercial_status"] = line.split("COMMERCIAL STATUS:")[
                    1
                ].strip()
                current_field = "commercial_status"
            elif line.startswith("MARKETING LEVERAGE:"):
                current_comp["marketing_leverage"] = line.split("MARKETING LEVERAGE:")[
                    1
                ].strip()
                current_field = "marketing_leverage"
            elif current_field and current_field in current_comp:
                # Continue previous field
                current_comp[current_field] += " " + line

        # Add the last comp if there is one
        if current_comp:
            comp_titles.append(current_comp)

        return comp_titles

    def create_positioning_statement(
        self, state: NovelState, comp_titles: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Create a marketing positioning statement based on comparable titles."""
        prompt = PromptTemplate(
            template="""You are a publishing marketing expert creating a positioning statement for a novel.

Novel Details:
Title: {title}
Genre: {genre}
Premise: {premise}
Themes: {themes}

Comparable Titles:
{comp_titles}

Create a compelling positioning statement that:
1. Places this novel in the market context
2. Uses the "X meets Y" formula with appropriate comp titles
3. Highlights what makes this novel both familiar and unique
4. Appeals specifically to readers of the comparable titles
5. Communicates the novel's core appeal in a concise, marketable way

Your positioning statement should be 1-2 sentences that could be used in marketing materials, query letters, or sales presentations.

Also provide three alternative positioning approaches with different emphasis and comp title combinations.

Format your response as:
PRIMARY POSITIONING: [1-2 sentence positioning statement]

ALTERNATIVES:
1. [Alternative positioning 1]
2. [Alternative positioning 2]
3. [Alternative positioning 3]

MARKETING STRATEGY:
[Brief explanation of how to leverage this positioning in marketing]
""",
            input_variables=["title", "genre", "premise", "themes", "comp_titles"],
        )

        # Format comp titles for the prompt
        comp_titles_text = ""
        for comp in comp_titles[:5]:  # Limit to first 5 comps
            comp_titles_text += (
                f"- {comp['title']} by {comp['author']}: {comp['comparison_points']}\n"
            )

        response = self.llm.invoke(
            prompt.format(
                title=state.project_name,
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
                comp_titles=comp_titles_text,
            )
        )

        # Extract sections from the response
        sections = {
            "primary_positioning": "",
            "alternatives": [],
            "marketing_strategy": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "PRIMARY POSITIONING:" in line:
                current_section = "primary_positioning"
                sections["primary_positioning"] = line.split("PRIMARY POSITIONING:")[
                    1
                ].strip()
            elif "ALTERNATIVES:" in line:
                current_section = "alternatives"
            elif "MARKETING STRATEGY:" in line:
                current_section = "marketing_strategy"
                sections["marketing_strategy"] = line.split("MARKETING STRATEGY:")[
                    1
                ].strip()
            elif (
                current_section == "alternatives" and line.strip() and line[0].isdigit()
            ):
                # This is an alternative positioning statement
                alternative = line.split(".", 1)[1].strip()
                sections["alternatives"].append(alternative)
            elif current_section == "marketing_strategy" and line.strip():
                # Continue marketing strategy
                sections["marketing_strategy"] += " " + line.strip()

        return sections


class TagAndCategorySpecialistAgent:
    """Tag and Category Specialist Agent that optimizes metadata for discovery."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "TagAndCategorySpecialist"

    def generate_metadata(self, state: NovelState) -> Dict[str, Any]:
        """Generate optimized metadata for book discovery platforms."""
        prompt = PromptTemplate(
            template="""You are a metadata specialist optimizing a novel's discoverability on book platforms.

Novel Details:
Title: {title}
Genre: {genre}
Subgenres: {subgenres}
Premise: {premise}
Themes: {themes}
Target Audience: {target_audience}

Create optimized metadata for this novel to maximize its discovery on platforms like Amazon, Goodreads, and library catalogs. Include:

1. PRIMARY CATEGORIES
   Identify 3-5 most appropriate BISAC categories with codes
   Recommend optimal Amazon/online bookstore category placements
   
2. KEYWORDS & SEARCH TERMS
   Provide 20-30 effective keywords for search optimization
   Group keywords by theme/category
   Include high-volume search terms and relevant niche terms
   
3. AUDIENCE TAGS
   Create specific tags that will resonate with the target audience
   Include age range, interest group, and reader profile tags
   
4. CONTENT DESCRIPTORS
   Provide descriptive tags for content elements and features
   Include tone, mood, setting, time period, etc.
   
5. COMPARABLE TITLE ASSOCIATIONS
   Suggest metadata connections to relevant comparable titles
   Include author, series, and thematic connections

Format your response with clear sections for each metadata type.
""",
            input_variables=[
                "title",
                "genre",
                "subgenres",
                "premise",
                "themes",
                "target_audience",
            ],
        )

        response = self.llm.invoke(
            prompt.format(
                title=state.project_name,
                genre=state.genre,
                subgenres=", ".join(state.subgenres),
                premise=state.premise,
                themes=", ".join(state.themes),
                target_audience=state.target_audience,
            )
        )

        # Extract sections from the response
        sections = {
            "primary_categories": [],
            "keywords": [],
            "audience_tags": [],
            "content_descriptors": [],
            "comparable_associations": [],
        }

        current_section = None

        for line in response.content.split("\n"):
            if "PRIMARY CATEGORIES" in line or "1. PRIMARY CATEGORIES" in line:
                current_section = "primary_categories"
                continue
            elif "KEYWORDS & SEARCH TERMS" in line or "2. KEYWORDS" in line:
                current_section = "keywords"
                continue
            elif "AUDIENCE TAGS" in line or "3. AUDIENCE TAGS" in line:
                current_section = "audience_tags"
                continue
            elif "CONTENT DESCRIPTORS" in line or "4. CONTENT DESCRIPTORS" in line:
                current_section = "content_descriptors"
                continue
            elif "COMPARABLE TITLE ASSOCIATIONS" in line or "5. COMPARABLE" in line:
                current_section = "comparable_associations"
                continue

            if current_section and line.strip():
                # Check if this is a list item
                if (
                    line.strip().startswith("-")
                    or line.strip().startswith("•")
                    or (line.strip()[0].isdigit() and "." in line[:3])
                ):
                    # Extract the item text
                    item_text = line.strip()
                    for prefix in ["-", "•"]:
                        if item_text.startswith(prefix):
                            item_text = item_text[len(prefix) :].strip()

                    if line.strip()[0].isdigit() and "." in line[:3]:
                        item_text = line.strip().split(".", 1)[1].strip()

                    sections[current_section].append(item_text)

        # For keywords, if we didn't extract any as list items, try to extract from text
        if not sections["keywords"] and current_section == "keywords":
            keyword_text = " ".join(
                [
                    line
                    for line in response.content.split("\n")
                    if "KEYWORDS" not in line
                ]
            )
            # Extract terms in quotes or comma-separated
            quoted_keywords = re.findall(r'"([^"]*)"', keyword_text)
            comma_keywords = [
                k.strip()
                for k in keyword_text.split(",")
                if k.strip() and len(k.strip()) > 3
            ]
            sections["keywords"] = list(set(quoted_keywords + comma_keywords))

        return {"metadata": sections, "full_metadata_document": response.content}

    def optimize_categories(
        self, metadata: Dict[str, Any], target_platform: str
    ) -> Dict[str, Any]:
        """Optimize category selection for a specific platform."""
        prompt = PromptTemplate(
            template="""You are a book category optimization specialist for {target_platform}.

Novel Metadata:
{metadata}

Create an optimized category strategy for this novel specifically on {target_platform}. Your response should include:

1. PRIMARY CATEGORY
   The single best top-level category for this book on {target_platform}
   
2. SECONDARY CATEGORIES
   3-5 additional categories where this book should be listed
   
3. CATEGORY STRATEGY
   Explanation of why these categories are optimal
   Analysis of competition level in each category
   Recommendation for category ranking potential
   
4. BROWSE PATH OPTIMIZATION
   The optimal browse paths for readers to discover this book
   Category combinations that will increase visibility
   
5. CATEGORY-SPECIFIC KEYWORDS
   Keywords that will strengthen category placement
   Terms that will improve ranking within these categories

Format your response with clear sections and specific, actionable recommendations.
""",
            input_variables=["target_platform", "metadata"],
        )

        response = self.llm.invoke(
            prompt.format(
                target_platform=target_platform,
                metadata="\n".join(
                    [
                        f"{k}: {v}"
                        for k, v in metadata.items()
                        if k != "full_metadata_document"
                    ]
                ),
            )
        )

        # Extract sections from the response
        sections = {
            "primary_category": "",
            "secondary_categories": [],
            "category_strategy": "",
            "browse_paths": [],
            "category_keywords": [],
        }

        current_section = None

        for line in response.content.split("\n"):
            if "PRIMARY CATEGORY" in line or "1. PRIMARY CATEGORY" in line:
                current_section = "primary_category"
                continue
            elif "SECONDARY CATEGORIES" in line or "2. SECONDARY CATEGORIES" in line:
                current_section = "secondary_categories"
                continue
            elif "CATEGORY STRATEGY" in line or "3. CATEGORY STRATEGY" in line:
                current_section = "category_strategy"
                continue
            elif "BROWSE PATH" in line or "4. BROWSE PATH" in line:
                current_section = "browse_paths"
                continue
            elif "CATEGORY-SPECIFIC KEYWORDS" in line or "5. CATEGORY" in line:
                current_section = "category_keywords"
                continue

            if current_section and line.strip():
                if current_section == "primary_category":
                    if sections["primary_category"]:
                        sections["primary_category"] += " " + line.strip()
                    else:
                        sections["primary_category"] = line.strip()
                elif current_section == "category_strategy":
                    if sections["category_strategy"]:
                        sections["category_strategy"] += " " + line.strip()
                    else:
                        sections["category_strategy"] = line.strip()
                elif current_section in [
                    "secondary_categories",
                    "browse_paths",
                    "category_keywords",
                ]:
                    # Check if this is a list item
                    if (
                        line.strip().startswith("-")
                        or line.strip().startswith("•")
                        or (line.strip()[0].isdigit() and "." in line[:3])
                    ):
                        # Extract the item text
                        item_text = line.strip()
                        for prefix in ["-", "•"]:
                            if item_text.startswith(prefix):
                                item_text = item_text[len(prefix) :].strip()

                        if line.strip()[0].isdigit() and "." in line[:3]:
                            item_text = line.strip().split(".", 1)[1].strip()

                        sections[current_section].append(item_text)

        # Clean up the primary category
        if sections["primary_category"]:
            # Remove prefixes like "Primary Category:" if present
            if ":" in sections["primary_category"]:
                sections["primary_category"] = (
                    sections["primary_category"].split(":", 1)[1].strip()
                )

        return {
            "platform": target_platform,
            "optimized_categories": sections,
            "full_category_strategy": response.content,
        }
