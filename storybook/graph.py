    async def market_research(state: GraphState):
        """Analyze market positioning and trends."""
        results = await agents["market_researcher"].process_manuscript(state)
        return {
            "market_analysis": results,
            "state": "market_analyzed"
        }

    async def content_analysis(state: GraphState):
        """Analyze content and themes."""
        results = await agents["content_analyzer"].process_manuscript(state)
        return {
            "content_analysis": results,
            "state": "content_analyzed"
        }

    async def creative_development(state: GraphState):
        """Develop creative elements."""
        character_results = await agents["character_developer"].process_manuscript(state)
        dialogue_results = await agents["dialogue_enhancer"].process_manuscript(
            state,
            characters=character_results
        )
        world_results = await agents["world_builder"].process_manuscript(state)
        subplot_results = await agents["subplot_weaver"].process_manuscript(
            state,
            characters=character_results
        )
        
        return {
            "characters": character_results,
            "dialogue": dialogue_results,
            "world_building": world_results,
            "subplots": subplot_results,
            "state": "creative_complete"
        }

    async def story_development(state: GraphState):
        """Develop story structure and language."""
        arc_results = await agents["story_arc_analyst"].process_manuscript(
            state,
            characters=state["characters"],
            subplots=state["subplots"]
        )
        language_results = await agents["language_polisher"].process_manuscript(state)
        
        return {
            "story_arc": arc_results,
            "language": language_results,
            "state": "story_complete"
        }

    async def quality_review(state: GraphState):
        """Review and validate story elements."""
        review_results = await agents["quality_reviewer"].process_manuscript(
            state,
            context=state
        )
        return {
            "quality_review": review_results,
            "state": "complete"
        }

    # Add nodes to workflow with unique names
    workflow.add_node("analyze_market", market_research)
    workflow.add_node("analyze_content", content_analysis)
    workflow.add_node("develop_creative", creative_development)
    workflow.add_node("develop_story", story_development)
    workflow.add_node("review_quality", quality_review)

    # Configure workflow routing with new node names
    workflow.add_edge(START, "analyze_market")
    workflow.add_edge("analyze_market", "analyze_content")
    workflow.add_edge("analyze_content", "develop_creative")
    workflow.add_edge("develop_creative", "develop_story")
    workflow.add_edge("develop_story", "review_quality")
    workflow.add_edge("review_quality", END)
    
    return workflow.compile()
