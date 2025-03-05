class ResearchAgent(BaseAgent):
    """Base class for research-focused agents."""
    
    def __init__(self, name: str, tools: List[BaseTool], config: Dict[str, Any]):
        super().__init__(name, tools)
        self.research_config = config
        self.storage = ResearchStorage()
    
    async def execute_research_cycle(self, state: ResearchState) -> Dict[str, Any]:
        """Execute a full research cycle with storage."""
        report = ResearchReport(
            project_id=state.project_id,
            agent_name=self.name,
            topic=state.current_input["task"],
            query_context=state.project.synopsis,
            findings=[],
            sources=[],
            confidence_score=0.0
        )
        
        while state["iterations"] < self.max_iterations:
            # Create iteration record
            iteration = ResearchIteration(
                report_id=report.report_id,
                queries=state["queries"],
                raw_results=[],
                processed_findings={},
                quality_score=0.0
            )
            
            # Execute research
            results = await execute_research(state["queries"], self.research_config)
            iteration.raw_results = results
            
            # Process findings
            findings = self.process_results(results)
            iteration.processed_findings = findings
            report.findings.extend(findings)
            
            # Store iteration
            await self.storage.store_iteration(iteration)
            
            # Update report
            quality = analyze_research_quality(report)
            report.confidence_score = quality["score"]
            
            if quality["score"] >= self.quality_threshold:
                await self.storage.store_report(report)
                return {
                    "report": report,
                    "quality_score": quality["score"]
                }
            
            # Continue research if needed
            gaps = identify_knowledge_gaps(report)
            report.identified_gaps = gaps
            new_queries = generate_followup_queries(gaps, state["context"])
            
            state["queries"] = new_queries
            state["iterations"] += 1
        
        # Store final report even if quality threshold not met
        await self.storage.store_report(report)
        return {
            "report": report,
            "quality_score": quality["score"]
        }

class DomainKnowledgeSpecialist(ResearchAgent):
    """Specializes in technical and domain-specific research."""
    
    async def __call__(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        topic = state.current_input["task"]
        
        research_state = ResearchState(
            queries=[ResearchQuery(
                query=f"technical analysis {topic}",
                context=state.project.synopsis,
                topic=topic
            )],
            iterations=0
        )
        
        result = await self.execute_research_cycle(research_state)
        return {
            "messages": [
                AIMessage(content=result["report"].findings)
            ]
        }

class CulturalAuthenticityExpert(ResearchAgent):
    """Specializes in cultural and sociological research."""
    # Similar implementation

class MarketAlignmentDirector(ResearchAgent):
    """Specializes in market research and trend analysis."""
    # Similar implementation

class FactVerificationSpecialist(ResearchAgent):
    """Specializes in fact checking and verification."""
    # Similar implementation