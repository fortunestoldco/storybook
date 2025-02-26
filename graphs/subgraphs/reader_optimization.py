# graphs/subgraphs/reader_optimization.py
from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph

from ...agents.reader_experience import EmotionalArcAnalyzerAgent, HookOptimizationAgent, ReadabilitySpecialistAgent, PageTurnerDesignerAgent
from ...agents.project_management import ProjectLeadAgent
from ...utils.state import NovelState, ProjectStatus, Chapter
from ...config import NovelGenConfig

class ReaderOptimizationState(TypedDict):
    """State for the reader optimization subgraph."""
    novel_state: NovelState
    emotional_analysis: Dict[str, Any]
    emotionally_enhanced_chapters: Dict[int, Chapter]
    hook_optimized_chapters: Dict[int, Chapter]
    readability_metrics: Dict[int, Dict[str, Any]]
    adjusted_chapters: Dict[int, Chapter]
    page_turner_enhanced_chapters: Dict[int, Chapter]

def create_reader_optimization_graph(config: NovelGenConfig):
    """Create the reader optimization phase subgraph."""
    # Initialize agents
    emotional_arc_analyzer = EmotionalArcAnalyzerAgent(config)
    hook_optimization = HookOptimizationAgent(config)
    readability_specialist = ReadabilitySpecialistAgent(config)
    page_turner_designer = PageTurnerDesignerAgent(config)
    project_lead = ProjectLeadAgent(config)
    
    # Define state
    workflow = StateGraph(ReaderOptimizationState)
    
    # Define nodes
    
    # 1. Analyze emotional arc
    def analyze_emotional_arc(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        emotional_analysis = emotional_arc_analyzer.analyze_emotional_arc(novel_state)
        
        # Update novel state with emotional coherence score
        novel_state.narrative_engagement_metrics["emotional_coherence"] = emotional_analysis["coherence_score"]
        
        return {
            **state, 
            "emotional_analysis": emotional_analysis,
            "novel_state": novel_state
        }
    
    # 2. Enhance emotional impact
    def enhance_emotional_impact(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        emotional_analysis = state["emotional_analysis"]
        emotionally_enhanced_chapters = {}
        
        # Extract emotional goals for each chapter from the analysis
        chapter_emotions = {}
        if "sections" in emotional_analysis and "reader_journey" in emotional_analysis["sections"]:
            reader_journey = emotional_analysis["sections"]["reader_journey"]
            for line in reader_journey.split("\n"):
                if "Chapter" in line and ":" in line:
                    try:
                        chapter_num = int(''.join(filter(str.isdigit, line.split("Chapter")[1].split()[0])))
                        emotion = line.split(":", 1)[1].strip()
                        chapter_emotions[chapter_num] = emotion
                    except:
                        pass
        
        # Apply emotional enhancements to each chapter
        for chapter_num, chapter in novel_state.chapters.items():
            # Use specified emotion or a default if not found
            emotional_goal = chapter_emotions.get(
                chapter_num, 
                "A balanced emotional experience with appropriate intensity"
            )
            
            enhanced_chapter = emotional_arc_analyzer.enhance_emotional_impact(
                chapter=chapter,
                emotional_goal=emotional_goal
            )
            
            # Update novel state with enhanced chapter
            novel_state.chapters[chapter_num] = enhanced_chapter
            
            # Update word count
            word_count_diff = enhanced_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff
            
            emotionally_enhanced_chapters[chapter_num] = enhanced_chapter
        
        return {
            **state, 
            "emotionally_enhanced_chapters": emotionally_enhanced_chapters,
            "novel_state": novel_state
        }
    
    # 3. Optimize hooks
    def optimize_hooks(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        hook_optimized_chapters = {}
        
        for chapter_num, chapter in novel_state.chapters.items():
            optimized_chapter = hook_optimization.optimize_hooks(chapter)
            
            # Update novel state with optimized chapter
            novel_state.chapters[chapter_num] = optimized_chapter
            
            # Update word count
            word_count_diff = optimized_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff
            
            # Analyze hook effectiveness
            hook_analysis = hook_optimization.analyze_hook_effectiveness(optimized_chapter.content)
            novel_state.chapters[chapter_num].quality_metrics["hook_effectiveness"] = hook_analysis["overall_score"]
            
            hook_optimized_chapters[chapter_num] = optimized_chapter
        
        return {
            **state, 
            "hook_optimized_chapters": hook_optimized_chapters,
            "novel_state": novel_state
        }
    
    # 4. Analyze readability
    def analyze_readability(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        readability_metrics = {}
        
        for chapter_num, chapter in novel_state.chapters.items():
            metrics = readability_specialist.analyze_readability(chapter.content)
            readability_metrics[chapter_num] = metrics
            
            # Add readability metrics to chapter quality metrics
            novel_state.chapters[chapter_num].quality_metrics["grade_level"] = metrics["grade_level"]
        
        return {
            **state, 
            "readability_metrics": readability_metrics,
            "novel_state": novel_state
        }
    
    # 5. Adjust readability
    def adjust_readability(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        readability_metrics = state["readability_metrics"]
        adjusted_chapters = {}
        
        for chapter_num, chapter in novel_state.chapters.items():
            # Determine if readability adjustment is needed
            metrics = readability_metrics.get(chapter_num, {})
            target_audience = novel_state.target_audience
            
            # Adjust any chapters that don't match the target audience
            adjusted_chapter = readability_specialist.adjust_readability(
                chapter=chapter,
                target_audience=target_audience
            )
            
            # Update novel state with adjusted chapter
            novel_state.chapters[chapter_num] = adjusted_chapter
            
            # Update word count
            word_count_diff = adjusted_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff
            
            adjusted_chapters[chapter_num] = adjusted_chapter
        
        return {
            **state, 
            "adjusted_chapters": adjusted_chapters,
            "novel_state": novel_state
        }
    
    # 6. Enhance page-turner qualities
    def enhance_page_turner_qualities(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        page_turner_enhanced_chapters = {}
        
        for chapter_num, chapter in novel_state.chapters.items():
            enhanced_chapter = page_turner_designer.enhance_page_turner_qualities(chapter)
            
            # Update novel state with enhanced chapter
            novel_state.chapters[chapter_num] = enhanced_chapter
            
            # Update word count
            word_count_diff = enhanced_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff
            
            # Analyze page-turner qualities
            analysis = page_turner_designer.analyze_page_turner_qualities(enhanced_chapter.content)
            novel_state.chapters[chapter_num].quality_metrics["page_turner_score"] = analysis["overall_score"]
            
            page_turner_enhanced_chapters[chapter_num] = enhanced_chapter
        
        return {
            **state, 
            "page_turner_enhanced_chapters": page_turner_enhanced_chapters,
            "novel_state": novel_state
        }
    
    # 7. Set next phase
    def set_next_phase(state: ReaderOptimizationState) -> ReaderOptimizationState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(novel_state, ProjectStatus.PREPARING_PUBLICATION)
        
        # Update completion metrics
        novel_state.phase_progress = 1.0  # This phase is complete
        
        # Calculate overall quality metrics
        quality_metrics = {
            "emotional_coherence": novel_state.narrative_engagement_metrics.get("emotional_coherence", 0.0),
            "hook_effectiveness": 0.0,
            "page_turner_score": 0.0,
            "prose_quality": 0.0,
            "dialogue_quality": 0.0
        }
        
        # Average the quality metrics across all chapters
        for chapter in novel_state.chapters.values():
            for metric in ["hook_effectiveness", "page_turner_score", "prose_quality", "dialogue_quality"]:
                if metric in chapter.quality_metrics:
                    quality_metrics[metric] += chapter.quality_metrics[metric]
        
        # Calculate averages
        num_chapters = len(novel_state.chapters)
        if num_chapters > 0:
            for metric in ["hook_effectiveness", "page_turner_score", "prose_quality", "dialogue_quality"]:
                quality_metrics[metric] /= num_chapters
        
        # Update narrative engagement metrics
        novel_state.narrative_engagement_metrics.update(quality_metrics)
        
        return {**state, "novel_state": novel_state}
    
    # Add nodes to graph
    workflow.add_node("analyze_emotional_arc", analyze_emotional_arc)
    workflow.add_node("enhance_emotional_impact", enhance_emotional_impact)
    workflow.add_node("optimize_hooks", optimize_hooks)
    workflow.add_node("analyze_readability", analyze_readability)
    workflow.add_node("adjust_readability", adjust_readability)
    workflow.add_node("enhance_page_turner_qualities", enhance_page_turner_qualities)
    workflow.add_node("set_next_phase", set_next_phase)
    
    # Define edges
    workflow.add_edge("analyze_emotional_arc", "enhance_emotional_impact")
    workflow.add_edge("enhance_emotional_impact", "optimize_hooks")
    workflow.add_edge("optimize_hooks", "analyze_readability")
    workflow.add_edge("analyze_readability", "adjust_readability")
    workflow.add_edge("adjust_readability", "enhance_page_turner_qualities")
    workflow.add_edge("enhance_page_turner_qualities", "set_next_phase")
    
    # Set entry and exit points
    workflow.set_entry_point("analyze_emotional_arc")
    
    return workflow
