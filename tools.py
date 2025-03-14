from tavily import TavilyClient
from config import TAVILY_API_KEY
import traceback
from typing import Dict, Any, List, Optional
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ResearchTools:
    def __init__(self, tavily_client=None):
        """Initialize research tools with Tavily client."""
        try:
            self.tavily = tavily_client or TavilyClient(api_key=TAVILY_API_KEY)
            if not TAVILY_API_KEY:
                logger.warning("No Tavily API key provided. Research capabilities will be limited.")
        except Exception as e:
            logger.error(f"Error initializing ResearchTools: {str(e)}")
            logger.debug(traceback.format_exc())
            self.tavily = None

    def domain_research(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Conduct domain-specific research using Tavily."""
        try:
            if not self.tavily:
                raise ValueError("Tavily client not available. Check your API key.")
                
            logger.info(f"Conducting domain research: {query[:100]}...")
            
            # Enhance query with context if available
            full_query = query
            if context:
                full_query = f"{query}. Context: {context}"

            # Use Tavily for research with comprehensive search depth
            search_result = self.tavily.search(query=full_query, search_depth="comprehensive")
            logger.info(f"Domain research complete with {len(search_result.get('results', []))} results")

            # Extract and format results
            results = []
            for result in search_result.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                })

            # Create a summary and structured response
            summary = search_result.get("answer", "No summary available")

            return {
                "query": query,
                "summary": summary,
                "results": results,
                "domain_specific_data": {
                    "sources": len(results),
                    "comprehensive_answer": summary
                }
            }

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in domain research: {str(e)}")
            logger.debug(error_details)

            # Return error information
            return {
                "query": query,
                "summary": f"Research error: {str(e)}",
                "results": [],
                "domain_specific_data": {
                    "error": str(e),
                    "error_details": error_details
                }
            }

    def cultural_research(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Conduct cultural research using Tavily."""
        try:
            if not self.tavily:
                raise ValueError("Tavily client not available. Check your API key.")
                
            logger.info(f"Conducting cultural research: {query[:100]}...")
                
            # Enhance query with cultural focus and context
            cultural_query = f"Cultural context and authenticity information about: {query}"
            if context:
                cultural_query = f"{cultural_query}. Context: {context}"

            # Use Tavily for research with comprehensive search depth
            search_result = self.tavily.search(query=cultural_query, search_depth="comprehensive")
            logger.info(f"Cultural research complete with {len(search_result.get('results', []))} results")

            # Extract and format results
            results = []
            for result in search_result.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                })

            # Create a summary and structured response
            summary = search_result.get("answer", "No cultural context information available")

            return {
                "query": query,
                "summary": summary,
                "results": results,
                "cultural_context": {
                    "authenticity_check": True,
                    "cultural_insights": summary
                }
            }

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in cultural research: {str(e)}")
            logger.debug(error_details)

            # Return error information
            return {
                "query": query,
                "summary": f"Cultural research error: {str(e)}",
                "results": [],
                "cultural_context": {
                    "error": str(e),
                    "error_details": error_details
                }
            }

    def market_research(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Conduct market research using Tavily."""
        try:
            if not self.tavily:
                raise ValueError("Tavily client not available. Check your API key.")
                
            logger.info(f"Conducting market research: {query[:100]}...")
                
            # Enhance query with market focus and context
            market_query = f"Market trends and audience preferences for: {query}"
            if context:
                market_query = f"{market_query}. Context: {context}"

            # Use Tavily for research with comprehensive search depth
            search_result = self.tavily.search(query=market_query, search_depth="comprehensive")
            logger.info(f"Market research complete with {len(search_result.get('results', []))} results")

            # Extract and format results
            results = []
            for result in search_result.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                })

            # Create a summary and structured response
            summary = search_result.get("answer", "No market information available")

            return {
                "query": query,
                "summary": summary,
                "results": results,
                "market_trends": {
                    "audience_preferences": self._extract_audience_info(summary),
                    "market_insights": summary
                }
            }

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in market research: {str(e)}")
            logger.debug(error_details)

            # Return error information
            return {
                "query": query,
                "summary": f"Market research error: {str(e)}",
                "results": [],
                "market_trends": {
                    "error": str(e),
                    "error_details": error_details
                }
            }

    def fact_verification(self, statements: List[str], context: Optional[str] = None) -> Dict[str, Any]:
        """Verify factual statements using Tavily."""
        try:
            if not self.tavily:
                raise ValueError("Tavily client not available. Check your API key.")

            logger.info(f"Conducting fact verification on {len(statements)} statements")
            
            verification_status = {}
            results = []
            summary = ""

            # Process each statement
            for statement in statements:
                # Create verification query
                verify_query = f"Verify if this is factually accurate: '{statement}'"
                if context:
                    verify_query = f"{verify_query}. Context: {context}"

                # Use Tavily to verify
                search_result = self.tavily.search(query=verify_query, search_depth="moderate")

                # Extract result
                answer = search_result.get("answer", "")

                # Determine if verified (simplistic approach - could be improved)
                is_verified = not any(x in answer.lower() for x in ["incorrect", "false", "wrong", "not accurate", "inaccurate"])
                verification_status[statement] = is_verified

                # Add to results
                for result in search_result.get("results", [])[:2]:  # Limit to top 2 results per statement
                    results.append({
                        "statement": statement,
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", "")
                    })

                # Add to summary
                summary += f"Statement: {statement}\nVerification: {'Supported' if is_verified else 'Questioned'}\n\n"

            logger.info(f"Fact verification complete. {sum(1 for v in verification_status.values() if v)} statements verified.")

            return {
                "query": "Fact verification",
                "summary": summary.strip(),
                "results": results,
                "verification_status": verification_status
            }

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in fact verification: {str(e)}")
            logger.debug(error_details)

            # Return error information
            return {
                "query": "Fact verification",
                "summary": f"Verification error: {str(e)}",
                "results": [],
                "verification_status": {statement: None for statement in statements}
            }

    def _extract_audience_info(self, text: str) -> Dict[str, Any]:
        """Extract audience information from research summary."""
        audience_info = {
            "age_groups": [],
            "demographics": [],
            "preferences": []
        }

        # Simple regex patterns to extract information
        # This could be enhanced using LLM extraction for better results
        age_pattern = r"(?:age|ages)\s+(\d+(?:\s*-\s*\d+)?)"
        age_matches = re.findall(age_pattern, text.lower())
        if age_matches:
            audience_info["age_groups"] = age_matches

        # Look for demographic keywords
        demographics = ["young adult", "adult", "teen", "children", "male", "female",
                       "urban", "rural", "suburban", "high income", "middle income"]
        for demo in demographics:
            if demo.lower() in text.lower():
                audience_info["demographics"].append(demo)

        # Look for preference keywords
        preference_patterns = [r"prefer(?:s|red)?\s+(\w+)", r"interest(?:s|ed)?\s+in\s+(\w+)"]
        for pattern in preference_patterns:
            pref_matches = re.findall(pattern, text.lower())
            audience_info["preferences"].extend(pref_matches)

        return audience_info