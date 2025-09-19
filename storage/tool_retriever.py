import os
import sys
from typing import List, Any, Dict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    EC2AndKubernetesTools,
    DatabaseTools,
    MonitoringTools,
    NetworkingTools,
    UtilityTools
)

# Load environment
load_dotenv()

class ToolRetriever:
    """
    Retrieves relevant DevOps tools using semantic search and re-ranking
    """
    
    def __init__(self):
        """Initialize Pinecone, Cohere, and tool instances"""
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index("agent-tools")
        
        # Initialize embedding model (same as storage)
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # Initialize Cohere for re-ranking
        self.cohere_client = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
        
        # Pre-load tool instances for method access
        self.tool_instances = {
            'EC2AndKubernetesTools': EC2AndKubernetesTools(),
            'DatabaseTools': DatabaseTools(),
            'MonitoringTools': MonitoringTools(),
            'NetworkingTools': NetworkingTools(),
            'UtilityTools': UtilityTools()
        }
        
        print("✅ ToolRetriever initialized")
        print(f"   📊 Index ready: {self.index.describe_index_stats()['total_vector_count']} tools available")

    def retrieve_tools(self, user_query: str, top_k: int = 10, quality_threshold: float = 0.6) -> List[Any]:
        """
        Retrieve relevant tools using adaptive approach with semantic search and re-ranking
        
        Args:
            user_query (str): User's natural language query
            top_k (int): Number of tools to return (default: 10)
            quality_threshold (float): Minimum relevance score for high-quality results (default: 0.4)
            
        Returns:
            List[Any]: List of callable method references ready for LangChain bind_tools()
        """
        
        print(f"🔍 Retrieving tools for query: '{user_query}'")
        
        # Step 1: Semantic search with Pinecone (get 3x more candidates)
        retrieve_candidates = max(top_k * 3, 30)  # At least 30 candidates
        pinecone_results = self._semantic_search(user_query, retrieve_candidates)
        
        if not pinecone_results:
            print("⚠️ No results found in Pinecone")
            return []
        
        print(f"📊 Found {len(pinecone_results)} candidates from Pinecone")
        
        # Step 2: Re-rank with Cohere (narrow down to 2x target)
        rerank_candidates = min(top_k * 2, len(pinecone_results))
        reranked_results = self._rerank_with_cohere(user_query, pinecone_results, rerank_candidates)
        
        print(f"🔄 Re-ranked to top {len(reranked_results)} candidates")
        
        # Step 3: Adaptive selection based on quality
        selected_tools = self._adaptive_selection(reranked_results, top_k, quality_threshold)
        
        # Step 4: Convert to method references
        method_references = self._get_method_references(selected_tools)
        
        print(f"✅ Retrieved {len(method_references)} tools:")
        for i, method in enumerate(method_references[:5], 1):  # Show first 5
            print(f"   {i}. {method.__self__.__class__.__name__}.{method.__name__}")
        if len(method_references) > 5:
            print(f"   ... and {len(method_references) - 5} more")
        
        return method_references

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic search with Pinecone"""
        
        # Generate embedding for query
        query_embedding = self.model.encode(query).tolist()
        
        # Search Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"chunk_type": "individual_tool"}  # Only get individual tool chunks
            )
            
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")
            return []

    def _rerank_with_cohere(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Re-rank candidates using Cohere"""
        
        try:
            # Prepare documents for re-ranking
            documents = []
            for candidate in candidates:
                metadata = candidate['metadata']
                # Create rich document text for re-ranking
                doc_text = f"Tool: {metadata['tool_name']} - {metadata['description']}"
                documents.append(doc_text)
            
            # Re-rank with Cohere
            rerank_response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents
            )
            
            # Map re-ranked results back to original candidates
            reranked_candidates = []
            for result in rerank_response.results[:top_k]:
                original_candidate = candidates[result.index]
                original_candidate['cohere_score'] = result.relevance_score
                reranked_candidates.append(original_candidate)
            
            return reranked_candidates
            
        except Exception as e:
            print(f"⚠️ Error in re-ranking, falling back to Pinecone scores: {str(e)}")
            # Fallback to Pinecone scores if Cohere fails
            return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]

    def _adaptive_selection(self, reranked_results: List[Dict], top_k: int, quality_threshold: float) -> List[Dict]:
        """Adaptive selection based on quality threshold"""
        
        # Get high-quality results above threshold
        high_quality = [
            result for result in reranked_results 
            if result.get('cohere_score', result['score']) >= quality_threshold
        ]
        
        print(f"📈 Found {len(high_quality)} high-quality tools (score >= {quality_threshold})")
        
        if len(high_quality) >= top_k:
            # We have enough high-quality results
            selected = high_quality[:top_k]
            print("✨ Using only high-quality results")
        else:
            # Fill remaining slots with lower-scored tools to reach top_k
            selected = reranked_results[:top_k]
            print(f"🔄 Using top {top_k} results (mixed quality)")
        
        return selected

    def _get_method_references(self, selected_tools: List[Dict]) -> List[Any]:
        """Convert tool metadata to actual method references"""
        
        method_references = []
        
        for tool in selected_tools:
            try:
                metadata = tool['metadata']
                class_name = metadata['class_name']
                method_name = metadata['tool_name']
                
                # Get the tool instance
                if class_name in self.tool_instances:
                    instance = self.tool_instances[class_name]
                    
                    # Get the method reference
                    if hasattr(instance, method_name):
                        method_ref = getattr(instance, method_name)
                        method_references.append(method_ref)
                    else:
                        print(f"⚠️ Method {method_name} not found in {class_name}")
                else:
                    print(f"⚠️ Class {class_name} not found in tool instances")
                    
            except Exception as e:
                print(f"⚠️ Error getting method reference: {str(e)}")
                continue
        
        return method_references

    def get_tool_info(self, method_references: List[Any]) -> List[Dict]:
        """Get detailed information about retrieved tools (for debugging)"""
        
        tool_info = []
        for method in method_references:
            info = {
                'class_name': method.__self__.__class__.__name__,
                'method_name': method.__name__,
                'description': method.__doc__.split('\n')[0] if method.__doc__ else "No description",
                'module': method.__module__
            }
            tool_info.append(info)
        
        return tool_info

def main():
    """Demo the tool retriever"""
    
    print("🔧 Tool Retriever Demo")
    
    # Initialize retriever
    retriever = ToolRetriever()
    
    # Test queries - realistic user scenarios
    test_queries = [
        "My EC2 instance seems to be running slow, how can I check what's going on?",
        "I need to backup my production database before the maintenance window",
        "Our application is getting too much traffic, I need to scale up the Kubernetes deployment",
        "Users are complaining about slow website response times, can you help me check the load balancer?", 
        "I accidentally deleted some files, can you help me see what's in our S3 storage?",
        "There's an alarm going off in CloudWatch, how do I acknowledge it?",
        "My Lambda function failed last night, I need to see the error logs",
        "The deployment pipeline broke, I need to rollback to the previous version",
        "I can't reach our database server, can you help me check the network connectivity?",
        "I need to rotate the API keys for security compliance",
        "The disk space is running low on our servers, how can I check usage?",
        "Our DNS records need to be updated for the new CDN setup"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve tools
        methods = retriever.retrieve_tools(query, top_k=5)
        
        # Show tool info
        if methods:
            tool_info = retriever.get_tool_info(methods)
            print("\n📋 Retrieved Tools:")
            for i, info in enumerate(tool_info, 1):
                print(f"   {i}. {info['class_name']}.{info['method_name']}")
                print(f"      Description: {info['description']}")
        else:
            print("❌ No tools retrieved")

if __name__ == "__main__":
    main()