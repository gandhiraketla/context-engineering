#!/usr/bin/env python3
"""
RAG Context Pruning Demo
========================

Interactive demo comparing two RAG approaches:
1. Plain RAG
2. LLM-based Context Pruning

Required packages:
pip install langchain langchain-community langchain-pinecone pypdf rich openai python-dotenv requests
"""

import os
import warnings
from typing import List, Dict
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Rich console for better formatting
from rich.console import Console
from rich.panel import Panel

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Context Engineering - Context Pruning"

console = Console()

class DeepSeekLLM:
    """DeepSeek API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        
    def invoke(self, prompt: str, tags: List[str] = None) -> str:
        """Generate response using DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"

class RAGDemo:
    """Main RAG demo with LLM-based pruning"""
    
    def __init__(self):
        # Load API keys
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        
        if not all([openai_api_key, pinecone_api_key, deepseek_api_key]):
            missing = []
            if not openai_api_key: missing.append("OPENAI_API_KEY")
            if not pinecone_api_key: missing.append("PINECONE_API_KEY") 
            if not deepseek_api_key: missing.append("DEEPSEEK_API_KEY")
            raise ValueError(f"Missing API keys in .env file: {', '.join(missing)}")
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        if langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = DeepSeekLLM(deepseek_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index_name = "irs-documents"
        self.vectorstore = None
        
        # Pruning prompt template
        self.pruning_prompt = """
        You are an expert at summarizing technical and regulatory documents.

        Your task: Analyze the provided document and produce a condensed version that fully answers or supports the user's specific request. 
        The summary must preserve every fact, condition, and exception that impacts accuracy, while eliminating redundancy and verbosity.

        User's Request:
        {initial_request}

        Document Content:
        {context}

        Summarization Rules:
        1. PRESERVE:
        - All key facts, numbers, and examples relevant to the request
        - Critical conditions, eligibility requirements, exceptions, and limitations
        - Relationships between concepts (logical flow must remain intact)

        2. CONDENSE:
        - Remove repetition, disclaimers, and filler text
        - Replace long phrases with concise, precise wording
        - Compress verbose explanations into shorter, clear sentences

        3. STYLE:
        - Present the summary in a clear, structured format
        - Use bullet points or short paragraphs where appropriate
        - Maintain neutral tone: do not add interpretation, only restate essential content

        Guiding Principle:
        The result should be 50–70% shorter than the original text while retaining 100% of the essential information needed for accuracy.
        """


    def setup_vectorstore(self):
        """Setup Pinecone vectorstore"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.pinecone_index_name not in existing_indexes:
                console.print(f"🔧 Creating Pinecone index: {self.pinecone_index_name}...", style="yellow")
                
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                console.print("✅ Pinecone index created successfully!", style="green")
                
                # Load and index documents since index is new
                self.load_and_index_documents()
            else:
                console.print(f"📋 Using existing Pinecone index: {self.pinecone_index_name}", style="green")
                console.print("📚 Skipping document loading - using existing vectors", style="blue")
            
            # Connect to vectorstore
            self.vectorstore = PineconeVectorStore(
                index_name=self.pinecone_index_name,
                embedding=self.embeddings
            )
            
        except Exception as e:
            console.print(f"❌ Error setting up vectorstore: {e}", style="red")
            raise
    
    def load_and_index_documents(self):
        """Load and index IRS documents - only called when index is newly created"""
        console.print("📥 Loading and indexing IRS documents...", style="yellow")
        
        urls = [
            "https://www.irs.gov/pub/irs-pdf/p17.pdf",   # Your Federal Income Tax
            "https://www.irs.gov/pub/irs-pdf/p970.pdf",  # Tax Benefits for Education
            "https://www.irs.gov/pub/irs-pdf/p463.pdf",  # Travel, Gift, and Car Expenses
            "https://www.irs.gov/pub/irs-pdf/p502.pdf"   # Medical and Dental Expenses
        ]
        
        all_docs = []
        for url in urls:
            console.print(f"  Loading: {url.split('/')[-1]}")
            try:
                loader = PyPDFLoader(url)
                docs = loader.load()
                all_docs.extend(docs)
                console.print(f"    ✅ Loaded {len(docs)} pages", style="green")
            except Exception as e:
                console.print(f"    ❌ Error loading {url}: {e}", style="red")
        
        # Split and index documents
        chunks = self.text_splitter.split_documents(all_docs)
        console.print(f"  📄 Created {len(chunks)} chunks")
        
        # Build vectorstore
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.pinecone_index_name
        )
        console.print("✅ Documents indexed successfully!", style="green")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k relevant chunks"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def diagnostic_retrieval_analysis(self, query: str, k: int = 5):
        """Run detailed analysis of what's being retrieved"""
        console.print(f"\n🔍 RETRIEVAL ANALYSIS FOR: {query}", style="bold yellow")
        console.print("="*80)
        
        # Get documents with scores
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        analysis = {
            'total_chunks': len(docs),
            'total_length': 0,
            'avg_score': 0,
            'key_terms_found': set(),
            'chunk_types': {}
        }
        
        key_terms = ['deduct', 'qualify', 'AGI', '7.5%', 'Schedule A', 'medical expenses', 
                    'includible', 'expenses', 'tax', 'dependent', 'spouse']
        
        for i, (doc, score) in enumerate(docs):
            content = doc.page_content
            analysis['total_length'] += len(content)
            
            console.print(f"\n📄 CHUNK {i+1} | Score: {score:.4f} | Length: {len(content)} chars", 
                         style="blue")
            
            # Show beginning and end of chunk
            console.print("🔹 Beginning:", style="cyan")
            console.print(content[:300] + "..." if len(content) > 300 else content)
            
            if len(content) > 600:
                console.print("\n🔹 Ending:", style="cyan")
                console.print("..." + content[-300:])
            
            # Check for key terms
            found_terms = []
            for term in key_terms:
                if term.lower() in content.lower():
                    found_terms.append(term)
                    analysis['key_terms_found'].add(term)
            
            console.print(f"\n🔑 Key terms found: {found_terms}")
            
            # Identify content type
            content_type = self._identify_content_type(content)
            analysis['chunk_types'][content_type] = analysis['chunk_types'].get(content_type, 0) + 1
            console.print(f"📋 Content type: {content_type}")
            
            console.print("-" * 80)
        
        # Summary analysis
        analysis['avg_score'] = sum(score for _, score in docs) / len(docs) if docs else 0
        analysis['avg_length'] = analysis['total_length'] / len(docs) if docs else 0
        
        console.print(f"\n📊 SUMMARY ANALYSIS", style="bold green")
        console.print(f"Total chunks retrieved: {analysis['total_chunks']}")
        console.print(f"Average similarity score: {analysis['avg_score']:.4f}")
        console.print(f"Average chunk length: {analysis['avg_length']:.0f} characters")
        console.print(f"Total content length: {analysis['total_length']:,} characters")
        console.print(f"Key terms coverage: {len(analysis['key_terms_found'])}/{len(key_terms)} terms found")
        console.print(f"Key terms found: {sorted(list(analysis['key_terms_found']))}")
        console.print(f"Content types: {dict(analysis['chunk_types'])}")
        
        return analysis
    
    def _identify_content_type(self, content: str) -> str:
        """Identify the type of content in a chunk"""
        content_lower = content.lower()
        
        if 'table of contents' in content_lower or 'contents' in content_lower[:100]:
            return 'table_of_contents'
        elif any(term in content_lower for term in ['page', 'fileid:', 'publication']):
            return 'metadata/header'
        elif any(term in content_lower for term in ['what expenses', 'includible', 'deduction']):
            return 'qualifying_criteria'
        elif any(term in content_lower for term in ['can\'t include', 'aren\'t includible', 'exclude']):
            return 'exclusion_criteria'
        elif any(term in content_lower for term in ['schedule a', 'form 1040', 'report']):
            return 'procedural_info'
        elif any(term in content_lower for term in ['introduction', 'publication explains']):
            return 'introduction'
        elif len(content.strip()) < 100:
            return 'fragment'
        else:
            return 'general_content'
    
    def llm_judge_comparison(self, query: str, answers: Dict[str, str], contexts: Dict[str, str]) -> Dict:
        """Use LLM as a judge to compare the two RAG approaches"""
        try:
            judge_prompt = f"""
You are an expert evaluator assessing RAG (Retrieval-Augmented Generation) system outputs. 

QUESTION: {query}

Please evaluate the following two answers based on these criteria:
1. RELEVANCE: How well does the answer address the specific question?
2. ACCURACY: How factually correct is the answer based on the provided context?
3. COMPLETENESS: How thoroughly does the answer cover the question?
4. CLARITY: How clear and well-structured is the answer?

Plain RAG:
{answers['plain']}

LLM Summarized RAG:
{answers['pruned']}

For each answer, provide:
1. A score from 1-10 for each criterion (Relevance, Accuracy, Completeness, Clarity)
2. Brief justification for each score
3. Overall ranking (1st, 2nd) with explanation

Format your response as:
Plain RAG EVALUATION:
- Relevance: X/10 - [justification]
- Accuracy: X/10 - [justification]  
- Completeness: X/10 - [justification]
- Clarity: X/10 - [justification]
- Overall Score: X/40

LLM Summarized RAG EVALUATION:
[same format]

FINAL RANKING:
1st Place: [Answer X] - [reason]
2nd Place: [Answer X] - [reason]

SUMMARY:
[Brief comparison highlighting key differences and why the ranking was chosen]
"""
            
            judge_result = self.llm.invoke(judge_prompt, tags=["llm-judge"])
            
            # Parse the judge result to extract scores
            scores = self._parse_judge_scores(judge_result)
            
            return {
                'evaluation': judge_result,
                'scores': scores,
                'success': True
            }
            
        except Exception as e:
            console.print(f"Warning: Error in LLM judge evaluation: {str(e)}", style="yellow")
            return {
                'evaluation': f"Error during evaluation: {str(e)}",
                'scores': {},
                'success': False
            }
    
    def _parse_judge_scores(self, judge_result: str) -> Dict:
        """Parse LLM judge result to extract numerical scores"""
        try:
            scores = {
                'plain': {'relevance': 0, 'accuracy': 0, 'completeness': 0, 'clarity': 0, 'total': 0},
                'pruned': {'relevance': 0, 'accuracy': 0, 'completeness': 0, 'clarity': 0, 'total': 0}
            }
            
            lines = judge_result.split('\n')
            current_answer = None
            
            for line in lines:
                line = line.strip()
                
                # Identify which answer we're parsing
                if 'ANSWER A' in line.upper():
                    current_answer = 'plain'
                elif 'ANSWER B' in line.upper():
                    current_answer = 'pruned'
                
                # Extract scores
                if current_answer and ':' in line:
                    if 'relevance:' in line.lower():
                        score = self._extract_score_from_line(line)
                        if score: scores[current_answer]['relevance'] = score
                    elif 'accuracy:' in line.lower():
                        score = self._extract_score_from_line(line)
                        if score: scores[current_answer]['accuracy'] = score
                    elif 'completeness:' in line.lower():
                        score = self._extract_score_from_line(line)
                        if score: scores[current_answer]['completeness'] = score
                    elif 'clarity:' in line.lower():
                        score = self._extract_score_from_line(line)
                        if score: scores[current_answer]['clarity'] = score
                    elif 'overall score:' in line.lower():
                        score = self._extract_score_from_line(line)
                        if score: scores[current_answer]['total'] = score
            
            # Calculate totals if not provided
            for approach in scores:
                if scores[approach]['total'] == 0:
                    scores[approach]['total'] = sum([
                        scores[approach]['relevance'],
                        scores[approach]['accuracy'], 
                        scores[approach]['completeness'],
                        scores[approach]['clarity']
                    ])
            
            return scores
            
        except Exception as e:
            console.print(f"Warning: Could not parse judge scores: {str(e)}", style="yellow")
            return {}
    
    def _extract_score_from_line(self, line: str) -> int:
        """Extract numerical score from a line of text"""
        import re
        # Look for patterns like "8/10", "8", "8.5"
        matches = re.findall(r'(\d+(?:\.\d+)?)', line)
        if matches:
            return int(float(matches[0]))
        return 0
    
    def run_comparison(self, query: str, return_results: bool = False):
        """Run comparison of Plain RAG vs LLM Pruning"""
        console.print(Panel(f"🔍 Query: {query}", style="bold blue"))
        
        # Retrieve context
        console.print("🔍 Retrieving relevant chunks...", style="yellow")
        retrieved_chunks = self.retrieve_context(query, k=5)
        original_context = "\n\n".join(retrieved_chunks)
        
        # Display original context (only in console mode)
        if not return_results:
            console.print(Panel(original_context[:1000] + "..." if len(original_context) > 1000 else original_context, 
                              title="📄 Original Context", border_style="white"))
        
        # 1. Plain RAG
        console.print("🤖 Running Plain RAG...", style="yellow")
        plain_answer = self.llm.invoke(
            f"Based on the following context, please answer the question.\n\nContext:\n{original_context}\n\nQuestion: {query}\n\nAnswer:",
            tags=["plain-rag"]
        )
        
        # 2. LLM-based Pruning
        console.print("📝 Running LLM-based Pruning...", style="yellow")
        prompt_pruned_context = self.llm.invoke(
            self.pruning_prompt.format(initial_request=query, context=original_context),
            tags=["llm-pruning-step1"]
        )
        
        if not return_results:
            console.print(Panel(prompt_pruned_context[:1000] + "..." if len(prompt_pruned_context) > 1000 else prompt_pruned_context,
                              title="📝 LLM Pruned Context", border_style="blue"))
        
        pruned_answer = self.llm.invoke(
            f"Based on the following context, please answer the question.\n\nContext:\n{prompt_pruned_context}\n\nQuestion: {query}\n\nAnswer:",
            tags=["llm-pruning-step2"]
        )
        
        # Use LLM-as-a-judge evaluation if returning results
        if return_results:
            console.print("⚖️ Running LLM-as-a-judge evaluation...", style="yellow")
            
            answers = {
                'plain': plain_answer,
                'pruned': pruned_answer
            }
            
            contexts = {
                'original': original_context,
                'pruned': prompt_pruned_context
            }
            
            judge_evaluation = self.llm_judge_comparison(query, answers, contexts)
            
            return {
                'query': query,
                'contexts': {
                    'original': original_context,
                    'pruned': prompt_pruned_context
                },
                'answers': {
                    'plain': plain_answer,
                    'pruned': pruned_answer
                },
                'evaluation': judge_evaluation
            }
        
        # Display answers (console mode only)
        console.print("\n" + "="*80)
        console.print("📋 ANSWERS COMPARISON", style="bold magenta", justify="center")
        console.print("="*80)
        
        console.print(Panel(plain_answer, title="🤖 Plain RAG Answer", border_style="white"))
        console.print(Panel(pruned_answer, title="📝 LLM Pruned Answer", border_style="blue"))
    
    def run_interactive_demo(self):
        """Run interactive demo"""
        console.print(Panel("🚀 RAG Context Pruning Demo", style="bold magenta"))
        console.print("Type 'quit' or 'exit' to stop the demo.", style="yellow")
        console.print("Type 'analyze [question]' to run retrieval analysis.", style="yellow")
        console.print("Type 'compare [question]' to run full comparison.\n", style="yellow")
        
        # Setup vectorstore
        self.setup_vectorstore()
        
        while True:
            try:
                user_input = input("\n💬 Enter your command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '']:
                    console.print("👋 Demo ended. Check LangSmith for detailed metrics!", style="green")
                    break
                
                if user_input.lower().startswith('analyze '):
                    query = user_input[8:].strip()
                    if query:
                        self.diagnostic_retrieval_analysis(query)
                    else:
                        console.print("Please provide a question to analyze. Example: analyze What medical expenses are deductible?", style="red")
                
                elif user_input.lower().startswith('compare '):
                    query = user_input[8:].strip()
                    if query:
                        self.run_comparison(query, return_results=False)
                    else:
                        console.print("Please provide a question to compare. Example: compare What medical expenses are deductible?", style="red")
                
                else:
                    # Default behavior - treat as comparison
                    self.run_comparison(user_input, return_results=False)
                
            except KeyboardInterrupt:
                console.print("\n👋 Demo ended. Check LangSmith for detailed metrics!", style="green")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")

def main():
    """Main function"""
    try:
        demo = RAGDemo()
        demo.run_interactive_demo()
    except Exception as e:
        console.print(f"❌ Setup Error: {str(e)}", style="red")
        console.print("\n🔧 Make sure your .env file contains:", style="yellow")
        console.print("OPENAI_API_KEY=your-key", style="yellow")
        console.print("PINECONE_API_KEY=your-key", style="yellow") 
        console.print("DEEPSEEK_API_KEY=your-key", style="yellow")
        console.print("LANGSMITH_API_KEY=your-key", style="yellow")

if __name__ == "__main__":
    main()