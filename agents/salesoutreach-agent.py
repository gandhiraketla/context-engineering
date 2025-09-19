# Context Quarantine Sales Outreach System
# Supervisor pattern with strict context isolation between agents

import json
import os
from datetime import datetime
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class IsolatedAgent(ABC):
    """Base class for agents with strict context quarantine"""
    
    def __init__(self, agent_name: str, system_prompt: str):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
    
    def _create_fresh_llm(self) -> ChatOpenAI:
        """Create fresh LLM instance for each task - no context persistence"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _invoke_llm(self, prompt: str) -> str:
        """Single LLM invocation with fresh context"""
        llm = self._create_fresh_llm()
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    
    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """Simple post-processing: extract JSON object"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in {self.agent_name} response")
        return json.loads(text[start:end])
    
    def _extract_json_array(self, text: str) -> List[Dict[str, Any]]:
        """Simple post-processing: extract JSON array"""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found in {self.agent_name} response")
        return json.loads(text[start:end])
    
    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single task with fresh context"""
        pass

class CompanySearchAgent(IsolatedAgent):
    def __init__(self):
        super().__init__(
            "CompanySearchAgent",
            """You are a B2B company research specialist. Find companies using Perplexity search.
            
Your job: Research companies in the specified domain and geography.
Return structured JSON data only."""
        )
    
    def _perplexity_search(self, query: str) -> str:
        """Use Perplexity for company search"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY not found")
        
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Provide factual company information in structured format."},
                {"role": "user", "content": query}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content or "No results found"
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        domain = task_input["domain"]
        geography = task_input["geography"]
        
        query = f"""Find 10 companies in {domain} industry in {geography}.

Return ONLY a valid JSON array:
[
  {{"name": "Company Name", "size": "startup|mid-size|enterprise"}}
]

Size guidelines: startup (<100 employees), mid-size (100-999), enterprise (1000+)
Return only the JSON array, nothing else."""

        result = self._perplexity_search(query)
        companies = self._extract_json_array(result)
        
        return {
            "success": True,
            "companies": companies[:10]
        }

class MessagingStrategyAgent(IsolatedAgent):
    def __init__(self):
        super().__init__(
            "MessagingStrategyAgent", 
            """You are a B2B messaging strategist. Create compelling messaging frameworks.
            
Your job: Develop messaging strategy based on domain, objective, and tone.
Use your expertise to create persuasive messaging."""
        )
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        domain = task_input["domain"]
        objective = task_input["objective"]
        tone = task_input["tone"]
        
        prompt = f"""Create a messaging strategy for {objective} solutions targeting {domain} industry.
        
Tone: {tone}

Based on your knowledge of {domain} industry, create a compelling messaging framework.

Return ONLY a JSON object:
{{
  "value_proposition": "Clear value proposition addressing {domain} pain points (max 100 words)",
  "key_benefits": ["benefit 1", "benefit 2", "benefit 3"],
  "call_to_action": "Specific CTA (max 20 words)",
  "tone_guidelines": "How to communicate in {tone} tone"
}}

Return only the JSON object, nothing else."""

        result = self._invoke_llm(prompt)
        strategy = self._extract_json_object(result)
        
        return {
            "success": True,
            "strategy": strategy
        }

class CompanyResearchAgent(IsolatedAgent):
    def __init__(self):
        super().__init__(
            "CompanyResearchAgent",
            """You are a company research specialist. Research individual companies using Perplexity.
            
Your job: Research specific companies and identify opportunities.
Focus on business model, challenges, and opportunities."""
        )
    
    def _perplexity_search(self, query: str) -> str:
        """Use Perplexity for company research"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Provide detailed company research in structured format."},
                {"role": "user", "content": query}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content or "No results found"
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        company_name = task_input["company_name"]
        objective = task_input["objective"]
        
        query = f"""Research {company_name} company. Focus on their business model, current challenges, and opportunities related to {objective}.

Return ONLY a JSON object:
{{
  "business_model": "How {company_name} makes money - be specific",
  "current_challenges": ["challenge 1", "challenge 2", "challenge 3"],
  "opportunities": ["opportunity 1 for {objective}", "opportunity 2", "opportunity 3"]
}}

Return only the JSON object, nothing else."""

        result = self._perplexity_search(query)
        research = self._extract_json_object(result)
        
        return {
            "success": True,
            "research": research
        }

class EmailDraftAgent(IsolatedAgent):
    def __init__(self):
        super().__init__(
            "EmailDraftAgent",
            """You are a sales email copywriter. Write personalized emails using provided data.
            
Your job: Create personalized emails that combine messaging strategy with company research.
Make each email specific to the company."""
        )
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        company_name = task_input["company_name"]
        company_research = task_input["company_research"]
        messaging_strategy = task_input["messaging_strategy"]
        tone = task_input["tone"]
        
        prompt = f"""Write a personalized sales email for {company_name}.

COMPANY RESEARCH:
Business Model: {company_research.get('business_model', 'N/A')}
Challenges: {', '.join(company_research.get('current_challenges', []))}
Opportunities: {', '.join(company_research.get('opportunities', []))}

MESSAGING STRATEGY:
Value Proposition: {messaging_strategy.get('value_proposition', '')}
Key Benefits: {', '.join(messaging_strategy.get('key_benefits', []))}
Call to Action: {messaging_strategy.get('call_to_action', '')}

Requirements:
- {tone} tone
- Under 150 words
- Reference specific company details from research
- Use messaging strategy appropriately
- Make it personal and relevant to {company_name}

Return ONLY a JSON object:
{{
  "subject": "Personalized subject line mentioning {company_name}",
  "body": "Complete email body that combines strategy with {company_name} specifics",
  "personalization_elements": ["element 1 showing research", "element 2"]
}}

Return only the JSON object, nothing else."""

        result = self._invoke_llm(prompt)
        email = self._extract_json_object(result)
        
        return {
            "success": True,
            "email": email
        }

class SupervisorAgent:
    """Supervisor that orchestrates workflow with context quarantine"""
    
    def __init__(self):
        self.agents = {
            "company_search": CompanySearchAgent(),
            "messaging_strategy": MessagingStrategyAgent(), 
            "company_research": CompanyResearchAgent(),
            "email_draft": EmailDraftAgent()
        }
    
    def log_delegation_trace(self, from_agent: str, to_agent: str, task: str):
        """Log delegation trace for visibility"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"""
DELEGATION TRACE:
├─ FROM: {from_agent}
├─ TO: {to_agent}
├─ TASK: {task}
├─ TIMESTAMP: {timestamp}
""")
    
    def execute_workflow(self, domain: str, geography: str, objective: str, tone: str) -> Dict[str, Any]:
        """Execute complete workflow with strict context quarantine"""
        
        print("🎯 SUPERVISOR: Starting context quarantine workflow...")
        print(f"Domain: {domain} | Geography: {geography} | Objective: {objective} | Tone: {tone}")
        
        try:
            # Step 1: Search for companies
            self.log_delegation_trace("supervisor", "company_search_agent", f"Find {domain} companies in {geography}")
            
            company_result = self.agents["company_search"].execute_task({
                "domain": domain,
                "geography": geography
            })
            
            if not company_result["success"]:
                return {"success": False, "error": "Company search failed"}
            
            companies = company_result["companies"]
            print(f"✅ Found {len(companies)} companies")
            
            # Step 2: Create messaging strategy
            self.log_delegation_trace("supervisor", "messaging_strategy_agent", f"Create messaging strategy for {objective}")
            
            strategy_result = self.agents["messaging_strategy"].execute_task({
                "domain": domain,
                "objective": objective,
                "tone": tone
            })
            
            if not strategy_result["success"]:
                return {"success": False, "error": "Messaging strategy failed"}
            
            messaging_strategy = strategy_result["strategy"]
            print(f"✅ Created messaging strategy")
            
            # Step 3: Research each company and generate emails
            emails = []
            
            for i, company in enumerate(companies, 1):
                company_name = company["name"]
                print(f"\n[{i}/{len(companies)}] Processing {company_name}...")
                
                # Research company (fresh context)
                self.log_delegation_trace("supervisor", "company_research_agent", f"Research {company_name} for {objective}")
                
                research_result = self.agents["company_research"].execute_task({
                    "company_name": company_name,
                    "objective": objective
                })
                
                if not research_result["success"]:
                    print(f"❌ Research failed for {company_name}")
                    continue
                
                print(f"✅ Research complete for {company_name}")
                
                # Generate email (fresh context)
                self.log_delegation_trace("supervisor", "email_draft_agent", f"Draft email for {company_name}")
                
                email_result = self.agents["email_draft"].execute_task({
                    "company_name": company_name,
                    "company_research": research_result["research"],
                    "messaging_strategy": messaging_strategy,
                    "tone": tone
                })
                
                if not email_result["success"]:
                    print(f"❌ Email draft failed for {company_name}")
                    continue
                
                # Aggregate results
                emails.append({
                    "company_name": company_name,
                    "company_size": company.get("size", "unknown"),
                    "company_research": research_result["research"],
                    "email": email_result["email"]
                })
                
                print(f"✅ Email generated for {company_name}")
            
            # Return comprehensive results
            return {
                "success": True,
                "workflow_summary": {
                    "domain": domain,
                    "geography": geography,
                    "objective": objective,
                    "tone": tone,
                    "companies_found": len(companies),
                    "emails_generated": len(emails)
                },
                "companies": companies,
                "messaging_strategy": messaging_strategy,
                "emails": emails
            }
            
        except Exception as e:
            print(f"💥 SUPERVISOR: Workflow failed: {str(e)}")
            return {"success": False, "error": str(e)}

def run_context_quarantine_demo():
    """Run the context quarantine sales outreach demo"""
    
    print("=== CONTEXT QUARANTINE SALES OUTREACH SYSTEM ===")
    
    supervisor = SupervisorAgent()
    
    # Execute workflow
    results = supervisor.execute_workflow(
        domain="fintech",
        geography="USA",
        objective="cloud cost optimization",
        tone="professional"
    )
    
    if not results.get("success"):
        print(f"Workflow failed: {results.get('error')}")
        return None
    
    # Display results
    print("\n" + "="*80)
    print("📊 WORKFLOW RESULTS")
    print("="*80)
    
    summary = results["workflow_summary"]
    print(f"Domain: {summary['domain']}")
    print(f"Geography: {summary['geography']}")
    print(f"Objective: {summary['objective']}")
    print(f"Companies Found: {summary['companies_found']}")
    print(f"Emails Generated: {summary['emails_generated']}")
    
    print(f"\n💬 MESSAGING STRATEGY:")
    strategy = results["messaging_strategy"]
    print(f"Value Prop: {strategy.get('value_proposition', 'N/A')}")
    print(f"Key Benefits: {', '.join(strategy.get('key_benefits', []))}")
    print(f"CTA: {strategy.get('call_to_action', 'N/A')}")
    
    print(f"\n📧 GENERATED EMAILS:")
    print("="*80)
    
    for i, email_data in enumerate(results["emails"], 1):
        print(f"\n📧 EMAIL {i} - {email_data['company_name']} ({email_data['company_size']})")
        print("-" * 60)
        print(f"Subject: {email_data['email']['subject']}")
        print(f"\nBody:\n{email_data['email']['body']}")
        if email_data['email'].get('personalization_elements'):
            print(f"\nPersonalization: {', '.join(email_data['email']['personalization_elements'])}")
        print("="*80)
    
    print(f"\n🎉 Generated {len(results['emails'])} personalized emails with strict context quarantine!")
    
    return results

# Execute demo
if __name__ == "__main__":
    result = run_context_quarantine_demo()