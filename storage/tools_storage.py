import os
import inspect
import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import tools from the tools package
from tools import (
    EC2AndKubernetesTools,
    DatabaseTools,
    MonitoringTools,
    NetworkingTools,
    UtilityTools
)

# Load environment from root directory
load_dotenv()

class IndividualToolStorage:
    """Individual tool storage - one chunk per tool for precise Tool Loadout"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index("agent-tools")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("✅ Individual Tool Storage initialized")

    def extract_individual_tools(self, tool_classes):
        """Extract each tool as individual item"""
        print("🔍 Extracting individual tools...")
        
        individual_tools = []
        
        for cls in tool_classes:
            class_name = cls.__class__.__name__
            category = self._get_category(class_name)
            
            # Get each method individually
            for method_name in dir(cls):
                if not method_name.startswith('_'):
                    method = getattr(cls, method_name)
                    if callable(method):
                        try:
                            # Extract method details
                            doc = inspect.getdoc(method) or "No description available"
                            description = doc.split('\n')[0].strip()
                            
                            # Get parameters
                            sig = inspect.signature(method)
                            params = []
                            for param_name, param in sig.parameters.items():
                                if param_name != 'self':
                                    param_str = param_name
                                    if param.annotation != inspect.Parameter.empty:
                                        param_str += f": {param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}"
                                    params.append(param_str)
                            
                            # Get return type
                            return_type = "Unknown"
                            if sig.return_annotation != inspect.Signature.empty:
                                return_type = sig.return_annotation.__name__ if hasattr(sig.return_annotation, '__name__') else str(sig.return_annotation)
                            
                            # Create individual tool entry
                            tool = {
                                'tool_name': method_name,
                                'class_name': class_name,
                                'category': category,
                                'description': description,
                                'parameters': params,
                                'return_type': return_type
                            }
                            
                            individual_tools.append(tool)
                            
                        except Exception as e:
                            print(f"⚠️ Skipping {class_name}.{method_name}: {str(e)}")
                            continue
            
            print(f"  ✅ Extracted tools from {class_name}")
        
        print(f"📊 Total individual tools extracted: {len(individual_tools)}")
        return individual_tools

    def create_individual_chunks(self, tools):
        """Create one chunk per tool"""
        print("📦 Creating individual chunks (one per tool)...")
        
        chunks = []
        
        for tool in tools:
            # Create focused embedding text for each tool
            embedding_text = f"""
            Tool: {tool['tool_name']}
            Class: {tool['class_name']}
            Category: {tool['category']}
            Description: {tool['description']}
            Parameters: {', '.join(tool['parameters'])}
            Returns: {tool['return_type']}
            """
            
            # Generate embedding for this specific tool
            embedding = self.model.encode(embedding_text.strip()).tolist()
            
            # Create individual chunk
            chunk = {
                'id': f"tool_{tool['tool_name']}_{uuid.uuid4().hex[:8]}",
                'values': embedding,
                'metadata': {
                    'tool_name': tool['tool_name'],
                    'class_name': tool['class_name'], 
                    'category': tool['category'],
                    'description': tool['description'],
                    'parameters': tool['parameters'],
                    'return_type': tool['return_type'],
                    'chunk_type': 'individual_tool'
                }
            }
            
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} individual tool chunks")
        return chunks

    def store_chunks(self, chunks):
        """Store individual chunks in Pinecone"""
        print(f"🚀 Storing {len(chunks)} individual tool chunks in Pinecone...")
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                vectors.append({
                    'id': chunk['id'],
                    'values': chunk['values'],
                    'metadata': chunk['metadata']
                })
            
            # Upload in batches of 20
            batch_size = 20
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"  ✅ Uploaded batch {i//batch_size + 1}: {len(batch)} tools")
            
            print(f"🎉 Successfully stored {len(chunks)} individual tools in Pinecone!")
            
        except Exception as e:
            print(f"❌ Error storing chunks: {str(e)}")

    def get_stats(self):
        """Display index statistics"""
        try:
            stats = self.index.describe_index_stats()
            print(f"\n📊 Pinecone Index Stats:")
            print(f"   Total vectors: {stats['total_vector_count']}")
            print(f"   Dimension: {stats['dimension']}")
            print(f"   ✅ Ready for precise Tool Loadout!")
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")

    def run(self, tool_classes=None):
        """Complete process to store individual tools"""
        print("🚀 Starting individual tool storage process...\n")
        
        # Use provided classes or create default instances
        if tool_classes is None:
            tool_classes = [
                EC2AndKubernetesTools(),
                DatabaseTools(),
                MonitoringTools(),
                NetworkingTools(),
                UtilityTools()
            ]
        
        # Step 1: Extract individual tools
        tools = self.extract_individual_tools(tool_classes)
        
        # Step 2: Create individual chunks  
        chunks = self.create_individual_chunks(tools)
        
        # Step 3: Store in Pinecone
        self.store_chunks(chunks)
        
        # Step 4: Show stats
        self.get_stats()
        
        print(f"\n🎉 Complete! Stored {len(chunks)} individual tools for precise Tool Loadout")

    def _get_category(self, class_name):
        """Get category from class name"""
        categories = {
            'EC2AndKubernetesTools': 'Infrastructure',
            'DatabaseTools': 'Database',
            'MonitoringTools': 'Monitoring',
            'NetworkingTools': 'Networking', 
            'UtilityTools': 'Utility'
        }
        return categories.get(class_name, 'Unknown')

def main():
    """Main function to run the storage process"""
    
    print("🔧 Individual Tool Storage - VSCode Version")
    print("📁 Loading tools from tools/ directory...")
    
    # Initialize and run storage
    storage = IndividualToolStorage()
    storage.run()

if __name__ == "__main__":
    main()