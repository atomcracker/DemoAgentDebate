import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import AzureOpenAI
import requests
import json
# Triggering new deployment
app = Flask(__name__)
CORS(app)

# HARDCODED API KEYS - Replace with your actual values
AZURE_OPENAI_ENDPOINT = "https://paperdemo.openai.azure.com/"
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# Azure AI Studio endpoints - Replace with your actual values
MISTRAL_ENDPOINT = "https://jakko-azure-certificate-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

LLAMA_ENDPOINT = "https://jakko-azure-certificate-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

class Agent:
    def __init__(self, name, model, system_prompt):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def update_model(self, new_model):
        self.model = new_model
    
    def update_system_prompt(self, new_prompt):
        self.system_prompt = new_prompt
    
    def call_model(self, messages, temperature=0.7, max_tokens=1000):
        """Call the appropriate model"""
        if self.model.startswith("gpt"):
            # Use Azure OpenAI
            try:
                response = azure_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"
        
        elif self.model.startswith("claude"):
            # Use Claude via Azure AI Studio
            try:
                system_msg = ""
                user_msg = ""
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    elif msg["role"] == "user":
                        user_msg += msg["content"] + "\n"
                
                payload = {
                    "model": "mistral-7b",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_msg,
                    "messages": [{"role": "user", "content": user_msg.strip()}]
                }
                
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    return response.json()["content"][0]["text"]
                else:
                    return f"Claude Error: {response.status_code} - {response.text}"
                    
            except Exception as e:
                return f"Claude Error: {str(e)}"
        
        elif self.model.startswith("llama"):
            # Use Llama via Azure AI Studio
            try:
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n\n"
                
                payload = {
                    "model": "llama3-8b",
                    "prompt": prompt.strip(),
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {LLAMA_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(LLAMA_ENDPOINT, json=payload, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    return f"Llama Error: {response.status_code} - {response.text}"
                    
            except Exception as e:
                return f"Llama Error: {str(e)}"
        
        return f"Unknown model: {self.model}"
    
    def think(self, user_prompt, other_agents_responses=None):
        """Agent thinks about the prompt"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        
        if other_agents_responses:
            other_responses_text = "\n\nOther agents' responses:\n"
            for agent_name, response in other_agents_responses.items():
                if agent_name != self.name:
                    other_responses_text += f"\n{agent_name}: {response}\n"
            messages.append({"role": "user", "content": other_responses_text})
        
        try:
            agent_response = self.call_model(messages, temperature=0.7, max_tokens=1000)
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            return agent_response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def vote(self, responses):
        """Agent votes on the best response"""
        voting_prompt = f"""You are {self.name}. Please analyze the following responses and vote for the BEST one. 
        Consider clarity, accuracy, helpfulness, and completeness.
        
        Responses to evaluate:
        """
        
        for i, (agent_name, response) in enumerate(responses.items(), 1):
            voting_prompt += f"\n{i}. {agent_name}: {response}\n"
        
        voting_prompt += "\nPlease respond with ONLY the number (1, 2, 3, or 4) of the best response."
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": voting_prompt}
            ]
            vote = self.call_model(messages, temperature=0.3, max_tokens=10)
            return vote.strip()
        except Exception as e:
            return "1"  # Default vote

class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.default_models = ["gpt-4", "gpt-35-turbo", "claude-3-sonnet", "llama3"]
        self.default_prompts = [
            "You are a logical and analytical agent. Focus on facts, data, and systematic thinking.",
            "You are a creative and innovative agent. Think outside the box and consider unconventional solutions.",
            "You are a practical and implementation-focused agent. Consider real-world constraints and feasibility.",
            "You are a critical and skeptical agent. Question assumptions and identify potential problems."
        ]
        self.initialize_agents()
    
    def initialize_agents(self):
        agent_names = ["Analyst", "Innovator", "Implementer", "Critic"]
        for i, name in enumerate(agent_names):
            self.agents[name] = Agent(
                name=name,
                model=self.default_models[i],
                system_prompt=self.default_prompts[i]
            )
    
    def run_debate(self, user_prompt, debate_rounds=2):
        """Run the multi-agent debate system"""
        all_responses = {}
        
        # Initial round
        print(f"Round 1: Initial responses")
        for agent_name, agent in self.agents.items():
            response = agent.think(user_prompt)
            all_responses[agent_name] = response
            print(f"{agent_name}: {response[:100]}...")
        
        # Debate rounds
        for round_num in range(2, debate_rounds + 1):
            print(f"\nRound {round_num}: Debate")
            new_responses = {}
            
            for agent_name, agent in self.agents.items():
                other_responses = {name: resp for name, resp in all_responses.items() if name != agent_name}
                response = agent.think(user_prompt, other_responses)
                new_responses[agent_name] = response
                print(f"{agent_name}: {response[:100]}...")
            
            all_responses = new_responses
        
        return all_responses
    
    def vote_on_responses(self, responses):
        """All agents vote on the best response"""
        votes = {}
        for agent_name, agent in self.agents.items():
            vote = agent.vote(responses)
            votes[agent_name] = vote
            print(f"{agent_name} voted for: {vote}")
        
        # Count votes
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Find the winning response
        winning_vote = max(vote_counts, key=vote_counts.get)
        agent_names = list(responses.keys())
        winning_agent = agent_names[int(winning_vote) - 1]
        
        return winning_agent, responses[winning_agent], votes, vote_counts

# Initialize the multi-agent system
multi_agent_system = MultiAgentSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get current agent configurations"""
    agents_info = {}
    for name, agent in multi_agent_system.agents.items():
        agents_info[name] = {
            'model': agent.model,
            'system_prompt': agent.system_prompt
        }
    return jsonify(agents_info)

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get all available models"""
    return jsonify({
        "azure_openai": ["gpt-4", "gpt-35-turbo", "gpt-4o"],
        "azure_ai_catalog": ["claude-3-sonnet", "llama3", "mistral"]
    })

@app.route('/api/agents/<agent_name>/model', methods=['PUT'])
def update_agent_model(agent_name):
    """Update an agent's model"""
    data = request.get_json()
    new_model = data.get('model')
    
    if agent_name not in multi_agent_system.agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    multi_agent_system.agents[agent_name].update_model(new_model)
    return jsonify({'message': f'Updated {agent_name} model to {new_model}'})

@app.route('/api/agents/<agent_name>/prompt', methods=['PUT'])
def update_agent_prompt(agent_name):
    """Update an agent's system prompt"""
    data = request.get_json()
    new_prompt = data.get('prompt')
    
    if agent_name not in multi_agent_system.agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    multi_agent_system.agents[agent_name].update_system_prompt(new_prompt)
    return jsonify({'message': f'Updated {agent_name} system prompt'})

@app.route('/api/debate', methods=['POST'])
def run_debate():
    """Run the multi-agent debate system"""
    data = request.get_json()
    user_prompt = data.get('prompt')
    debate_rounds = data.get('debate_rounds', 2)
    
    if not user_prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        # Run debate
        responses = multi_agent_system.run_debate(user_prompt, debate_rounds)
        
        # Vote on responses
        winning_agent, winning_response, votes, vote_counts = multi_agent_system.vote_on_responses(responses)
        
        return jsonify({
            'user_prompt': user_prompt,
            'debate_rounds': debate_rounds,
            'all_responses': responses,
            'winning_agent': winning_agent,
            'winning_response': winning_response,
            'votes': votes,
            'vote_counts': vote_counts
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Simple Multi-Agent Debate System")
    print("🌐 Server will be available at: http://0.0.0.0:5000")
    print("🤖 Available models: OpenAI GPT, Claude, Llama")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
