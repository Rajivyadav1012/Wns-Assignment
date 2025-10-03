# joke_generator.py
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from guardrails import guardrails

class JokeGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
    
    def generate_joke(self, topic, joke_type="pun"):
        is_safe, message = guardrails.check_topic(topic)
        
        if not is_safe:
            return {
                'success': False,
                'error': message,
                'safe_topics': guardrails.get_safe_topics()
            }
        
        prompt = self._create_prompt(topic, joke_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hilarious comedian who creates clean, family-friendly jokes. Keep jokes short, clever, and appropriate for all ages."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            
            joke_text = response.choices[0].message.content.strip()
            
            if joke_type == "pun":
                return self._parse_pun(joke_text, topic)
            else:
                return self._parse_oneliner(joke_text, topic)
        
        except Exception as e:
            return {
                'success': False,
                'error': f"⚠️ Error generating joke: {str(e)}\n\nPlease check your API key and try again."
            }
    
    def _create_prompt(self, topic, joke_type):
        if joke_type == "pun":
            return f"""Create a short, funny pun joke about "{topic}".

Format your response EXACTLY like this:
Setup: [Your setup question here]
Punchline: [Your punchline here]

Make it clever and family-friendly!"""
        else:
            return f"""Create a funny one-liner joke about "{topic}".

Just give me ONE short, punchy joke. No setup needed, just a clever one-liner.
Make it clean and family-friendly!"""
    
    def _parse_pun(self, text, topic):
        try:
            lines = text.split('\n')
            setup = ""
            punchline = ""
            
            for line in lines:
                if line.strip().lower().startswith('setup:'):
                    setup = line.split(':', 1)[1].strip()
                elif line.strip().lower().startswith('punchline:'):
                    punchline = line.split(':', 1)[1].strip()
            
            if setup and punchline:
                return {
                    'success': True,
                    'type': 'pun',
                    'setup': setup,
                    'punchline': punchline,
                    'topic': topic
                }
            else:
                return {
                    'success': True,
                    'type': 'oneliner',
                    'joke': text,
                    'topic': topic
                }
        except:
            return {
                'success': True,
                'type': 'oneliner',
                'joke': text,
                'topic': topic
            }
    
    def _parse_oneliner(self, text, topic):
        joke = text.strip()
        
        prefixes = ['joke:', 'here\'s a joke:', 'one-liner:']
        for prefix in prefixes:
            if joke.lower().startswith(prefix):
                joke = joke[len(prefix):].strip()
        
        return {
            'success': True,
            'type': 'oneliner',
            'joke': joke,
            'topic': topic
        }

joke_generator = JokeGenerator()