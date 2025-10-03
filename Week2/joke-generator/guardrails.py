# guardrails.py
import re

class SmartGuardrails:
    def __init__(self):
        self.sensitive_categories = {
            'violence': ['kill', 'murder', 'attack', 'shoot', 'bomb', 'war', 'weapon', 'terror', 'assault', 'violence'],
            'death': ['death', 'dead', 'die', 'suicide', 'funeral', 'corpse'],
            'religion': ['god', 'allah', 'jesus', 'buddha', 'religious', 'church', 'mosque', 'temple', 'bible', 'quran'],
            'race': ['race', 'racist', 'ethnic', 'discrimination', 'racial'],
            'sexual': ['sex', 'rape', 'porn', 'nude', 'sexual'],
            'politics': ['politics', 'election', 'government', 'president', 'minister', 'political'],
            'disability': ['disability', 'disabled', 'handicap', 'cripple'],
            'tragedy': ['tragedy', 'disaster', 'accident', 'crash', 'catastrophe']
        }
        
        self.sensitive_patterns = [
            r'\b\d{1,2}/\d{1,2}\b',
            r'\b(19|20)\d{2}\s*(attack|bombing|shooting|tragedy)\b',
            r'\b\w+\s*(attack|bombing|shooting|massacre|terror)\b',
        ]
    
    def check_topic(self, topic):
        if not topic or len(topic.strip()) < 2:
            return False, "⚠️ Please enter a valid topic (at least 2 characters)."
        
        topic_clean = topic.lower().strip()
        
        for category, keywords in self.sensitive_categories.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}\w*\b', topic_clean):
                    return False, self._generate_block_message(topic, category)
        
        for pattern in self.sensitive_patterns:
            if re.search(pattern, topic_clean, re.IGNORECASE):
                return False, self._generate_block_message(topic, 'sensitive_event')
        
        if self._detect_violent_context(topic_clean):
            return False, self._generate_block_message(topic, 'violent_context')
        
        return True, "✅ Topic is safe!"
    
    def _detect_violent_context(self, text):
        violence_words = ['attack', 'kill', 'murder', 'bomb', 'shoot', 'terror']
        location_words = ['hotel', 'school', 'city', 'building', 'tower', 'mall', 'station']
        
        has_violence = any(word in text for word in violence_words)
        has_location = any(word in text for word in location_words)
        
        return has_violence and has_location
    
    def _generate_block_message(self, topic, reason):
        messages = {
            'violence': "This topic involves violence or harmful content.",
            'death': "This topic relates to death or loss.",
            'religion': "Religious topics are sensitive and best avoided.",
            'race': "Racial topics are inappropriate for jokes.",
            'sexual': "Sexual content is not appropriate.",
            'politics': "Political topics can be divisive.",
            'disability': "Disability-related humor can be hurtful.",
            'tragedy': "This relates to tragic events.",
            'sensitive_event': "This appears to reference a sensitive event.",
            'violent_context': "This suggests violent or sensitive content."
        }
        
        return (
            f"🚫 Cannot generate jokes about '{topic}'.\n\n"
            f"{messages.get(reason, 'This topic contains sensitive content.')}\n\n"
            "Let's keep humor light and fun! 😊\n\n"
            "Try: coffee, programming, cats, weather, work, pizza"
        )
    
    def get_safe_topics(self):
        return [
            'coffee', 'pizza', 'cats', 'dogs', 'programming', 
            'work', 'monday', 'weather', 'internet', 'technology'
        ]

guardrails = SmartGuardrails()