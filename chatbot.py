"""
Sentiment Analysis Chatbot
A production-ready chatbot with statement and conversation-level sentiment analysis.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re
import json
import os
from pathlib import Path


@dataclass
class Message:
    """Represents a single message in the conversation."""
    speaker: str
    text: str
    sentiment: str = ""
    sentiment_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert message to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ConversationSummary:
    """Encapsulates conversation-level analysis results."""
    overall_sentiment: str
    average_score: float
    sentiment_distribution: Dict[str, int]
    trend_description: str
    total_messages: int
    
    def to_dict(self) -> dict:
        """Convert summary to dictionary for JSON serialization."""
        return asdict(self)


class SentimentAnalyzer:
    """
    Lexicon-based sentiment analyzer with context awareness.
    Uses weighted keyword matching and negation handling.
    """
    
    def __init__(self):
        # Positive keywords with weights
        self.positive_words = {
            'excellent': 2.0, 'amazing': 2.0, 'fantastic': 2.0, 'wonderful': 2.0,
            'great': 1.5, 'good': 1.0, 'happy': 1.5, 'love': 2.0, 'best': 2.0,
            'pleased': 1.5, 'satisfied': 1.5, 'thank': 1.0, 'thanks': 1.0,
            'appreciate': 1.5, 'better': 1.0, 'perfect': 2.0, 'awesome': 2.0,
            'outstanding': 2.0, 'brilliant': 2.0, 'helpful': 1.5, 'efficient': 1.0,
            'enjoyable': 1.5, 'impressive': 1.5, 'positive': 1.0, 'recommend': 1.5
        }
        
        # Negative keywords with weights
        self.negative_words = {
            'terrible': -2.0, 'awful': -2.0, 'horrible': -2.0, 'worst': -2.0,
            'bad': -1.0, 'poor': -1.5, 'disappointing': -1.5, 'disappointed': -1.5,
            'hate': -2.0, 'angry': -1.5, 'frustrated': -1.5, 'annoying': -1.5,
            'useless': -2.0, 'waste': -1.5, 'problem': -1.0, 'issue': -1.0,
            'fail': -1.5, 'failed': -1.5, 'broken': -1.5, 'slow': -1.0,
            'difficult': -1.0, 'confusing': -1.0, 'unhappy': -1.5, 'dissatisfied': -1.5,
            'concern': -0.5, 'worry': -1.0, 'unfortunately': -1.0, 'negative': -1.0
        }
        
        # Negation words that flip sentiment
        self.negations = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                         'nowhere', 'hardly', 'barely', 'scarcely', "n't", 'cannot'}
        
        # Intensifiers
        self.intensifiers = {'very': 1.5, 'really': 1.5, 'extremely': 2.0, 
                           'absolutely': 2.0, 'completely': 1.8, 'totally': 1.8}
    
    def _tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase tokens."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _check_negation(self, tokens: List[str], index: int, window: int = 3) -> bool:
        """Check if a word is preceded by a negation within a window."""
        start = max(0, index - window)
        return any(token in self.negations for token in tokens[start:index])
    
    def _get_intensifier(self, tokens: List[str], index: int) -> float:
        """Get intensifier multiplier if present before the word."""
        if index > 0 and tokens[index - 1] in self.intensifiers:
            return self.intensifiers[tokens[index - 1]]
        return 1.0
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of text.
        
        Returns:
            Tuple of (sentiment_label, score)
            Score ranges: < -0.15 = Negative, -0.15 to 0.15 = Neutral, > 0.15 = Positive
        """
        if not text.strip():
            return "Neutral", 0.0
        
        tokens = self._tokenize(text)
        score = 0.0
        
        for i, token in enumerate(tokens):
            # Check for positive words
            if token in self.positive_words:
                word_score = self.positive_words[token]
                intensifier = self._get_intensifier(tokens, i)
                is_negated = self._check_negation(tokens, i)
                
                # Apply negation and intensifier
                word_score *= intensifier
                if is_negated:
                    word_score *= -1
                score += word_score
            
            # Check for negative words
            elif token in self.negative_words:
                word_score = self.negative_words[token]
                intensifier = self._get_intensifier(tokens, i)
                is_negated = self._check_negation(tokens, i)
                
                # Apply negation and intensifier
                word_score *= intensifier
                if is_negated:
                    word_score *= -1
                score += word_score
        
        # Normalize by number of tokens
        if len(tokens) > 0:
            score = score / len(tokens)
        
        # Classify sentiment with adjusted thresholds
        if score > 0.15:
            sentiment = "Positive"
        elif score < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return sentiment, score


class ResponseGenerator:
    """Generates dynamic, context-aware chatbot responses based on sentiment and conversation history."""
    
    def __init__(self):
        self.conversation_count = 0
        self.last_sentiment = None
        self.sentiment_streak = 0
        
        # Expanded response templates with more variety
        self.response_templates = {
            'Positive': {
                'acknowledgment': [
                    "That's great to hear!",
                    "I'm really glad about that!",
                    "Wonderful!",
                    "That's fantastic!",
                    "I'm pleased to hear that!"
                ],
                'follow_up': [
                    "What else can I help you with?",
                    "Is there anything else you'd like to discuss?",
                    "How else may I assist you today?",
                    "What would you like to explore next?",
                    "Feel free to share more!"
                ]
            },
            'Negative': {
                'acknowledgment': [
                    "I'm sorry to hear that.",
                    "I understand your frustration.",
                    "That must be disappointing.",
                    "I apologize for that experience.",
                    "I hear your concern."
                ],
                'action': [
                    "Let me help resolve this for you.",
                    "I'll make sure this gets addressed.",
                    "What can I do to make this right?",
                    "How can I assist you with this issue?",
                    "Let's work together to fix this."
                ]
            },
            'Neutral': {
                'acknowledgment': [
                    "I understand.",
                    "I see.",
                    "Got it.",
                    "Noted.",
                    "Okay."
                ],
                'follow_up': [
                    "What else would you like to know?",
                    "How can I help you further?",
                    "Is there anything specific I can assist with?",
                    "What would you like to discuss?",
                    "Tell me more about what you need."
                ]
            }
        }
        
        # Empathy responses for consecutive negative sentiment
        self.empathy_responses = [
            "I really want to help turn this around for you.",
            "Your feedback is important to us, and I'm committed to helping.",
            "I appreciate you sharing these concerns with me.",
            "Let's focus on finding a solution together."
        ]
        
        # Encouraging responses for improving sentiment
        self.encouragement_responses = [
            "I'm glad things are looking up!",
            "It sounds like we're making progress!",
            "That's a positive turn!",
            "I'm happy to hear things are getting better!"
        ]
        
        # Greeting responses
        self.greetings = [
            "Hello! How can I assist you today?",
            "Hi there! What brings you here?",
            "Welcome! I'm here to help. What's on your mind?",
            "Greetings! How may I support you today?",
            "Hey! What can I do for you?"
        ]
    
    def generate(self, sentiment: str, sentiment_score: float = 0.0, 
                 user_message: str = "", is_first: bool = False) -> str:
        """
        Generate dynamic response based on sentiment, context, and conversation history.
        
        Args:
            sentiment: Current message sentiment (Positive/Negative/Neutral)
            sentiment_score: Numeric sentiment score
            user_message: The actual user message for context
            is_first: Whether this is the first message
        """
        import random
        
        if is_first:
            return random.choice(self.greetings)
        
        self.conversation_count += 1
        
        # Track sentiment streaks for context-aware responses
        if sentiment == self.last_sentiment:
            self.sentiment_streak += 1
        else:
            self.sentiment_streak = 1
        self.last_sentiment = sentiment
        
        # Generate contextual response
        response = self._build_contextual_response(
            sentiment, sentiment_score, user_message, random
        )
        
        return response
    
    def _build_contextual_response(self, sentiment: str, score: float, 
                                   user_message: str, random) -> str:
        """Build a contextual response based on multiple factors."""
        
        # Handle negative sentiment with empathy
        if sentiment == "Negative":
            # Show extra empathy for consecutive negative messages
            if self.sentiment_streak >= 2:
                ack = random.choice(self.empathy_responses)
            else:
                ack = random.choice(self.response_templates['Negative']['acknowledgment'])
            
            action = random.choice(self.response_templates['Negative']['action'])
            return f"{ack} {action}"
        
        # Handle positive sentiment
        elif sentiment == "Positive":
            # Extra enthusiasm for strong positive sentiment
            if score > 0.5:
                ack = random.choice(self.response_templates['Positive']['acknowledgment']) + " 😊"
            else:
                ack = random.choice(self.response_templates['Positive']['acknowledgment'])
            
            # Encourage if sentiment improved from negative
            if self.last_sentiment == "Negative" and self.conversation_count > 1:
                return f"{random.choice(self.encouragement_responses)} {random.choice(self.response_templates['Positive']['follow_up'])}"
            
            follow_up = random.choice(self.response_templates['Positive']['follow_up'])
            return f"{ack} {follow_up}"
        
        # Handle neutral sentiment
        else:
            # Be more engaging for neutral responses
            if self.conversation_count <= 2:
                ack = random.choice(self.response_templates['Neutral']['acknowledgment'])
                follow_up = random.choice(self.response_templates['Neutral']['follow_up'])
                return f"{ack} {follow_up}"
            else:
                # Vary the response structure
                if random.random() < 0.5:
                    return random.choice(self.response_templates['Neutral']['follow_up'])
                else:
                    ack = random.choice(self.response_templates['Neutral']['acknowledgment'])
                    follow_up = random.choice(self.response_templates['Neutral']['follow_up'])
                    return f"{ack} {follow_up}"


class ConversationStorage:
    """Handles persistence of conversation history and sentiment analysis."""
    
    def __init__(self, storage_dir: str = "conversations"):
        """
        Initialize storage manager.
        
        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_conversation(self, messages: List[Message], summary: ConversationSummary) -> str:
        """
        Save conversation to JSON file.
        
        Args:
            messages: List of conversation messages
            summary: Conversation summary with sentiment analysis
        
        Returns:
            Path to saved file
        """
        # Prepare conversation data
        conversation_data = {
            "session_id": self.current_session_id,
            "start_time": messages[0].timestamp if messages else datetime.now().isoformat(),
            "end_time": messages[-1].timestamp if messages else datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": [msg.to_dict() for msg in messages],
            "sentiment_analysis": {
                "summary": summary.to_dict(),
                "statement_level": [
                    {
                        "message_number": i + 1,
                        "text": msg.text,
                        "sentiment": msg.sentiment,
                        "sentiment_score": msg.sentiment_score,
                        "timestamp": msg.timestamp
                    }
                    for i, msg in enumerate(messages)
                    if msg.speaker == "User"
                ]
            }
        }
        
        # Save to JSON file
        filename = f"conversation_{self.current_session_id}.json"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            return str(filepath)
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return ""
    
    def save_conversation_txt(self, messages: List[Message], summary: ConversationSummary) -> str:
        """
        Save conversation to human-readable text file.
        
        Args:
            messages: List of conversation messages
            summary: Conversation summary
        
        Returns:
            Path to saved file
        """
        filename = f"conversation_{self.current_session_id}.txt"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 70 + "\n")
                f.write("SENTIMENT ANALYSIS CHATBOT - CONVERSATION TRANSCRIPT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Session ID: {self.current_session_id}\n")
                if messages:
                    f.write(f"Start Time: {messages[0].timestamp}\n")
                    f.write(f"End Time: {messages[-1].timestamp}\n")
                f.write(f"Total Messages: {len(messages)}\n\n")
                
                # Conversation
                f.write("=" * 70 + "\n")
                f.write("CONVERSATION\n")
                f.write("=" * 70 + "\n\n")
                
                for msg in messages:
                    f.write(f"[{msg.timestamp}] {msg.speaker}: {msg.text}\n")
                    if msg.speaker == "User" and msg.sentiment:
                        f.write(f"  → Sentiment: {msg.sentiment} (score: {msg.sentiment_score:.3f})\n")
                    f.write("\n")
                
                # Statement-level analysis
                user_messages = [msg for msg in messages if msg.speaker == "User"]
                if user_messages:
                    f.write("=" * 70 + "\n")
                    f.write("STATEMENT-LEVEL SENTIMENT ANALYSIS (Tier 2)\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for i, msg in enumerate(user_messages, 1):
                        f.write(f"Message {i}:\n")
                        f.write(f"User: \"{msg.text}\"\n")
                        f.write(f"→ Sentiment: {msg.sentiment} (score: {msg.sentiment_score:.3f})\n\n")
                
                # Conversation-level analysis
                f.write("=" * 70 + "\n")
                f.write("CONVERSATION-LEVEL SENTIMENT ANALYSIS (Tier 1)\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Overall Sentiment: {summary.overall_sentiment}\n")
                f.write(f"Average Score: {summary.average_score:.3f}\n\n")
                f.write("Sentiment Distribution:\n")
                for sentiment, count in summary.sentiment_distribution.items():
                    percentage = (count / summary.total_messages * 100) if summary.total_messages > 0 else 0
                    f.write(f"  {sentiment}: {count} messages ({percentage:.1f}%)\n")
                f.write(f"\nTrend Analysis: {summary.trend_description}\n")
                f.write(f"Total User Messages: {summary.total_messages}\n")
            
            return str(filepath)
        except Exception as e:
            print(f"Error saving text file: {e}")
            return ""
    
    def load_conversation(self, session_id: str) -> dict:
        """
        Load conversation from JSON file.
        
        Args:
            session_id: Session ID to load
        
        Returns:
            Dictionary with conversation data
        """
        filename = f"conversation_{session_id}.json"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Conversation {session_id} not found")
            return {}
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return {}
    
    def list_conversations(self) -> List[str]:
        """List all saved conversation session IDs."""
        try:
            json_files = list(self.storage_dir.glob("conversation_*.json"))
            session_ids = [f.stem.replace("conversation_", "") for f in json_files]
            return sorted(session_ids, reverse=True)  # Most recent first
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []
    
    def get_storage_info(self) -> dict:
        """Get information about stored conversations."""
        conversations = self.list_conversations()
        return {
            "storage_directory": str(self.storage_dir.absolute()),
            "total_conversations": len(conversations),
            "recent_sessions": conversations[:5]  # 5 most recent
        }


class SentimentChatbot:
    """Main chatbot class orchestrating conversation and analysis."""
    
    def __init__(self, storage_dir: str = "conversations"):
        self.conversation_history: List[Message] = []
        self.analyzer = SentimentAnalyzer()
        self.response_generator = ResponseGenerator()
        self.storage = ConversationStorage(storage_dir)
        self.is_running = False
    
    def process_user_message(self, user_input: str) -> str:
        """Process user message, analyze sentiment, and generate response."""
        # Analyze sentiment
        sentiment, score = self.analyzer.analyze(user_input)
        
        # Store user message with sentiment
        user_msg = Message(
            speaker="User",
            text=user_input,
            sentiment=sentiment,
            sentiment_score=score
        )
        self.conversation_history.append(user_msg)
        
        # Generate bot response with context
        is_first = len(self.conversation_history) == 1
        bot_response = self.response_generator.generate(
            sentiment=sentiment,
            sentiment_score=score,
            user_message=user_input,
            is_first=is_first
        )
        
        # Store bot message (no sentiment analysis for bot)
        bot_msg = Message(
            speaker="Chatbot",
            text=bot_response
        )
        self.conversation_history.append(bot_msg)
        
        return bot_response
    
    def get_user_messages(self) -> List[Message]:
        """Get only user messages from history."""
        return [msg for msg in self.conversation_history if msg.speaker == "User"]
    
    def analyze_conversation(self) -> ConversationSummary:
        """Generate comprehensive conversation-level analysis."""
        user_messages = self.get_user_messages()
        
        if not user_messages:
            return ConversationSummary(
                overall_sentiment="Neutral",
                average_score=0.0,
                sentiment_distribution={'Positive': 0, 'Neutral': 0, 'Negative': 0},
                trend_description="No messages to analyze",
                total_messages=0
            )
        
        # Calculate statistics
        total_score = sum(msg.sentiment_score for msg in user_messages)
        avg_score = total_score / len(user_messages)
        
        # Count sentiment distribution
        distribution = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for msg in user_messages:
            distribution[msg.sentiment] += 1
        
        # Determine overall sentiment with adjusted thresholds
        if avg_score > 0.15:
            overall = "Positive"
        elif avg_score < -0.15:
            overall = "Negative"
        else:
            overall = "Neutral"
        
        # Analyze trend
        trend = self._analyze_trend(user_messages)
        
        return ConversationSummary(
            overall_sentiment=overall,
            average_score=avg_score,
            sentiment_distribution=distribution,
            trend_description=trend,
            total_messages=len(user_messages)
        )
    
    def _analyze_trend(self, messages: List[Message]) -> str:
        """Analyze sentiment trend across conversation."""
        if len(messages) < 2:
            return "Insufficient data for trend analysis"
        
        # Split into first and second half
        mid = len(messages) // 2
        first_half = messages[:mid]
        second_half = messages[mid:]
        
        first_avg = sum(m.sentiment_score for m in first_half) / len(first_half)
        second_avg = sum(m.sentiment_score for m in second_half) / len(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 0.3:
            return "Improving - sentiment became more positive over time"
        elif diff < -0.3:
            return "Declining - sentiment became more negative over time"
        else:
            return "Stable - sentiment remained consistent throughout"
    
    def print_statement_analysis(self):
        """Print statement-level sentiment analysis (Tier 2)."""
        print("\n" + "="*70)
        print("STATEMENT-LEVEL SENTIMENT ANALYSIS (Tier 2)")
        print("="*70)
        
        user_messages = self.get_user_messages()
        
        for i, msg in enumerate(user_messages, 1):
            print(f"\nMessage {i}:")
            print(f"User: \"{msg.text}\"")
            print(f"→ Sentiment: {msg.sentiment} (score: {msg.sentiment_score:.3f})")
    
    def print_conversation_analysis(self):
        """Print conversation-level analysis (Tier 1)."""
        summary = self.analyze_conversation()
        
        print("\n" + "="*70)
        print("CONVERSATION-LEVEL SENTIMENT ANALYSIS (Tier 1)")
        print("="*70)
        print(f"\nOverall Sentiment: {summary.overall_sentiment}")
        print(f"Average Score: {summary.average_score:.3f}")
        print(f"\nSentiment Distribution:")
        for sentiment, count in summary.sentiment_distribution.items():
            percentage = (count / summary.total_messages * 100) if summary.total_messages > 0 else 0
            print(f"  {sentiment}: {count} messages ({percentage:.1f}%)")
        print(f"\nTrend Analysis: {summary.trend_description}")
        print(f"Total User Messages: {summary.total_messages}")
        
        return summary
    
    def save_conversation(self) -> Tuple[str, str]:
        """
        Save conversation to files.
        
        Returns:
            Tuple of (json_path, txt_path)
        """
        if not self.get_user_messages():
            return "", ""
        
        summary = self.analyze_conversation()
        
        # Save in both formats
        json_path = self.storage.save_conversation(self.conversation_history, summary)
        txt_path = self.storage.save_conversation_txt(self.conversation_history, summary)
        
        return json_path, txt_path
    
    def run(self):
        """Run the interactive chatbot."""
        self.is_running = True
        
        print("="*70)
        print("SENTIMENT ANALYSIS CHATBOT")
        print("="*70)
        print("\nWelcome! I'm here to chat with you.")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        # Initial greeting
        greeting = self.response_generator.generate('greeting', is_first=True)
        print(f"Chatbot: {greeting}\n")
        
        while self.is_running:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nChatbot: Thank you for chatting! Generating analysis...\n")
                    break
                
                # Process message and get response
                bot_response = self.process_user_message(user_input)
                print(f"Chatbot: {bot_response}\n")
                
            except KeyboardInterrupt:
                print("\n\nConversation interrupted. Generating analysis...\n")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Display analysis
        if self.get_user_messages():
            self.print_statement_analysis()
            self.print_conversation_analysis()
            
            # Save conversation
            print("\n" + "="*70)
            print("SAVING CONVERSATION")
            print("="*70)
            json_path, txt_path = self.save_conversation()
            
            if json_path and txt_path:
                print(f"\n✓ Conversation saved successfully!")
                print(f"  JSON: {json_path}")
                print(f"  TXT:  {txt_path}")
                print(f"\nStorage Info:")
                info = self.storage.get_storage_info()
                print(f"  Directory: {info['storage_directory']}")
                print(f"  Total Saved Conversations: {info['total_conversations']}")
            else:
                print("\n✗ Error saving conversation")
        else:
            print("\nNo messages to analyze.")


def main():
    """Entry point for the chatbot application."""
    chatbot = SentimentChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()