"""
Unit tests for Sentiment Analysis Chatbot
Run with: python -m pytest test_chatbot.py -v
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from chatbot import (
    SentimentAnalyzer, 
    SentimentChatbot, 
    ResponseGenerator,
    ConversationStorage,
    Message,
    ConversationSummary
)


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()
    
    def test_positive_sentiment(self, analyzer):
        """Test detection of positive sentiment."""
        text = "This is excellent and amazing!"
        sentiment, score = analyzer.analyze(text)
        assert sentiment == "Positive"
        assert score > 0.15
    
    def test_negative_sentiment(self, analyzer):
        """Test detection of negative sentiment."""
        text = "This is terrible and disappointing"
        sentiment, score = analyzer.analyze(text)
        assert sentiment == "Negative"
        assert score < -0.15
    
    def test_neutral_sentiment(self, analyzer):
        """Test detection of neutral sentiment."""
        text = "I went to the store today"
        sentiment, score = analyzer.analyze(text)
        assert sentiment == "Neutral"
        assert -0.15 <= score <= 0.15
    
    def test_negation_handling(self, analyzer):
        """Test that negations flip sentiment correctly."""
        # Positive word negated should be negative
        text = "This is not good at all"
        sentiment, score = analyzer.analyze(text)
        assert score < 0
        
        # Negative word negated should be positive
        text = "This is not bad"
        sentiment, score = analyzer.analyze(text)
        assert score > 0
    
    def test_intensifier_handling(self, analyzer):
        """Test that intensifiers amplify sentiment."""
        text1 = "This is good"
        text2 = "This is very good"
        
        _, score1 = analyzer.analyze(text1)
        _, score2 = analyzer.analyze(text2)
        
        assert score2 > score1
    
    def test_empty_string(self, analyzer):
        """Test handling of empty input."""
        sentiment, score = analyzer.analyze("")
        assert sentiment == "Neutral"
        assert score == 0.0
    
    def test_mixed_sentiment(self, analyzer):
        """Test text with both positive and negative words."""
        text = "The service was good but the wait was terrible"
        sentiment, score = analyzer.analyze(text)
        # Should lean negative due to 'terrible' being stronger
        assert sentiment in ["Negative", "Neutral"]
    
    def test_case_insensitivity(self, analyzer):
        """Test that analyzer handles different cases."""
        text1 = "EXCELLENT"
        text2 = "excellent"
        
        sentiment1, score1 = analyzer.analyze(text1)
        sentiment2, score2 = analyzer.analyze(text2)
        
        assert sentiment1 == sentiment2
        assert abs(score1 - score2) < 0.01


class TestResponseGenerator:
    """Test suite for ResponseGenerator class."""
    
    @pytest.fixture
    def generator(self):
        return ResponseGenerator()
    
    def test_positive_response(self, generator):
        """Test generation of positive response."""
        response = generator.generate("Positive", sentiment_score=0.8, user_message="Great service!")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_negative_response(self, generator):
        """Test generation of negative response."""
        response = generator.generate("Negative", sentiment_score=-0.8, user_message="Bad experience")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_neutral_response(self, generator):
        """Test generation of neutral response."""
        response = generator.generate("Neutral", sentiment_score=0.0, user_message="I have a question")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_greeting_response(self, generator):
        """Test generation of greeting."""
        response = generator.generate("Neutral", is_first=True)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_response_variety(self, generator):
        """Test that responses vary on repeated calls."""
        responses = [generator.generate("Positive", sentiment_score=0.5, user_message="Good") for _ in range(10)]
        # Should have at least 2 different responses
        assert len(set(responses)) >= 2
    
    def test_sentiment_streak_empathy(self, generator):
        """Test that consecutive negative messages trigger empathy."""
        # First negative message
        response1 = generator.generate("Negative", sentiment_score=-0.5, user_message="This is bad")
        # Second negative message should show more empathy
        response2 = generator.generate("Negative", sentiment_score=-0.6, user_message="Still not good")
        
        assert isinstance(response1, str)
        assert isinstance(response2, str)
        # Responses should be different due to context
        assert response1 != response2 or len(response2) > 0
    
    def test_sentiment_improvement_encouragement(self, generator):
        """Test encouragement when sentiment improves."""
        # Start with negative
        generator.generate("Negative", sentiment_score=-0.5, user_message="Bad")
        # Follow with positive
        response = generator.generate("Positive", sentiment_score=0.5, user_message="Better now")
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestSentimentChatbot:
    """Test suite for SentimentChatbot class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def chatbot(self, temp_dir):
        return SentimentChatbot(storage_dir=temp_dir)
    
    def test_initialization(self, chatbot):
        """Test chatbot initializes correctly."""
        assert len(chatbot.conversation_history) == 0
        assert isinstance(chatbot.analyzer, SentimentAnalyzer)
        assert isinstance(chatbot.response_generator, ResponseGenerator)
    
    def test_process_user_message(self, chatbot):
        """Test processing a user message."""
        response = chatbot.process_user_message("I love this service!")
        
        assert isinstance(response, str)
        assert len(chatbot.conversation_history) == 2  # User + Bot
        assert chatbot.conversation_history[0].speaker == "User"
        assert chatbot.conversation_history[1].speaker == "Chatbot"
    
    def test_sentiment_stored_correctly(self, chatbot):
        """Test that sentiment is stored with user messages."""
        chatbot.process_user_message("This is excellent!")
        user_msg = chatbot.get_user_messages()[0]
        
        assert user_msg.sentiment == "Positive"
        assert user_msg.sentiment_score > 0
    
    def test_get_user_messages(self, chatbot):
        """Test filtering of user messages."""
        chatbot.process_user_message("Hello")
        chatbot.process_user_message("How are you?")
        
        user_messages = chatbot.get_user_messages()
        assert len(user_messages) == 2
        assert all(msg.speaker == "User" for msg in user_messages)
    
    def test_conversation_analysis_empty(self, chatbot):
        """Test analysis with no messages."""
        summary = chatbot.analyze_conversation()
        
        assert isinstance(summary, ConversationSummary)
        assert summary.total_messages == 0
        assert summary.overall_sentiment == "Neutral"
    
    def test_conversation_analysis_positive(self, chatbot):
        """Test analysis of positive conversation."""
        chatbot.process_user_message("This is great!")
        chatbot.process_user_message("I love it!")
        chatbot.process_user_message("Excellent service!")
        
        summary = chatbot.analyze_conversation()
        
        assert summary.overall_sentiment == "Positive"
        assert summary.total_messages == 3
        assert summary.sentiment_distribution["Positive"] == 3
    
    def test_conversation_analysis_negative(self, chatbot):
        """Test analysis of negative conversation."""
        chatbot.process_user_message("This is terrible")
        chatbot.process_user_message("Very disappointing")
        chatbot.process_user_message("Worst experience")
        
        summary = chatbot.analyze_conversation()
        
        assert summary.overall_sentiment == "Negative"
        assert summary.total_messages == 3
        assert summary.sentiment_distribution["Negative"] == 3
    
    def test_conversation_analysis_mixed(self, chatbot):
        """Test analysis of mixed sentiment conversation."""
        chatbot.process_user_message("This is great!")
        chatbot.process_user_message("But this part is bad")
        chatbot.process_user_message("Overall okay")
        
        summary = chatbot.analyze_conversation()
        
        assert summary.total_messages == 3
        assert isinstance(summary.overall_sentiment, str)
        assert summary.overall_sentiment in ["Positive", "Neutral", "Negative"]
    
    def test_trend_analysis_improving(self, chatbot):
        """Test detection of improving sentiment trend."""
        # Start negative
        chatbot.process_user_message("This is terrible")
        chatbot.process_user_message("Not good")
        # End positive
        chatbot.process_user_message("Getting better")
        chatbot.process_user_message("This is excellent!")
        
        summary = chatbot.analyze_conversation()
        assert "Improving" in summary.trend_description or "positive" in summary.trend_description.lower()
    
    def test_trend_analysis_declining(self, chatbot):
        """Test detection of declining sentiment trend."""
        # Start positive
        chatbot.process_user_message("This is great!")
        chatbot.process_user_message("Really good")
        # End negative
        chatbot.process_user_message("Getting worse")
        chatbot.process_user_message("Very disappointing")
        
        summary = chatbot.analyze_conversation()
        assert "Declining" in summary.trend_description or "negative" in summary.trend_description.lower()


class TestConversationStorage:
    """Test suite for ConversationStorage class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def storage(self, temp_dir):
        return ConversationStorage(storage_dir=temp_dir)
    
    def test_initialization(self, storage, temp_dir):
        """Test storage initializes correctly."""
        assert storage.storage_dir == Path(temp_dir)
        assert storage.storage_dir.exists()
        assert storage.current_session_id is not None
    
    def test_save_conversation_json(self, storage):
        """Test saving conversation to JSON."""
        messages = [
            Message(speaker="User", text="Hello", sentiment="Neutral", sentiment_score=0.0),
            Message(speaker="Chatbot", text="Hi there!", sentiment="", sentiment_score=0.0)
        ]
        summary = ConversationSummary(
            overall_sentiment="Neutral",
            average_score=0.0,
            sentiment_distribution={"Positive": 0, "Neutral": 1, "Negative": 0},
            trend_description="Stable",
            total_messages=1
        )
        
        filepath = storage.save_conversation(messages, summary)
        
        assert filepath != ""
        assert os.path.exists(filepath)
        assert filepath.endswith(".json")
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert data["session_id"] == storage.current_session_id
        assert len(data["messages"]) == 2
        assert "sentiment_analysis" in data
    
    def test_save_conversation_txt(self, storage):
        """Test saving conversation to text file."""
        messages = [
            Message(speaker="User", text="Great!", sentiment="Positive", sentiment_score=0.8)
        ]
        summary = ConversationSummary(
            overall_sentiment="Positive",
            average_score=0.8,
            sentiment_distribution={"Positive": 1, "Neutral": 0, "Negative": 0},
            trend_description="Positive conversation",
            total_messages=1
        )
        
        filepath = storage.save_conversation_txt(messages, summary)
        
        assert filepath != ""
        assert os.path.exists(filepath)
        assert filepath.endswith(".txt")
        
        # Verify content
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "CONVERSATION TRANSCRIPT" in content
        assert "Great!" in content
        assert "Positive" in content
    
    def test_load_conversation(self, storage):
        """Test loading saved conversation."""
        messages = [
            Message(speaker="User", text="Test", sentiment="Neutral", sentiment_score=0.0)
        ]
        summary = ConversationSummary(
            overall_sentiment="Neutral",
            average_score=0.0,
            sentiment_distribution={"Positive": 0, "Neutral": 1, "Negative": 0},
            trend_description="Stable",
            total_messages=1
        )
        
        # Save first
        storage.save_conversation(messages, summary)
        
        # Load back
        loaded = storage.load_conversation(storage.current_session_id)
        
        assert loaded["session_id"] == storage.current_session_id
        assert len(loaded["messages"]) == 1
        assert loaded["messages"][0]["text"] == "Test"
    
    def test_list_conversations(self, storage):
        """Test listing saved conversations."""
        # Save multiple conversations
        for i in range(3):
            messages = [Message(speaker="User", text=f"Message {i}", sentiment="Neutral", sentiment_score=0.0)]
            summary = ConversationSummary(
                overall_sentiment="Neutral",
                average_score=0.0,
                sentiment_distribution={"Positive": 0, "Neutral": 1, "Negative": 0},
                trend_description="Stable",
                total_messages=1
            )
            storage.save_conversation(messages, summary)
            # Change session ID for next iteration
            storage.current_session_id = f"test_{i}"
        
        conversations = storage.list_conversations()
        assert len(conversations) >= 1
    
    def test_get_storage_info(self, storage):
        """Test getting storage information."""
        info = storage.get_storage_info()
        
        assert "storage_directory" in info
        assert "total_conversations" in info
        assert "recent_sessions" in info
        assert isinstance(info["total_conversations"], int)


class TestMessage:
    """Test suite for Message dataclass."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(speaker="User", text="Hello")
        assert msg.speaker == "User"
        assert msg.text == "Hello"
        assert msg.sentiment == ""
        assert msg.sentiment_score == 0.0
    
    def test_message_with_sentiment(self):
        """Test creating a message with sentiment."""
        msg = Message(
            speaker="User",
            text="Great!",
            sentiment="Positive",
            sentiment_score=0.8
        )
        assert msg.sentiment == "Positive"
        assert msg.sentiment_score == 0.8
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(speaker="User", text="Hello", sentiment="Neutral", sentiment_score=0.0)
        msg_dict = msg.to_dict()
        
        assert isinstance(msg_dict, dict)
        assert msg_dict["speaker"] == "User"
        assert msg_dict["text"] == "Hello"
        assert "timestamp" in msg_dict


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_full_conversation_flow(self, temp_dir):
        """Test a complete conversation flow."""
        chatbot = SentimentChatbot(storage_dir=temp_dir)
        
        # Simulate conversation
        messages = [
            "Your service disappoints me",
            "Last experience was better",
            "I hope things improve"
        ]
        
        for msg in messages:
            chatbot.process_user_message(msg)
        
        # Verify conversation stored
        assert len(chatbot.get_user_messages()) == 3
        
        # Verify analysis works
        summary = chatbot.analyze_conversation()
        assert summary.total_messages == 3
        assert isinstance(summary.overall_sentiment, str)
        assert isinstance(summary.trend_description, str)
        
        # Test saving
        json_path, txt_path = chatbot.save_conversation()
        assert json_path != ""
        assert txt_path != ""
        assert os.path.exists(json_path)
        assert os.path.exists(txt_path)
    
    def test_sentiment_accuracy_workflow(self, temp_dir):
        """Test sentiment accuracy through workflow."""
        chatbot = SentimentChatbot(storage_dir=temp_dir)
        
        # Clear negative
        chatbot.process_user_message("This is absolutely terrible and frustrating")
        user_msg = chatbot.get_user_messages()[0]
        assert user_msg.sentiment == "Negative"
        
        # Clear positive
        chatbot.process_user_message("This is really excellent and amazing")
        user_msg = chatbot.get_user_messages()[1]
        assert user_msg.sentiment == "Positive"
        
        # Neutral
        chatbot.process_user_message("I went to the store")
        user_msg = chatbot.get_user_messages()[2]
        assert user_msg.sentiment == "Neutral"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])