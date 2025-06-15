#!/usr/bin/env python3
"""
Demo script for AI Chatbot Web GUI
Shows how to interact with the chatbot programmatically
"""

import requests
import json
import time
import threading
from datetime import datetime

class ChatbotDemo:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.conversation_history = []
    
    def send_message(self, message):
        """Send a message to the chatbot and return the response"""
        try:
            payload = {
                "message": message,
                "history": self.conversation_history
            }
            
            print(f"👤 User: {message}")
            print("🤖 AI: ", end="", flush=True)
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    ai_response = data.get('response', '')
                    print(ai_response)
                    
                    # Add to conversation history
                    self.conversation_history.extend([
                        {
                            "type": "user",
                            "content": message,
                            "timestamp": datetime.now().isoformat()
                        },
                        {
                            "type": "bot",
                            "content": ai_response,
                            "timestamp": data.get('timestamp', datetime.now().isoformat())
                        }
                    ])
                    
                    return ai_response
                else:
                    error_msg = data.get('error', 'Unknown error')
                    print(f"❌ Error: {error_msg}")
                    return None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
            return None
    
    def check_health(self):
        """Check if the chatbot service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                return None
        except requests.exceptions.RequestException:
            return None
    
    def interactive_demo(self):
        """Run an interactive demo session"""
        print("🤖 AI Chatbot Interactive Demo")
        print("=" * 40)
        print("Type 'quit' to exit, 'clear' to clear history")
        print()
        
        # Check health
        health = self.check_health()
        if health:
            status = "🟢 Online" if health.get('ollama_connected') else "🟡 Limited"
            print(f"Status: {status}")
            print(f"Model: {health.get('model', 'Unknown')}")
        else:
            print("Status: 🔴 Offline")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared")
                    continue
                elif not user_input:
                    continue
                
                self.send_message(user_input)
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\n👋 Demo ended. Goodbye!")
    
    def automated_demo(self):
        """Run an automated demo with predefined messages"""
        print("🤖 AI Chatbot Automated Demo")
        print("=" * 40)
        print()
        
        # Check health first
        health = self.check_health()
        if not health:
            print("❌ Chatbot service is not available")
            return
        
        demo_messages = [
            "Hello! How are you today?",
            "What can you help me with?",
            "Can you explain what artificial intelligence is?",
            "What's the weather like? (This should show how you handle questions you can't answer)",
            "Thank you for the demonstration!"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"📝 Demo Message {i}/{len(demo_messages)}")
            print("-" * 30)
            
            response = self.send_message(message)
            
            if response is None:
                print("❌ Demo failed - chatbot not responding")
                break
            
            print()
            
            # Wait between messages
            if i < len(demo_messages):
                print("⏳ Waiting 3 seconds...")
                time.sleep(3)
                print()
        
        print("✅ Automated demo completed!")
    
    def stress_test(self, num_messages=10):
        """Send multiple messages to test performance"""
        print(f"🔥 Stress Test - Sending {num_messages} messages")
        print("=" * 40)
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        for i in range(num_messages):
            message = f"Test message #{i+1}: Can you respond to this?"
            print(f"📤 Sending message {i+1}/{num_messages}...")
            
            response = self.send_message(message)
            
            if response:
                successful += 1
                print(f"✅ Response received ({len(response)} characters)")
            else:
                failed += 1
                print("❌ No response")
            
            print()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("📊 Stress Test Results:")
        print(f"   Total messages: {num_messages}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Average time per message: {duration/num_messages:.2f} seconds")
    
    def show_conversation_history(self):
        """Display the current conversation history"""
        print("📜 Conversation History")
        print("=" * 40)
        
        if not self.conversation_history:
            print("No messages in history")
            return
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = "👤 User" if msg['type'] == 'user' else "🤖 AI"
            timestamp = msg.get('timestamp', 'Unknown time')
            content = msg.get('content', '')
            
            print(f"{i}. {role} ({timestamp}):")
            print(f"   {content}")
            print()

def main():
    """Main demo function"""
    import sys
    
    demo = ChatbotDemo()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'interactive':
            demo.interactive_demo()
        elif command == 'auto':
            demo.automated_demo()
        elif command == 'stress':
            num_messages = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            demo.stress_test(num_messages)
        elif command == 'health':
            health = demo.check_health()
            if health:
                print("🟢 Chatbot service is healthy")
                print(json.dumps(health, indent=2))
            else:
                print("🔴 Chatbot service is not available")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: interactive, auto, stress, health")
    else:
        print("🤖 AI Chatbot Demo")
        print("=" * 20)
        print()
        print("Available demo modes:")
        print("  python demo.py interactive  - Interactive chat session")
        print("  python demo.py auto         - Automated demo messages")
        print("  python demo.py stress [n]   - Stress test with n messages")
        print("  python demo.py health       - Check service health")
        print()
        print("Choose a demo mode:")
        print("1. Interactive Demo")
        print("2. Automated Demo")
        print("3. Stress Test")
        print("4. Health Check")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                demo.interactive_demo()
            elif choice == '2':
                demo.automated_demo()
            elif choice == '3':
                num = input("Number of messages (default 10): ").strip()
                num_messages = int(num) if num.isdigit() else 10
                demo.stress_test(num_messages)
            elif choice == '4':
                health = demo.check_health()
                if health:
                    print("🟢 Chatbot service is healthy")
                    print(json.dumps(health, indent=2))
                else:
                    print("🔴 Chatbot service is not available")
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Demo cancelled")
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    main()