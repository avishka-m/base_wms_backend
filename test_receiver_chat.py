import asyncio
import traceback
from app.models.chatbot.chat_models import ChatMessage as ChatMessageRequest
from app.services.chatbot.auth_service import get_allowed_chatbot_roles

async def test_receiver_chat():
    print('🧪 Testing ReceivingClerk chat access...')
    
    # Simulate ReceivingClerk user
    mock_receiver_user = {
        "username": "receiver",
        "role": "ReceivingClerk",
        "email": "receiver@warehouse.com"
    }
    
    print(f"👤 User: {mock_receiver_user['username']} ({mock_receiver_user['role']})")
    
    # Test role validation
    allowed_roles = get_allowed_chatbot_roles(mock_receiver_user["role"])
    print(f"✅ Allowed chatbot roles: {allowed_roles}")
    
    # Test that clerk role is allowed
    if "clerk" in allowed_roles:
        print("✅ ReceivingClerk can access 'clerk' chatbot")
        
        # Test the chat endpoint logic
        from app.api.enhanced_chatbot_routes import agent_service
        
        try:
            # Test message processing with clerk role
            result = await agent_service.process_message(
                message="Hello, I need help with receiving items",
                role="clerk",
                context={"user_role": "ReceivingClerk"}
            )
            print(f"✅ Chat processing successful: {result.get('response', 'No response')[:100]}...")
            return True
            
        except Exception as e:
            print(f"❌ Chat processing failed: {e}")
            traceback.print_exc()
            return False
    else:
        print("❌ ReceivingClerk cannot access 'clerk' chatbot")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_receiver_chat())
    if success:
        print("\n🎉 ReceivingClerk chat access test PASSED!")
    else:
        print("\n💥 ReceivingClerk chat access test FAILED!") 