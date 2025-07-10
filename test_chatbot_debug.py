import sys
import asyncio
import traceback

async def test_services():
    print('Testing backend services...')
    
    try:
        from app.api.enhanced_chatbot_routes import agent_service, conversation_service
        print('✅ Services imported successfully')
        
        # Test agent service
        print('\n🤖 Testing agent service...')
        available_roles = agent_service.get_available_roles()
        print(f'Available roles: {available_roles}')
        
        # Test message processing
        print('Testing message processing...')
        result = await agent_service.process_message(
            message='Hello, test message',
            role='clerk',
            context={}
        )
        print(f'✅ Agent processing successful: {result.get("response", "No response")[:100]}...')
        
    except Exception as e:
        print(f'❌ Agent service error: {e}')
        traceback.print_exc()
        return False
    
    try:
        print('\n💬 Testing conversation service...')
        # Test conversation creation
        conv_result = await conversation_service.create_conversation(
            user_id='test_user',
            title='Test conversation',
            agent_role='clerk',
            available_roles=['clerk']
        )
        print(f'✅ Conversation creation successful: {conv_result}')
        
    except Exception as e:
        print(f'❌ Conversation service error: {e}')
        traceback.print_exc()
        return False
    
    print('\n✅ All services working correctly!')
    return True

if __name__ == "__main__":
    success = asyncio.run(test_services())
    if not success:
        sys.exit(1) 