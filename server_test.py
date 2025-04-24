from actions import ActionClient
import requests

# Test connection
try:
    response = requests.get("http://localhost:15702", timeout=1)
    print(f"Server response: {response.status_code}")
except Exception as e:
    print(f"Connection failed: {type(e).__name__}: {e}")

# Test client
try:
    client = ActionClient("http://localhost:15702")
    print("Client created successfully")
    print("Testing move action...")
    result = client.send_action(
        entity_id=1,
        action_type='move',
        action_data={'direction': [1,0], 'speed': 1.0}
    )
    print(f"Action result: {result}")
except Exception as e:
    print(f"Client failed: {type(e).__name__}: {e}")











"""from actions import ActionClient

client = ActionClient("http://localhost:15702")

# Test movement
print("Moving right:", client.send_action(
    entity_id=0,
    action_type='move',
    action_data={'direction': [1, 0], 'speed': 1.0}
))
"""