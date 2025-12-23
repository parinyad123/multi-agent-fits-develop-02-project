# test_sse_ownership.py

import requests

BASE_URL = "http://localhost:8003"

# User A
session_a = requests.Session()
session_a.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={"username": "testuser", "password": "password123"}
)

# User B
session_b = requests.Session()
session_b.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={"username": "user_b", "password": "password123"}
)

# User A submit analysis
response = session_a.post(
    f"{BASE_URL}/api/v2/analyze",
    json={"query": "test"}
)
task_id = response.json()["task_id"]

print(f"Task created by User A: {task_id}")

# ✅ Test 1: User A access their own workflow
print("\n1️⃣ User A accessing their own workflow:")
response = session_a.get(f"{BASE_URL}/api/v2/analyze/{task_id}/stream", stream=True)
print(f"   Status: {response.status_code}")
print(f"   Expected: 200 ✅")

# ❌ Test 2: User B try to access User A's workflow
print("\n2️⃣ User B trying to access User A's workflow:")
response = session_b.get(f"{BASE_URL}/api/v2/analyze/{task_id}/stream", stream=True)
print(f"   Status: {response.status_code}")
print(f"   Expected: 403 ✅")

if response.status_code == 403:
    print(f"   Message: {response.json()['detail']}")