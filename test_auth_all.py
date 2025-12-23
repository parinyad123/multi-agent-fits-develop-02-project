# test_auth_all.py

import requests

BASE_URL = "http://localhost:8003"
VALID_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkZDdhNjNiNi03MGZmLTQ0Y2EtYThjYi0zNTM1ZjE3NGIyODEiLCJleHAiOjE3NjY0NDYzNjYsImlhdCI6MTc2NjQxNzU2Nn0.alVUzpCJxnKOlAz0MLzNU0qerewD-8N4xwl3W1Tr5VM"

def test_bearer_token():
    """Test 1: Bearer Token"""
    print("\n1️⃣ Testing Bearer Token...")
    response = requests.get(
        f"{BASE_URL}/api/v2/analyze/some-task-id/status",
        headers={"Authorization": f"Bearer {VALID_TOKEN}"}
    )
    
    if response.status_code == 404:
        print("   ✅ PASS: Bearer token authentication works")
    else:
        print(f"   ❌ FAIL: Expected 404, got {response.status_code}")
        print(f"   Response: {response.json()}")

def test_cookie():
    """Test 2: Cookie"""
    print("\n2️⃣ Testing Cookie...")
    response = requests.get(
        f"{BASE_URL}/api/v2/analyze/some-task-id/status",
        cookies={"access_token": VALID_TOKEN}
    )
    
    if response.status_code == 404:
        print("   ✅ PASS: Cookie authentication works")
    else:
        print(f"   ❌ FAIL: Expected 404, got {response.status_code}")
        print(f"   Response: {response.json()}")

def test_priority():
    """Test 3: Priority (Bearer > Cookie)"""
    print("\n3️⃣ Testing Priority (Bearer > Cookie)...")
    response = requests.get(
        f"{BASE_URL}/api/v2/analyze/some-task-id/status",
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        cookies={"access_token": "invalid_token"}
    )
    
    if response.status_code == 404:
        print("   ✅ PASS: Bearer token has priority")
    else:
        print(f"   ❌ FAIL: Expected 404, got {response.status_code}")

def test_no_auth():
    """Test 4: No Authentication"""
    print("\n4️⃣ Testing No Authentication...")
    response = requests.get(
        f"{BASE_URL}/api/v2/analyze/some-task-id/status"
    )
    
    if response.status_code == 401:
        print("   ✅ PASS: No auth returns 401")
    else:
        print(f"   ❌ FAIL: Expected 401, got {response.status_code}")

def test_invalid_token():
    """Test 5: Invalid Token"""
    print("\n5️⃣ Testing Invalid Token...")
    response = requests.get(
        f"{BASE_URL}/api/v2/analyze/some-task-id/status",
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    if response.status_code == 401:
        print("   ✅ PASS: Invalid token returns 401")
    else:
        print(f"   ❌ FAIL: Expected 401, got {response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔐 Testing Flexible Authentication System")
    print("=" * 60)
    
    test_bearer_token()
    test_cookie()
    test_priority()
    test_no_auth()
    test_invalid_token()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)