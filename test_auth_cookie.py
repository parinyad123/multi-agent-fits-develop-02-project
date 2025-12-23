# test_auth_cookie.py

import requests

BASE_URL = "http://localhost:8003"
AUTH_BASE = f"{BASE_URL}/api/v1/auth"  # ✅ แก้ไขตรงนี้

USERNAME = "testuser"
PASSWORD = "password123"

session = requests.Session()

def test_login_with_cookie():
    """Test 1: Login with cookie (default)"""
    print("\n1️⃣ Testing Login with Cookie...")
    
    response = session.post(
        f"{AUTH_BASE}/login",  # ✅ ใช้ AUTH_BASE
        json={"username": USERNAME, "password": PASSWORD}
    )
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print("   ✅ Login successful")
        print(f"   Token: {response.json()['access_token'][:50]}...")
        
        # Check if cookie was set
        if 'access_token' in session.cookies:
            print(f"   ✅ Cookie set: {session.cookies['access_token'][:50]}...")
        else:
            print("   ❌ Cookie NOT set")
    else:
        print(f"   ❌ Login failed: {response.text}")

def test_login_without_cookie():
    """Test 2: Login without cookie"""
    print("\n2️⃣ Testing Login without Cookie (set_cookie=false)...")
    
    response = requests.post(
        f"{AUTH_BASE}/login?set_cookie=false",  # ✅ ใช้ AUTH_BASE
        json={"username": USERNAME, "password": PASSWORD}
    )
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print("   ✅ Login successful")
        
        # Check if cookie was NOT set
        if 'Set-Cookie' not in response.headers:
            print("   ✅ Cookie NOT set (as expected)")
        else:
            print("   ❌ Cookie was set (unexpected)")
    else:
        print(f"   ❌ Login failed: {response.text}")

def test_access_with_cookie():
    """Test 3: Access protected endpoint with cookie"""
    print("\n3️⃣ Testing Access with Cookie...")
    
    response = session.get(f"{AUTH_BASE}/me")  # ✅ ใช้ AUTH_BASE
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        user = response.json()
        print(f"   ✅ Access successful: {user['username']}")
    else:
        print(f"   ❌ Access denied: {response.text}")

def test_logout():
    """Test 4: Logout (delete cookie)"""
    print("\n4️⃣ Testing Logout...")
    
    response = session.post(f"{AUTH_BASE}/logout")  # ✅ ใช้ AUTH_BASE
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"   ✅ Logout successful: {response.json()['message']}")
        
        # Check if cookie was deleted
        if 'access_token' not in session.cookies:
            print("   ✅ Cookie deleted")
        else:
            print("   ⚠️ Cookie still exists (check Max-Age=0)")
    else:
        print(f"   ❌ Logout failed: {response.text}")

def test_access_after_logout():
    """Test 5: Access after logout should fail"""
    print("\n5️⃣ Testing Access after Logout...")
    
    response = session.get(f"{AUTH_BASE}/me")  # ✅ ใช้ AUTH_BASE
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 401:
        print("   ✅ Access denied (as expected)")
    else:
        print(f"   ❌ Access should be denied, got {response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("🍪 Testing Cookie-based Authentication")
    print("=" * 60)
    
    # ✅ Test URL first
    print(f"\n🔍 Testing endpoint: {AUTH_BASE}/login")
    test_response = requests.get(f"{AUTH_BASE}/login")
    print(f"   Status: {test_response.status_code}")
    
    if test_response.status_code == 404:
        print("\n❌ ERROR: Auth endpoint not found!")
        print("   Please check your API routes.")
        print("\n   Try these commands to find correct path:")
        print(f"   - curl {BASE_URL}/docs")
        print(f"   - curl {BASE_URL}/openapi.json | grep login")
        exit(1)
    
    test_login_with_cookie()
    test_login_without_cookie()
    test_access_with_cookie()
    test_logout()
    test_access_after_logout()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)