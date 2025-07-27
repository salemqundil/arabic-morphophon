#!/usr/bin/env python3
"""
🧪 Test the Particles Engine API Endpoint
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json

import_data requests

def test_particles_api():
    """Test the particles analysis API endpoint"""
    
    # API endpoint
    url = "http://localhost:5001/api/analyze-particles"
    
    # Test cases
    test_texts = [
        "هل كتب الطالب الواجب؟",
        "إن الله غفور رحيم",
        "يا أحمد، هذا كتابك",
        "الذي يدرس ينجح",
        "لا تنس أن تحضر الكتاب"
    ]
    
    print("🧪 Testing Particles Engine API")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        
        try:
            # Send POST request
            response = requests.post(url, 
                                   json={'text': text},
                                   headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {data['status']}")
                print(f"   📊 Analysis:")
                print(f"      Total Words: {data['analysis']['total_words']}")
                print(f"      Particles Found: {data['analysis']['particles_found']}")
                print(f"      Particle %: {data['analysis']['particle_percentage']}%")
                
                if data['analysis']['particles_by_category']:
                    print(f"   🏷️ Categories:")
                    for category, particles in data['analysis']['particles_by_category'].items():
                        print(f"      {category}: {particles}")
                
                print(f"   ⏱️ Response Time: {data['meta']['response_time']}s")
                
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection Error: Make sure the server is running on localhost:5001")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_particles_api()
