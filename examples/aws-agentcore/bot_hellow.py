import socket
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

def test_udp():
    """Test UDP connectivity using STUN server"""
    stun_server = ("stun.l.google.com", 19302)
    msg = b'\x00\x01\x00\x00' + b'\x21\x12\xa4\x42' + b'\x00' * 12

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)

        print("Trying to contact STUN server...")
        sock.sendto(msg, stun_server)

        data, addr = sock.recvfrom(1024)
        print("STUN response received:", data)
        return True

    except Exception as e:
        print("STUN test failed:", e)
        return False
    finally:
        if sock:
            sock.close()

@app.entrypoint
def my_agent(payload):
    # Run the UDP test
    udp_test_passed = test_udp()
    
    # Return the result along with the greeting
    return {
        "result": f"Hello {payload.get('name', 'World')}!",
        "udp_test_passed": udp_test_passed,
        "status": "UDP test passed" if udp_test_passed else "UDP test failed"
    }

if __name__ == "__main__":
    app.run()