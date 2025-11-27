import socket
import asyncio
import aioice

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

        print("Testing UDP connectivity to STUN server...")
        sock.sendto(msg, stun_server)

        _, _ = sock.recvfrom(1024)
        print("STUN response received")
        return True

    except Exception as e:
        print("STUN test failed:", e)
        return False

    finally:
        if sock:
            sock.close()


async def _async_turn_test(turn_server, turn_port, username, password, turn_transport):
    """Internal async TURN test using aioice."""
    print(f"Testing TURN server: {turn_server}:{turn_port}:{turn_transport}")

    connection = aioice.Connection(
        ice_controlling=True,
        turn_server=(turn_server, turn_port),
        turn_username=username,
        turn_password=password,
        turn_transport=turn_transport,
    )
    try:
        print(f"Gathering ICE candidates via TURN {turn_server}:{turn_port} ...")
        await connection.gather_candidates()
        candidates = connection.local_candidates

        relay_candidates = [c for c in candidates if c.type == "relay"]

        if relay_candidates:
            print("TURN relay candidate acquired:", relay_candidates[0])
            return True

        print("No TURN relay candidates received — allocation failed.")
        return False

    except Exception as e:
        print(f"TURN test failed: {e}")
        return False

    finally:
        await connection.close()

def test_turn_with_auth(server, port, username, password, transport):
    """Sync wrapper for aioice TURN test."""
    return asyncio.run(_async_turn_test(server, port, username, password, transport))


def comprehensive_network_test():
    """Run comprehensive network connectivity tests."""
    results = {}

    # Test basic UDP connectivity
    results['udp_stun'] = test_udp()

    # TURN test list
    turn_servers = [
        (
            "turn.cloudflare.com",  # cleaned
            3478,
            "g001676bf83775dd93ddd77c69c80da1d13027af179ef51b4af7e3567a5028fd",
            "a77444990e4fe3b82fd0fbe795f7499409edfbc3b170cea4b28155021ab6623c",
            "udp"
        ),
    ]

    results['turn_tests'] = []

    for host, port, username, password, transport in turn_servers:
        success = test_turn_with_auth(host, port, username, password, transport)
        results['turn_tests'].append({
            "server": f"{host}:{port}",
            "success": success
        })

    return results


def test_tcp_connectivity(host, port):
    """Test TCP connectivity to a host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"TCP connection to {host}:{port} successful")
            return True

        print(f"TCP connection to {host}:{port} failed")
        return False

    except Exception as e:
        print(f"TCP test failed: {e}")
        return False


@app.entrypoint
def my_agent(payload):
    network_results = comprehensive_network_test()

    udp_ok = network_results.get('udp_stun', False)
    turn_ok = any(t['success'] for t in network_results['turn_tests'])
    tcp_ok = network_results.get('tcp_test', False)

    connectivity_status = []
    if udp_ok:
        connectivity_status.append("UDP/STUN")
    if turn_ok:
        connectivity_status.append("TURN")
    if tcp_ok:
        connectivity_status.append("TCP")

    return {
        "result": f"Hello {payload.get('name', 'World')}!",
        "network_test_results": network_results,
        "connectivity_status": ", ".join(connectivity_status) if connectivity_status else "No connectivity",
        "webrtc_feasible": udp_ok or turn_ok,
        "turn_available": turn_ok,
    }


if __name__ == "__main__":
    #
    results = comprehensive_network_test()
    print(results)
