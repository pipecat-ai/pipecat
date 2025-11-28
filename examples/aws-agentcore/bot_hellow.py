import asyncio
import os
import socket

import aioice
from bedrock_agentcore import BedrockAgentCoreApp
from dotenv import load_dotenv

load_dotenv(override=True)

app = BedrockAgentCoreApp()


def test_udp():
    """Test UDP connectivity using STUN server"""
    stun_server = ("stun.l.google.com", 19302)
    msg = b"\x00\x01\x00\x00" + b"\x21\x12\xa4\x42" + b"\x00" * 12

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


async def _async_turn_test(turn_server, turn_port, username, password, turn_transport, turn_ssl):
    """Internal async TURN test using aioice."""
    print(f"Testing TURN server: {turn_server}:{turn_port}:{turn_transport}")

    connection = aioice.Connection(
        ice_controlling=True,
        turn_server=(turn_server, turn_port),
        turn_username=username,
        turn_password=password,
        turn_ssl=turn_ssl,
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


def test_turn_with_auth(server, port, username, password, transport, turn_ssl=False):
    """Sync wrapper for aioice TURN test."""
    return asyncio.run(_async_turn_test(server, port, username, password, transport, turn_ssl))


def comprehensive_network_test():
    """Run comprehensive network connectivity tests."""
    results = {}

    # Test basic UDP connectivity
    results["udp_stun"] = test_udp()

    turn_username = os.getenv("TURN_USERNAME")
    turn_credential = os.getenv("TURN_CREDENTIAL")

    # TURN test list
    turn_servers = [
        (
            "turn.cloudflare.com",  # cleaned
            3478,
            turn_username,
            turn_credential,
            "udp",
            False,
        ),
        (
            "turn.cloudflare.com",  # cleaned
            5349,
            turn_username,
            turn_credential,
            "tcp",
            True,
        ),
        (
            "turn.cloudflare.com",  # cleaned
            443,
            turn_username,
            turn_credential,
            "tcp",
            True,
        ),
        (
            "turn.cloudflare.com",  # cleaned
            80,
            turn_username,
            turn_credential,
            "tcp",
            False,
        ),
        (
            "turn.cloudflare.com",  # cleaned
            3478,
            turn_username,
            turn_credential,
            "tcp",
            False,
        ),
    ]

    results["turn_tests"] = []

    for host, port, username, password, transport, tls in turn_servers:
        success = test_turn_with_auth(host, port, username, password, transport, tls)
        results["turn_tests"].append({"server": f"{host}:{port}", "success": success})

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

    udp_ok = network_results.get("udp_stun", False)
    turn_ok = any(t["success"] for t in network_results["turn_tests"])
    tcp_ok = network_results.get("tcp_test", False)

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
        "connectivity_status": ", ".join(connectivity_status)
        if connectivity_status
        else "No connectivity",
        "webrtc_feasible": udp_ok or turn_ok,
        "turn_available": turn_ok,
    }


if __name__ == "__main__":
    if os.getenv("PIPECAT_LOCAL_DEV") == "1":
        # Running locally
        results = comprehensive_network_test()
        print(results)
    else:
        # Running on AgentCore Runtime
        app.run()
