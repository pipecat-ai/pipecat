import http.server
import ssl
import argparse


def run_https_server(port, certfile, keyfile):
    handler = http.server.SimpleHTTPRequestHandler

    https_server = http.server.HTTPServer(("0.0.0.0", port), handler)

    https_server.socket = ssl.wrap_socket(
        https_server.socket, certfile=certfile, keyfile=keyfile, server_side=True
    )

    print(f"Server running on https://0.0.0.0:{port}")
    https_server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an HTTPS server")
    parser.add_argument("--port", type=int, default=443, help="Port to run the server on")
    parser.add_argument(
        "--cert",
        type=str,
        default="/your/path/to/sslcert.pem",
        help="Path to the SSL certificate file",
    )
    parser.add_argument(
        "--key", type=str, default="//your/path/to/sslkey.pem", help="Path to the SSL key file"
    )

    args = parser.parse_args()

    run_https_server(args.port, args.cert, args.key)
