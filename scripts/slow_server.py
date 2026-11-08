import asyncio
from pathlib import Path


async def handle_client(reader, writer):
    try:

        request = b""
        while b"\r\n\r\n" not in request:
            chunk = await reader.read(1024)
            if not chunk:
                break
            request += chunk

        if not request:
            return

        line = request.decode('utf-8', errors='ignore').split('\n')[0]
        parts = line.split(' ')
        path = parts[1] if len(parts) > 1 else '/'
        if path == '/':
            path = '/page0.html'

        await asyncio.sleep(0.1)

        file_path = Path(__file__).resolve().parent.parent / "test_site" / path.lstrip('/')
        if file_path.exists() and file_path.is_file():
            content = file_path.read_bytes()
            response = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: " + str(
                len(content)).encode() + b"\r\nConnection: close\r\n\r\n" + content
        else:
            response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

        writer.write(response)
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8001)
    print("Async slow server on http://127.0.0.1:8001 (0.1s delay)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())