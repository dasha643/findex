import os

os.makedirs("test_site", exist_ok=True)
NUM_PAGES = 200

for i in range(NUM_PAGES):
    links = "".join([f'<a href="page{(i + j) % NUM_PAGES}.html">Link {j}</a><br>' for j in range(1, 21)])
    with open(f"test_site/page{i}.html", "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head><title>Page {i}</title></head>
        <body>
            <h1>Hello from page {i}</h1>
            {links}
        </body>
        </html>
        """)
print(f"Generated {NUM_PAGES} pages with 20 links each in test_site/")