#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
from playwright.async_api import async_playwright

URL = "https://aidemos.meta.com/segment-anything/editor/convert-image-to-3d"

# 你的 SSH 反向隧道代理：
# Windows 127.0.0.1:7897 -> 服务器 127.0.0.1:17890
PROXY_SERVER = "http://127.0.0.1:17890"


async def main():
    out_dir = "./output_meta_web_debug"
    os.makedirs(out_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy={
                "server": PROXY_SERVER,
            },
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--use-gl=swiftshader",
                "--enable-webgl",
                "--disable-setuid-sandbox",
                "--window-size=1800,1000",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        context = await browser.new_context(
            viewport={"width": 1800, "height": 1000},
            ignore_https_errors=True,
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        page = await context.new_page()

        console_logs = []
        failed_requests = []
        page_errors = []

        page.on("console", lambda msg: console_logs.append(f"{msg.type}: {msg.text}"))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(
                f"{req.method} {req.url} :: {req.failure}"
            ),
        )

        print("[Proxy]", PROXY_SERVER)
        print("[Open]", URL)

        try:
            await page.goto(URL, wait_until="commit", timeout=60000)
        except Exception as e:
            print("[Goto error]", repr(e))

        # 注意：这里是累计等待。总等待约 10 + 30 + 60 + 120 秒。
        for wait_s in [10, 30, 60, 120]:
            await page.wait_for_timeout(wait_s * 1000)

            title = await page.title()
            html = await page.content()

            buttons = []
            links = []
            inputs = []

            try:
                buttons = await page.locator("button").all_inner_texts()
            except Exception:
                pass

            try:
                links = await page.locator("a").all_inner_texts()
            except Exception:
                pass

            try:
                inputs = await page.locator("input").evaluate_all(
                    "(els) => els.map(e => ({type:e.type, accept:e.accept, name:e.name, id:e.id}))"
                )
            except Exception:
                pass

            print("=" * 80)
            print(f"[After wait {wait_s}s]")
            print("title:", title)
            print("url:", page.url)
            print("html length:", len(html))
            print(
                "buttons:",
                [b.strip().replace("\n", " ") for b in buttons if b.strip()][:50],
            )
            print(
                "links:",
                [a.strip().replace("\n", " ") for a in links if a.strip()][:50],
            )
            print("inputs:", inputs)

            with open(
                os.path.join(out_dir, f"page_after_{wait_s}s.html"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(html)

            await page.screenshot(
                path=os.path.join(out_dir, f"screenshot_after_{wait_s}s.png"),
                full_page=True,
            )

        with open(os.path.join(out_dir, "console_logs.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(console_logs))

        with open(os.path.join(out_dir, "page_errors.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(page_errors))

        with open(os.path.join(out_dir, "failed_requests.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(failed_requests))

        print("\n[Saved]", os.path.abspath(out_dir))
        print("console logs:", len(console_logs))
        print("page errors:", len(page_errors))
        print("failed requests:", len(failed_requests))

        await context.close()
        await browser.close()


asyncio.run(main())