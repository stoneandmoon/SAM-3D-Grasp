#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
from playwright.async_api import async_playwright

URL = "https://aidemos.meta.com/segment-anything/editor/convert-image-to-3d"
PROXY_SERVER = "http://127.0.0.1:17890"


async def main():
    out_dir = "./output_meta_web_debug_click"
    os.makedirs(out_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy={"server": PROXY_SERVER},
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
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        page = await context.new_page()

        print("[Open]", URL)
        await page.goto(URL, wait_until="commit", timeout=60000)
        await page.wait_for_timeout(20000)

        print("[Before]")
        print("url:", page.url)
        print("title:", await page.title())

        await page.screenshot(
            path=os.path.join(out_dir, "01_before_click.png"),
            full_page=True,
        )

        links = await page.locator("a").evaluate_all(
            """els => els.map((e, i) => ({
                i,
                text: e.innerText,
                href: e.href,
                aria: e.getAttribute('aria-label')
            }))"""
        )

        print("\n[Links]")
        for item in links:
            text = (item.get("text") or "").strip().replace("\n", " ")
            href = item.get("href")
            aria = item.get("aria")
            if text or aria:
                print(f"[{item['i']}] text={text} | aria={aria} | href={href}")

        # 优先点击包含 Create 3D scenes SAM 3D 的链接
        clicked = False
        candidates = [
            "Create 3D scenes SAM 3D",
            "Create 3D scenes",
            "SAM 3D",
        ]

        for text in candidates:
            try:
                print(f"\n[Try click] {text}")
                await page.get_by_text(text, exact=False).first.click(timeout=8000)
                clicked = True
                print("[Clicked]", text)
                break
            except Exception as e:
                print("[Click failed]", text, repr(e))

        if not clicked:
            print("[WARN] 没有点击成功，尝试用 JS 点击包含 Create 3D scenes 的 a 标签")
            ok = await page.evaluate(
                """() => {
                    const links = Array.from(document.querySelectorAll('a'));
                    const a = links.find(x => (x.innerText || '').includes('Create 3D scenes'));
                    if (a) {
                        a.click();
                        return true;
                    }
                    return false;
                }"""
            )
            print("[JS click result]", ok)

        await page.wait_for_timeout(25000)

        print("\n[After]")
        print("url:", page.url)
        print("title:", await page.title())

        await page.screenshot(
            path=os.path.join(out_dir, "02_after_click.png"),
            full_page=True,
        )

        buttons = []
        inputs = []
        links2 = []

        try:
            buttons = await page.locator("button").all_inner_texts()
        except Exception:
            pass

        try:
            inputs = await page.locator("input").evaluate_all(
                "(els) => els.map(e => ({type:e.type, accept:e.accept, name:e.name, id:e.id}))"
            )
        except Exception:
            pass

        try:
            links2 = await page.locator("a").all_inner_texts()
        except Exception:
            pass

        print("\n[After DOM]")
        print("buttons:", [b.strip().replace("\n", " ") for b in buttons if b.strip()][:80])
        print("inputs:", inputs)
        print("links:", [a.strip().replace("\n", " ") for a in links2 if a.strip()][:80])

        html = await page.content()
        with open(os.path.join(out_dir, "02_after_click.html"), "w", encoding="utf-8") as f:
            f.write(html)

        await context.close()
        await browser.close()


asyncio.run(main())
