#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import mimetypes
import asyncio
from playwright.async_api import async_playwright

URL = "https://aidemos.meta.com/segment-anything/editor/convert-image-to-3d"
PROXY_SERVER = "http://127.0.0.1:17890"

RGB = "./data/test/000002/rgb/000000.png"


async def main():
    out_dir = "./output_meta_web_debug_drag"
    os.makedirs(out_dir, exist_ok=True)

    abs_rgb = os.path.abspath(RGB)
    mime = mimetypes.guess_type(abs_rgb)[0] or "image/png"

    with open(abs_rgb, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

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

        await page.screenshot(path=os.path.join(out_dir, "01_before_drop.png"), full_page=True)

        print("[Drop] dispatch file drop to document/body/largest elements")

        result = await page.evaluate(
            """
            async ({b64, filename, mime}) => {
                function b64ToUint8Array(b64) {
                    const binary = atob(b64);
                    const len = binary.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
                    return bytes;
                }

                const bytes = b64ToUint8Array(b64);
                const file = new File([bytes], filename, {type: mime});
                const dt = new DataTransfer();
                dt.items.add(file);

                const events = ["dragenter", "dragover", "drop"];

                function fire(el) {
                    let ok = [];
                    for (const name of events) {
                        const ev = new DragEvent(name, {
                            bubbles: true,
                            cancelable: true,
                            dataTransfer: dt,
                            clientX: window.innerWidth / 2,
                            clientY: window.innerHeight / 2,
                        });
                        const r = el.dispatchEvent(ev);
                        ok.push({name, result: r});
                    }
                    return ok;
                }

                const targets = [];
                targets.push(document);
                targets.push(document.body);
                targets.push(document.documentElement);

                // 选面积最大的 div/main/section
                const els = Array.from(document.querySelectorAll("main, section, div"));
                els.sort((a, b) => {
                    const ra = a.getBoundingClientRect();
                    const rb = b.getBoundingClientRect();
                    return (rb.width * rb.height) - (ra.width * ra.height);
                });

                for (const el of els.slice(0, 20)) {
                    targets.push(el);
                }

                const logs = [];
                for (const t of targets) {
                    try {
                        const rect = t.getBoundingClientRect ? t.getBoundingClientRect() : null;
                        logs.push({
                            tag: t.tagName || "document",
                            cls: t.className || "",
                            rect: rect ? {x: rect.x, y: rect.y, w: rect.width, h: rect.height} : null,
                            fired: fire(t),
                        });
                    } catch (e) {
                        logs.push({error: String(e)});
                    }
                }

                return logs;
            }
            """,
            {
                "b64": b64,
                "filename": os.path.basename(abs_rgb),
                "mime": mime,
            },
        )

        print("[Drop result count]", len(result))

        await page.wait_for_timeout(20000)
        await page.screenshot(path=os.path.join(out_dir, "02_after_drop.png"), full_page=True)

        buttons = []
        links = []
        inputs = []
        imgs = []

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

        try:
            imgs = await page.locator("img").evaluate_all(
                "(els) => els.map(e => ({src:e.src, w:e.naturalWidth, h:e.naturalHeight})).slice(0,20)"
            )
        except Exception:
            pass

        print("[After drop]")
        print("url:", page.url)
        print("title:", await page.title())
        print("buttons:", [b.strip().replace("\\n", " ") for b in buttons if b.strip()][:80])
        print("inputs:", inputs)
        print("links:", [a.strip().replace("\\n", " ") for a in links if a.strip()][:80])
        print("imgs:", imgs[:10])

        with open(os.path.join(out_dir, "drop_result.txt"), "w", encoding="utf-8") as f:
            for item in result:
                f.write(str(item) + "\n")

        html = await page.content()
        with open(os.path.join(out_dir, "02_after_drop.html"), "w", encoding="utf-8") as f:
            f.write(html)

        await context.close()
        await browser.close()


asyncio.run(main())
