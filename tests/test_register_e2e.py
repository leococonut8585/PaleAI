import os
from playwright.sync_api import sync_playwright


def test_register_progress_page_loads():
    html_path = os.path.abspath("static/register_progress.html")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_path}")
        assert "登録中" in page.content()
        browser.close()


def test_generated_profile_image_display_size():
    html_path = os.path.abspath("static/profile_generated.html")
    img_path = os.path.abspath("static/pic/Default.png")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        def handle(route, request):
            route.fulfill(path=img_path)

        page.route("**/static/profile/*", handle)
        page.goto(f"file://{html_path}?uid=1")
        page.wait_for_selector("#avatar")
        width = page.evaluate("document.getElementById('avatar').clientWidth")
        assert width <= 256
        browser.close()

