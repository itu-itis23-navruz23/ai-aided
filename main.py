import threading
import queue
import time
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import List, Dict, Set, Tuple, Optional, DefaultDict


class SimpleHTMLTextParser(HTMLParser):
    """
    Minimal HTML parser to extract:
    - <title> text
    - A small snippet of visible text (for preview)
    - Links (<a href="...">)

    This is intentionally simple and uses only the standard library.
    """

    def __init__(self, max_snippet_chars: int = 200):
        super().__init__()
        self.in_title = False
        self.in_script_or_style = False
        self.title_parts: List[str] = []
        self.snippet_parts: List[str] = []
        self.links: List[str] = []
        self.max_snippet_chars = max_snippet_chars

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()

        if tag_lower == "title":
            self.in_title = True
        elif tag_lower in ("script", "style"):
            self.in_script_or_style = True

        if tag_lower == "a":
            # Extract href attribute from <a> tag
            for (attr, value) in attrs:
                if attr.lower() == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self.in_title = False
        elif tag_lower in ("script", "style"):
            self.in_script_or_style = False

    def handle_data(self, data: str) -> None:
        # Ignore text inside <script> or <style> tags
        if self.in_script_or_style:
            return

        text = data.strip()
        if not text:
            return

        if self.in_title:
            self.title_parts.append(text)

        # Build up a snippet from visible text nodes until we hit the limit
        if len(" ".join(self.snippet_parts)) < self.max_snippet_chars:
            self.snippet_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def snippet(self) -> str:
        raw = " ".join(self.snippet_parts).strip()
        # Truncate to max_snippet_chars, being careful with length checks
        if len(raw) > self.max_snippet_chars:
            return raw[: self.max_snippet_chars].rstrip() + "..."
        return raw

    def get_links(self) -> List[str]:
        return self.links


class Crawler:
    """
    Core concurrent web crawler.

    Responsibilities:
    - Start from an origin URL and crawl up to a specified max depth.
    - Use a bounded queue (back-pressure) to coordinate work among threads.
    - Maintain a thread-safe visited set to avoid re-crawling the same URL.
    - For each page, extract:
        * URL
        * <title>
        * A small text snippet
      and store them in a thread-safe list.
    - Use only Python standard libraries (urllib + html.parser + threading + queue).

    This class does *not* implement indexing or search yet; it focuses solely
    on concurrent crawling and basic metadata extraction.
    """

    def __init__(
        self,
        origin_url: str,
        max_depth: int,
        max_queue_size: int = 100,
        num_workers: int = 4,
        max_snippet_chars: int = 200,
        max_pages: int = 200,
    ) -> None:
        """
        :param origin_url: Starting URL for the crawl.
        :param max_depth: Maximum depth to crawl (0 means only origin).
        :param max_queue_size: Maximum size for the work queue (back-pressure).
        :param num_workers: Number of worker threads.
        :param max_snippet_chars: Maximum characters for text snippet per page.
        """
        self.origin_url = origin_url
        self.max_depth = max_depth
        self.num_workers = max(1, num_workers)

        # Bounded queue to implement back-pressure. Each item is (url, depth, referrer)
        self.queue: "queue.Queue[Tuple[str, int, Optional[str]]]" = queue.Queue(
            maxsize=max_queue_size
        )

        # Thread-safe visited set to avoid revisiting URLs
        self.visited: Set[str] = set()
        self.visited_lock = threading.Lock()

        # Thread-safe storage for page results. Each entry:
        #   {"url": ..., "title": ..., "snippet": ..., "depth": ..., "origin": ...}
        self.pages: List[Dict[str, str]] = []
        self.pages_lock = threading.Lock()

        # Thread-safe inverted index:
        #   word -> list of (url, origin, depth, is_title)
        # This allows concurrent searching while crawling proceeds.
        self.index: DefaultDict[
            str, List[Tuple[str, str, int, bool]]
        ] = DefaultDict(list)
        self.index_lock = threading.Lock()

        # Misc configuration
        self.max_snippet_chars = max_snippet_chars
        self.max_pages = max_pages

        # Keep references to worker threads so we can join them
        self.workers: List[threading.Thread] = []

        # Simple counter to help with progress/debug output
        self._pages_processed = 0
        self._pages_processed_lock = threading.Lock()

    def _safe_add_visited(self, url: str) -> bool:
        """
        Add URL to visited set in a thread-safe way.

        :return: True if the URL was newly added (i.e., not visited before).
        """
        with self.visited_lock:
            if url in self.visited:
                return False
            self.visited.add(url)
            return True

    def _store_page(
        self, url: str, title: str, snippet: str, depth: int, origin: Optional[str]
    ) -> None:
        """
        Store page information in a thread-safe list.
        """
        record = {
            "url": url,
            "title": title,
            "snippet": snippet,
            "depth": str(depth),
            "origin": origin or "",
        }
        with self.pages_lock:
            self.pages.append(record)

        # Update the inverted index for this page so that search can run
        # concurrently with crawling. The index is kept deliberately simple.
        self._add_to_index(url, origin or "", depth, title, snippet)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Basic tokenizer: lowercase, split on whitespace, strip simple punctuation.
        This is intentionally minimal and relies only on the standard library.
        """
        tokens: List[str] = []
        for raw in text.lower().split():
            # Strip leading/trailing punctuation characters
            token = raw.strip(".,;:!?\"'()[]{}<>")
            if token:
                tokens.append(token)
        return tokens

    def _add_to_index(
        self,
        url: str,
        origin: str,
        depth: int,
        title: str,
        snippet: str,
    ) -> None:
        """
        Update the thread-safe inverted index for the given page.

        For each token in the title/snippet we append a posting:
            (url, origin, depth, is_title)

        This allows a simple frequency-based ranking in the SearchEngine.
        """
        title_tokens = self._tokenize(title)
        snippet_tokens = self._tokenize(snippet)

        with self.index_lock:
            for tok in title_tokens:
                self.index[tok].append((url, origin, depth, True))
            for tok in snippet_tokens:
                self.index[tok].append((url, origin, depth, False))

    def _fetch_url(self, url: str) -> Optional[bytes]:
        """
        Fetch the given URL using urllib.request.

        :return: Raw response bytes, or None on error.
        """
        try:
            req = urllib.request.Request(
                url,
                headers={
                    # Basic user-agent to avoid some trivial blocks
                    "User-Agent": "GoogleInADayCrawler/1.0 (standard-library-only)"
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                # Only handle basic HTML/text content types
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return None
                return resp.read()
        except Exception:
            # In a production system, we would log errors here
            return None

    def _parse_html(self, html_bytes: bytes) -> Tuple[str, str, List[str]]:
        """
        Parse HTML bytes and extract:
        - title
        - snippet
        - list of href links (possibly relative)

        :return: (title, snippet, links)
        """
        try:
            # Attempt to decode as UTF-8 with fallback replacement
            html_text = html_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return "", "", []

        parser = SimpleHTMLTextParser(max_snippet_chars=self.max_snippet_chars)
        parser.feed(html_text)
        parser.close()

        return parser.title, parser.snippet, parser.get_links()

    def _normalize_and_join_url(self, base_url: str, link: str) -> Optional[str]:
        """
        Normalize and resolve a possibly relative URL (`link`) against `base_url`.

        Returns None if the result is not an HTTP(S) URL.
        """
        # Resolve relative URLs against the base URL
        joined = urllib.parse.urljoin(base_url, link)
        parsed = urllib.parse.urlparse(joined)

        # Only accept http and https schemes
        if parsed.scheme not in ("http", "https"):
            return None

        # Normalize by removing fragments
        normalized = parsed._replace(fragment="")

        return urllib.parse.urlunparse(normalized)

    def _worker(self) -> None:
        """
        Worker thread function.

        Repeatedly:
        - Take a URL from the queue.
        - Fetch and parse the page.
        - Store its metadata.
        - Enqueue newly discovered links if depth allows.
        """
        while True:
            item = self.queue.get()

            # Ensure we always mark the task as done, even if exceptions occur
            try:
                # Sentinel value used to signal worker shutdown
                if item is None:
                    print(f"[{threading.current_thread().name}] Received stop sentinel.")
                    break

                url, depth, origin = item

                # Skip URLs that have already been visited by another worker
                if not self._safe_add_visited(url):
                    continue

                print(
                    f"[{threading.current_thread().name}] "
                    f"Processing depth={depth} url={url}"
                )

                # Fetch the URL
                body = self._fetch_url(url)
                if body is None:
                    print(
                        f"[{threading.current_thread().name}] "
                        f"Failed to fetch or non-HTML content: {url}"
                    )
                    continue

                # Parse HTML
                title, snippet, links = self._parse_html(body)

                # Store page information
                self._store_page(url, title, snippet, depth, origin)

                with self._pages_processed_lock:
                    self._pages_processed += 1
                    if self._pages_processed % 5 == 0:
                        print(
                            f"[{threading.current_thread().name}] "
                            f"Total pages processed so far: {self._pages_processed}"
                        )

                # If we haven't reached max depth and have not hit the page cap,
                # enqueue discovered links. We use non-blocking put to avoid
                # deadlocks from back-pressure; links are simply dropped when
                # the queue is full.
                with self.pages_lock:
                    should_expand = (
                        depth < self.max_depth and len(self.pages) < self.max_pages
                    )

                if should_expand:
                    for link in links:
                        normalized = self._normalize_and_join_url(url, link)
                        if not normalized:
                            continue

                        # Check visited here to avoid overfilling the queue
                        with self.visited_lock:
                            if normalized in self.visited:
                                continue

                        try:
                            self.queue.put_nowait((normalized, depth + 1, url))
                        except queue.Full:
                            # Back-pressure: drop excess links instead of blocking.
                            print(
                                f"[{threading.current_thread().name}] "
                                "Queue full; dropping discovered link."
                            )
                            continue
            finally:
                self.queue.task_done()

    def crawl(self) -> List[Dict[str, str]]:
        """
        Start the crawling process and block until all work is done.

        :return: List of page metadata dictionaries collected during the crawl.
        """
        print(f"[Main] Starting crawl from {self.origin_url} with depth {self.max_depth}")

        # Seed the queue with the origin URL at depth 0
        self.queue.put((self.origin_url, 0, None))

        # Start worker threads
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, name=f"CrawlerWorker-{i}")
            t.daemon = True  # Allows process to exit if main thread finishes
            t.start()
            self.workers.append(t)

        print(f"[Main] Spawned {self.num_workers} worker threads.")

        # Wait until the queue is fully processed
        self.queue.join()

        print("[Main] Queue fully processed; sending stop sentinels.")

        # Send one sentinel per worker to gracefully stop them
        for _ in self.workers:
            # Each sentinel is a separate task that will be consumed and marked done
            self.queue.put(None)

        # Wait for all workers to exit
        for t in self.workers:
            t.join(timeout=5.0)
            if t.is_alive():
                print(
                    f"[Main] Warning: worker {t.name} did not terminate within timeout."
                )

        # Return a snapshot of collected pages
        with self.pages_lock:
            return list(self.pages)


class SearchEngine:
    """
    Simple, thread-safe search engine over the crawler's live index.

    Search works concurrently with crawling by relying on the crawler's
    `index` structure, which is protected by `index_lock`.
    """

    def __init__(self, crawler: Crawler) -> None:
        self.crawler = crawler

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Search for pages relevant to the given query string.

        Returns a list of triples:
            (relevant_url, origin_url, depth)

        Ranking heuristic (very simple):
        - Tokenize the query.
        - For each query token, look up postings in the inverted index.
        - For each posting:
            * +2 score if the token came from the title
            * +1 score if the token came from the snippet
        - Aggregate scores across tokens and sort descending.
        """
        q = query.strip().lower()
        if not q:
            return []

        tokens = Crawler._tokenize(q)
        if not tokens:
            return []

        scores: Dict[Tuple[str, str, int], int] = {}

        with self.crawler.index_lock:
            for tok in tokens:
                postings = self.crawler.index.get(tok, [])
                for url, origin, depth, is_title in postings:
                    key = (url, origin, depth)
                    weight = 2 if is_title else 1
                    scores[key] = scores.get(key, 0) + weight

        # Sort results by score (descending), then by depth (ascending)
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0][2])
        )

        top_results: List[Tuple[str, str, int]] = []
        for (url, origin, depth), _score in ranked[:top_k]:
            top_results.append((url, origin, depth))

        return top_results


def dashboard_loop(
    crawler: Crawler,
    stop_event: threading.Event,
    controller_thread: threading.Thread,
    interval_seconds: float = 2.0,
) -> None:
    """
    Background reporting loop that periodically prints:
    - Indexing progress (pages crawled vs. pending in queue)
    - Current queue depth
    - Back-pressure status ("Queue Full" or "Normal")

    This runs concurrently with the crawler and the interactive search loop.
    """
    while not stop_event.is_set():
        try:
            with crawler.pages_lock:
                pages_crawled = len(crawler.pages)

            queue_depth = crawler.queue.qsize()
            queue_full = crawler.queue.full()

            status = "Queue Full" if queue_full else "Normal"

            print(
                f"[Dashboard] Pages crawled: {pages_crawled}, "
                f"Queue depth: {queue_depth}, "
                f"Back-pressure: {status}"
            )

            # If the controller thread is finished and queue is empty,
            # we can exit the dashboard loop after one last report.
            if not controller_thread.is_alive() and queue_depth == 0:
                print("[Dashboard] Crawler appears to be finished.")
                break

            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            # Allow graceful shutdown via Ctrl+C
            break


if __name__ == "__main__":
    """
    Interactive test harness for the concurrent Crawler + SearchEngine.

    - Starts the crawler in a background thread.
    - Starts a dashboard thread that reports metrics every few seconds.
    - Provides a simple REPL so you can run search queries in real time.
    """
    # Choose a conservative test site
    start_url = "https://quotes.toscrape.com"
    max_depth = 2

    crawler = Crawler(
        origin_url=start_url,
        max_depth=max_depth,
        max_queue_size=50,
        num_workers=4,
        max_snippet_chars=200,
        max_pages=150,  # keep the crawl bounded for testing
    )

    search_engine = SearchEngine(crawler)

    # Event to signal the dashboard loop to stop
    stop_event = threading.Event()

    # Start the crawler in its own controller thread so that the main thread
    # can remain responsive for user input.
    controller_thread = threading.Thread(
        target=crawler.crawl, name="CrawlerController"
    )
    controller_thread.start()

    # Start the dashboard reporting thread
    dashboard_thread = threading.Thread(
        target=dashboard_loop,
        args=(crawler, stop_event, controller_thread),
        name="Dashboard",
        daemon=True,
    )
    dashboard_thread.start()

    print("\nInteractive search is ready.")
    print("Type your query and press Enter to search.")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            query = input("Enter search query: ").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("Exiting interactive search...")
                break

            results = search_engine.search(query, top_k=10)
            if not results:
                print("No results found.\n")
                continue

            print(f"Top results for '{query}':")
            for i, (url, origin, depth) in enumerate(results, start=1):
                print(f"  {i}. [depth={depth}] {url} (origin: {origin})")
            print()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")

    # Signal dashboard to stop and wait for crawler to finish
    stop_event.set()
    controller_thread.join()
    # Dashboard is daemon=True; it will exit automatically when main exits.

    print("Done.")