import os
import sys
import unittest
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cons
from utilities.webscraper import gen_urls

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# --- cat search, 300 images -> 3 pages (skip 0, 100, 200) ---
exp_urls_cat_300 = [
    "https://free-images.com/search/?q=cat&skip=0",
    "https://free-images.com/search/?q=cat&skip=100",
    "https://free-images.com/search/?q=cat&skip=200",
]
obs_urls_cat_300 = gen_urls(search="cat", n_images=300, home_url=cons.home_url)

# --- dog search, 100 images -> 1 page ---
exp_urls_dog_100 = [
    "https://free-images.com/search/?q=dog&skip=0",
]
obs_urls_dog_100 = gen_urls(search="dog", n_images=100, home_url=cons.home_url)

# --- 250 images -> ceil(250/100) = 3 pages ---
exp_urls_cat_250_len = 3
obs_urls_cat_250 = gen_urls(search="cat", n_images=250, home_url=cons.home_url)

# --- 1 image -> 1 page ---
obs_urls_single = gen_urls(search="cat", n_images=1, home_url=cons.home_url)

# --- exactly 200 images -> 2 pages ---
obs_urls_200 = gen_urls(search="cat", n_images=200, home_url=cons.home_url)

# --- large n = 1000 images -> 10 pages ---
obs_urls_1000 = gen_urls(search="cat", n_images=1000, home_url=cons.home_url)

# --- search term with spaces / special chars (URL-encoded) ---
obs_urls_multi = gen_urls(search="persian cat", n_images=100, home_url=cons.home_url)

# --- custom home_url ---
custom_home = "https://example.com"
obs_urls_custom_home = gen_urls(search="cat", n_images=100, home_url=custom_home)


class TestGenUrls(unittest.TestCase):
    """
    Comprehensive unit tests for gen_urls.

    Verifies return type, list length (pages = ceil(n_images/100)),
    URL format, skip parameter increments, search-term injection,
    no duplicates, query-string structure, boundary values (200 exact,
    1000 images), custom home URL, and multi-word search terms.
    """

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type_cat_300(self):
        self.assertEqual(type(obs_urls_cat_300), type(exp_urls_cat_300))

    def test_type_dog_100(self):
        self.assertIsInstance(obs_urls_dog_100, list)

    def test_type_elements_are_strings(self):
        for url in obs_urls_cat_300:
            self.assertIsInstance(url, str)

    # ------------------------------------------------------------------
    # Length tests
    # ------------------------------------------------------------------

    def test_len_cat_300(self):
        self.assertEqual(len(obs_urls_cat_300), len(exp_urls_cat_300))

    def test_len_dog_100(self):
        self.assertEqual(len(obs_urls_dog_100), len(exp_urls_dog_100))

    def test_len_cat_250(self):
        """ceil(250 / 100) = 3 pages."""
        self.assertEqual(len(obs_urls_cat_250), exp_urls_cat_250_len)

    def test_len_single_image(self):
        """Even a single requested image should yield exactly 1 URL."""
        self.assertEqual(len(obs_urls_single), 1)

    def test_len_exactly_200(self):
        """200 images -> exactly 2 pages (no over-counting)."""
        self.assertEqual(len(obs_urls_200), 2)

    def test_len_1000(self):
        """1000 images -> 10 pages."""
        self.assertEqual(len(obs_urls_1000), 10)

    # ------------------------------------------------------------------
    # Value tests (exact URL match)
    # ------------------------------------------------------------------

    def test_values_cat_300(self):
        self.assertEqual(obs_urls_cat_300, exp_urls_cat_300)

    def test_values_dog_100(self):
        self.assertEqual(obs_urls_dog_100, exp_urls_dog_100)

    # ------------------------------------------------------------------
    # URL format tests
    # ------------------------------------------------------------------

    def test_all_urls_start_with_home_url(self):
        for url in obs_urls_cat_300:
            self.assertTrue(url.startswith(cons.home_url))

    def test_custom_home_url_used(self):
        for url in obs_urls_custom_home:
            self.assertTrue(url.startswith(custom_home))

    def test_search_term_in_all_urls(self):
        for url in obs_urls_cat_300:
            self.assertIn("cat", url)

    def test_skip_param_in_all_urls(self):
        for url in obs_urls_cat_300:
            self.assertIn("skip=", url)

    def test_query_param_q_present(self):
        """Every URL must contain the q= query parameter."""
        for url in obs_urls_cat_300:
            self.assertIn("q=", url)

    # ------------------------------------------------------------------
    # Query-string structure tests (parsed)
    # ------------------------------------------------------------------

    def test_query_string_q_value(self):
        """Parsed q= value should equal the search term."""
        for url in obs_urls_cat_300:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            self.assertIn("q", params)
            self.assertEqual(params["q"][0], "cat")

    def test_query_string_skip_value_first(self):
        parsed = urlparse(obs_urls_cat_300[0])
        params = parse_qs(parsed.query)
        self.assertEqual(params["skip"][0], "0")

    def test_query_string_skip_value_second(self):
        parsed = urlparse(obs_urls_cat_300[1])
        params = parse_qs(parsed.query)
        self.assertEqual(params["skip"][0], "100")

    def test_query_string_exactly_two_params(self):
        """Each URL should have exactly 2 query params: q and skip."""
        for url in obs_urls_cat_300:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            self.assertEqual(len(params), 2)

    # ------------------------------------------------------------------
    # Skip increment tests
    # ------------------------------------------------------------------

    def test_first_url_skip_is_zero(self):
        self.assertIn("skip=0", obs_urls_cat_300[0])

    def test_second_url_skip_is_100(self):
        self.assertIn("skip=100", obs_urls_cat_300[1])

    def test_third_url_skip_is_200(self):
        self.assertIn("skip=200", obs_urls_cat_300[2])

    def test_skip_increments_by_100(self):
        """Each successive URL must have a skip value 100 greater than the previous."""
        for i, url in enumerate(obs_urls_cat_300):
            expected_skip = i * 100
            self.assertIn(f"skip={expected_skip}", url)

    def test_skip_increments_1000(self):
        """For 1000 images the skip should go 0, 100, 200, … 900."""
        for i, url in enumerate(obs_urls_1000):
            self.assertIn(f"skip={i * 100}", url)

    # ------------------------------------------------------------------
    # No-duplicate tests
    # ------------------------------------------------------------------

    def test_no_duplicate_urls_cat_300(self):
        self.assertEqual(len(obs_urls_cat_300), len(set(obs_urls_cat_300)))

    def test_no_duplicate_urls_1000(self):
        self.assertEqual(len(obs_urls_1000), len(set(obs_urls_1000)))

    # ------------------------------------------------------------------
    # Search-term substitution tests
    # ------------------------------------------------------------------

    def test_dog_search_term_in_urls(self):
        for url in obs_urls_dog_100:
            self.assertIn("dog", url)

    def test_cat_term_not_in_dog_urls(self):
        for url in obs_urls_dog_100:
            self.assertNotIn("cat", url)

    def test_multi_word_search_produces_urls(self):
        """Multi-word search terms should produce a non-empty list."""
        self.assertGreater(len(obs_urls_multi), 0)

    def test_multi_word_search_term_in_url(self):
        """The search term (possibly URL-encoded) should appear in each URL."""
        for url in obs_urls_multi:
            # urllib may encode the space; check for both possibilities
            self.assertTrue("persian" in url or "persian+cat" in url or "persian%20cat" in url)

    # ------------------------------------------------------------------
    # URL ordering tests
    # ------------------------------------------------------------------

    def test_urls_ordered_by_skip(self):
        """URLs must be ordered with ascending skip values."""
        skip_values = []
        for url in obs_urls_cat_300:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            skip_values.append(int(params["skip"][0]))
        self.assertEqual(skip_values, sorted(skip_values))


if __name__ == "__main__":
    unittest.main()
