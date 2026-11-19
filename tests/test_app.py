"""Headless smoke test: every dashboard page renders without an exception."""
import pytest

from naf.config import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "results" / "summary_primary.csv").exists(),
                                reason="results not built")

PAGES = ["Overview", "Dataset", "Replay", "Model comparison", "Drift & unseen families",
         "Alert-policy explorer", "About & limitations"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "Replay":
        assert any("identical" in s.value for s in at.success)
