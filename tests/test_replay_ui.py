"""Tests for the Replay tab animation controls.

Uses ``streamlit.testing.v1.AppTest`` to exercise the slider-sync,
play/pause/reset, and loop logic that lives in ``network_attack/ui.py``
without launching a browser.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


# ---------------------------------------------------------------------------
# Minimal replay widget extracted from ui.py – keeps only the control
# logic so that we can test state transitions without network I/O,
# heavy Plotly rendering, or ``time.sleep`` blocking.
# ---------------------------------------------------------------------------

_REPLAY_SCRIPT = """
import streamlit as st

# ---- bootstrap replay data (normally done by "Prepare Replay") ----
if "replay_snapshots" not in st.session_state:
    st.session_state["replay_snapshots"] = list(range(10))  # 10 fake epochs
    st.session_state["replay_epoch_idx"] = 0
    st.session_state["replay_playing"] = False

snapshots = st.session_state["replay_snapshots"]

# ---- playback buttons (same logic as ui.py) ----
cols = st.columns(4)
with cols[0]:
    if st.button("Play", key="replay_play"):
        st.session_state["replay_playing"] = True
with cols[1]:
    if st.button("Pause", key="replay_pause"):
        st.session_state["replay_playing"] = False
with cols[2]:
    if st.button("Reset", key="replay_reset"):
        st.session_state["replay_playing"] = False
        st.session_state["replay_epoch_idx"] = 0
with cols[3]:
    loop_animation = st.checkbox("Loop", key="replay_loop")

is_playing = st.session_state.get("replay_playing", False)

# ---- slider with on_change callback (exact same pattern as ui.py) ----
def _on_slider_change() -> None:
    st.session_state["replay_playing"] = False
    st.session_state["replay_epoch_idx"] = st.session_state["replay_slider"]

current_idx = st.session_state.get("replay_epoch_idx", 0)
st.session_state["replay_slider"] = current_idx          # <-- THE FIX

epoch_idx = st.slider(
    "Epoch",
    0,
    len(snapshots) - 1,
    value=current_idx,
    key="replay_slider",
    on_change=_on_slider_change,
)
st.session_state["replay_epoch_idx"] = epoch_idx

# ---- auto-advance (same as ui.py but WITHOUT time.sleep / st.rerun
#      so that AppTest doesn't block) ----
at_end = epoch_idx >= len(snapshots) - 1
if is_playing and not at_end:
    st.session_state["replay_epoch_idx"] = epoch_idx + 1
    # In the real app this is followed by time.sleep + st.rerun().
    # We skip that here; the test simulates the rerun by calling .run() again.
elif is_playing and at_end:
    if loop_animation:
        st.session_state["replay_epoch_idx"] = 0
    else:
        st.session_state["replay_playing"] = False
"""


@pytest.fixture()
def app() -> AppTest:
    """Return a freshly initialised ``AppTest`` instance."""
    at = AppTest.from_string(_REPLAY_SCRIPT, default_timeout=5)
    at.run(timeout=5)
    assert not at.exception, at.exception
    return at


class TestReplayControls:
    """Verify play / pause / reset / slider-sync behaviour."""

    def test_initial_state(self, app: AppTest) -> None:
        """After first run the slider sits at epoch 0 and is not playing."""
        assert app.slider[0].value == 0
        assert app.session_state["replay_playing"] is False
        assert app.session_state["replay_epoch_idx"] == 0

    def test_play_advances_epoch(self, app: AppTest) -> None:
        """Clicking Play should advance replay_epoch_idx by 1 each run."""
        app.button[0].click()          # "Play"
        app.run(timeout=5)
        assert app.session_state["replay_playing"] is True
        # auto-advance incremented the index
        assert app.session_state["replay_epoch_idx"] == 1

        # Simulate another rerun (the real app calls st.rerun())
        app.run(timeout=5)
        assert app.session_state["replay_epoch_idx"] == 2

    def test_slider_syncs_with_auto_advance(self, app: AppTest) -> None:
        """The slider widget must reflect the auto-advanced epoch.

        This is the exact bug that was reported: without syncing
        ``replay_slider`` before the widget renders, the slider stays
        stuck at the old value and overwrites ``replay_epoch_idx``
        back, creating an infinite loop.
        """
        app.button[0].click()          # "Play"
        app.run(timeout=5)             # renders epoch 0 → advances to 1
        app.run(timeout=5)             # renders epoch 1 → advances to 2
        app.run(timeout=5)             # renders epoch 2 → advances to 3

        # The slider shows the epoch rendered during the last run (2),
        # while replay_epoch_idx has already been advanced to 3 for
        # the next render.  The key check: the slider is NOT stuck at
        # 0 (the old broken behaviour) and correctly keeps up.
        assert app.slider[0].value == 2
        assert app.session_state["replay_epoch_idx"] == 3

        # One more run to confirm it keeps advancing
        app.run(timeout=5)             # renders epoch 3 → advances to 4
        assert app.slider[0].value == 3
        assert app.session_state["replay_epoch_idx"] == 4

    def test_pause_stops_advance(self, app: AppTest) -> None:
        """Pressing Pause should freeze the epoch."""
        app.button[0].click()          # Play
        app.run(timeout=5)
        app.run(timeout=5)             # epoch should be 2

        app.button[1].click()          # Pause
        app.run(timeout=5)
        frozen = app.session_state["replay_epoch_idx"]

        app.run(timeout=5)
        assert app.session_state["replay_epoch_idx"] == frozen

    def test_reset_returns_to_zero(self, app: AppTest) -> None:
        """Reset should jump back to epoch 0 and stop playback."""
        app.button[0].click()          # Play
        app.run(timeout=5)
        app.run(timeout=5)             # advance to epoch 2

        app.button[2].click()          # Reset
        app.run(timeout=5)

        assert app.session_state["replay_epoch_idx"] == 0
        assert app.session_state["replay_playing"] is False
        assert app.slider[0].value == 0

    def test_manual_slider_pauses_playback(self, app: AppTest) -> None:
        """Dragging the slider should pause playback and jump."""
        app.button[0].click()          # Play
        app.run(timeout=5)             # playing, auto-advanced

        # User drags slider to epoch 7
        app.slider[0].set_value(7).run(timeout=5)

        assert app.session_state["replay_playing"] is False
        assert app.session_state["replay_epoch_idx"] == 7
        assert app.slider[0].value == 7

    def test_loop_wraps_to_zero(self, app: AppTest) -> None:
        """With Loop enabled, reaching the last epoch should wrap to 0."""
        # Enable loop checkbox
        app.checkbox[0].check().run(timeout=5)

        # Jump to just before the end
        app.session_state["replay_epoch_idx"] = 8
        app.session_state["replay_playing"] = True
        app.run(timeout=5)             # epoch 8 → auto-advance to 9
        assert app.session_state["replay_epoch_idx"] == 9

        app.run(timeout=5)             # epoch 9 (end) → loop to 0
        assert app.session_state["replay_epoch_idx"] == 0
        assert app.session_state["replay_playing"] is True  # still playing

    def test_no_loop_stops_at_end(self, app: AppTest) -> None:
        """Without Loop, reaching the end should stop playback."""
        app.session_state["replay_epoch_idx"] = 8
        app.session_state["replay_playing"] = True
        app.run(timeout=5)             # epoch 8 → auto-advance to 9

        app.run(timeout=5)             # epoch 9 (end) → stops
        assert app.session_state["replay_playing"] is False
