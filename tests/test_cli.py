import signal
from unittest.mock import Mock

import pytest

from face_camera import cli
from face_camera.config import Settings


def test_help_and_invalid_frame_limit():
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    with pytest.raises(SystemExit) as result:
        cli.main(["run", "--max-frames", "0"])
    assert result.value.code == 2


def test_missing_config_returns_failure():
    assert cli.main(["run", "--config", "missing.toml"]) == 1


@pytest.mark.parametrize(
    "arguments, expected", [([], True), (["--show"], False), (["--headless"], True)]
)
def test_display_flag_defaults(tmp_path, monkeypatch, arguments, expected):
    monkeypatch.chdir(tmp_path)
    run = Mock(return_value=0)
    monkeypatch.setattr(cli, "run", run)
    assert cli.main(["run", *arguments]) == 0
    assert run.call_args.args[0].headless is expected


@pytest.mark.parametrize("fail", [False, True])
def test_max_frames_and_processing_failure_close_stream(tmp_path, frame, monkeypatch, fail):
    engine = Mock()
    engine.recognize.return_value = []
    if fail:
        engine.recognize.side_effect = ValueError("inference failed")
    monkeypatch.setattr(cli, "FaceEngine", lambda _: engine)
    closed = []

    def stream(*_):
        try:
            while True:
                yield frame
        finally:
            closed.append(True)

    monkeypatch.setattr(cli, "frames", stream)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    settings = Settings(output_dir=tmp_path)
    if fail:
        with pytest.raises(ValueError, match="inference failed"):
            cli.run(settings, max_frames=2)
    else:
        assert cli.run(settings, max_frames=2) == 0
        assert engine.recognize.call_count == 2
    assert closed == [True]
    assert handlers == {sig: signal.getsignal(sig) for sig in handlers}
