"""Legacy launcher for interactive recording replay."""

from tools.recording.replay_recording import parse_args, run_viewer


if __name__ == "__main__":
    run_viewer(parse_args())

