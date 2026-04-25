"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

type ReplayControlsProps = {
  playing: boolean;
  speed: number;
  canStepBack: boolean;
  canStepForward: boolean;
  onPlayPause: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  onSpeedChange: (speed: number) => void;
};

export function ReplayControls({
  playing, speed, canStepBack, canStepForward,
  onPlayPause, onStepBack, onStepForward, onSpeedChange,
}: ReplayControlsProps) {
  return (
    <section className="dt-panel flex flex-wrap items-center gap-2 p-3">
      <p className="dt-label mr-2">Replay</p>
      <button onClick={onPlayPause} className="dt-btn-ghost rounded-lg p-2">
        {playing ? <Pause size={15} /> : <Play size={15} />}
      </button>
      <button onClick={onStepBack} disabled={!canStepBack} className="dt-btn-ghost rounded-lg p-2 disabled:opacity-20">
        <SkipBack size={15} />
      </button>
      <button onClick={onStepForward} disabled={!canStepForward} className="dt-btn-ghost rounded-lg p-2 disabled:opacity-20">
        <SkipForward size={15} />
      </button>

      <div className="ml-2 flex items-center gap-1.5">
        {[1, 2, 4].map((s) => (
          <button
            key={s}
            onClick={() => onSpeedChange(s)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
              s === speed
                ? "bg-orange-500/15 text-orange-300 ring-1 ring-orange-500/30"
                : "text-white/30 hover:bg-white/[0.04] hover:text-white/60"
            }`}
          >
            x{s}
          </button>
        ))}
      </div>
    </section>
  );
}
