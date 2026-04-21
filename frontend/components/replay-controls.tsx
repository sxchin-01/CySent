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
  playing,
  speed,
  canStepBack,
  canStepForward,
  onPlayPause,
  onStepBack,
  onStepForward,
  onSpeedChange,
}: ReplayControlsProps) {
  return (
    <section className="flex flex-wrap items-center gap-2 rounded-2xl border border-white/10 bg-black/25 p-3 backdrop-blur-xl">
      <p className="mr-2 text-xs uppercase tracking-[0.12em] text-slate-300">Replay Controls</p>
      <button
        onClick={onPlayPause}
        className="rounded-lg border border-white/15 bg-white/5 p-2 text-slate-100 transition hover:bg-white/10"
      >
        {playing ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <button
        onClick={onStepBack}
        disabled={!canStepBack}
        className="rounded-lg border border-white/15 bg-white/5 p-2 text-slate-100 transition disabled:opacity-40 hover:bg-white/10"
      >
        <SkipBack size={16} />
      </button>
      <button
        onClick={onStepForward}
        disabled={!canStepForward}
        className="rounded-lg border border-white/15 bg-white/5 p-2 text-slate-100 transition disabled:opacity-40 hover:bg-white/10"
      >
        <SkipForward size={16} />
      </button>

      <div className="ml-2 flex items-center gap-2">
        {[1, 2, 4].map((s) => (
          <button
            key={s}
            onClick={() => onSpeedChange(s)}
            className={`rounded-md px-2 py-1 text-xs transition ${
              s === speed
                ? "bg-cyan-300/20 text-cyan-100 border border-cyan-300/50"
                : "bg-white/5 text-slate-200 border border-white/15 hover:bg-white/10"
            }`}
          >
            x{s}
          </button>
        ))}
      </div>
    </section>
  );
}
