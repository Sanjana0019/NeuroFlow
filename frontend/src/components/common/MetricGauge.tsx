import clsx from "clsx";

interface MetricGaugeProps {
  label: string;
  score: number | null | undefined;
  weight?: string;
}

export function MetricGauge({ label, score, weight }: MetricGaugeProps) {
  const value = score !== null && score !== undefined ? Math.round(score * 100) : null;

  let barColor = "bg-emerald-500";
  let textColor = "text-emerald-400";
  if (value !== null) {
    if (value < 60) {
      barColor = "bg-rose-500";
      textColor = "text-rose-400";
    } else if (value <= 80) {
      barColor = "bg-amber-500";
      textColor = "text-amber-400";
    }
  }

  return (
    <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-300">
          {label}
          {weight && <span className="text-[10px] text-slate-500 ml-1">({weight})</span>}
        </span>
        <span className={clsx("text-xs font-bold font-mono", value !== null ? textColor : "text-slate-500")}>
          {value !== null ? `${value}%` : "—"}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-700 ease-out", barColor)}
          style={{ width: `${value ?? 0}%` }}
        />
      </div>
    </div>
  );
}
