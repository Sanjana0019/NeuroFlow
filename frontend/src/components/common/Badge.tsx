import clsx from "clsx";

interface ScoreBadgeProps {
  score: number | null | undefined;
  label?: string;
  className?: string;
}

export function ScoreBadge({ score, label, className }: ScoreBadgeProps) {
  if (score === null || score === undefined) {
    return (
      <span
        className={clsx(
          "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium font-mono bg-slate-800 text-slate-400 border border-slate-700",
          className
        )}
      >
        {label ? `${label}: ` : ""}N/A
      </span>
    );
  }

  let colorStyle = "bg-emerald-950/60 text-emerald-400 border-emerald-800/50";
  if (score < 0.6) {
    colorStyle = "bg-rose-950/60 text-rose-400 border-rose-800/50";
  } else if (score <= 0.8) {
    colorStyle = "bg-amber-950/60 text-amber-400 border-amber-800/50";
  }

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold font-mono border",
        colorStyle,
        className
      )}
    >
      {label ? `${label}: ` : ""}
      {(score * 100).toFixed(0)}%
    </span>
  );
}

interface StatusBadgeProps {
  status: "pending" | "processing" | "completed" | "failed" | string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  switch (status) {
    case "completed":
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 mr-1.5" />
          Completed
        </span>
      );
    case "processing":
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-950/60 text-blue-400 border border-blue-800/40">
          <span className="h-2 w-2 rounded-full bg-blue-400 animate-ping mr-1.5" />
          Processing
        </span>
      );
    case "pending":
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-950/60 text-amber-400 border border-amber-800/40">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400 mr-1.5" />
          Pending
        </span>
      );
    case "failed":
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-rose-950/60 text-rose-400 border border-rose-800/40">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-400 mr-1.5" />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-800 text-slate-400">
          {status}
        </span>
      );
  }
}
