import { Evaluation } from "../../types";
import { ScoreBadge } from "../common/Badge";

interface CompareScorecardProps {
  nameA?: string;
  nameB?: string;
  evalA: Evaluation | null;
  evalB: Evaluation | null;
}

export function CompareScorecard({
  nameA = "Pipeline A",
  nameB = "Pipeline B",
  evalA,
  evalB,
}: CompareScorecardProps) {
  const metrics = [
    { label: "Faithfulness", keyA: evalA?.faithfulness, keyB: evalB?.faithfulness, weight: "35%" },
    { label: "Answer Relevance", keyA: evalA?.answer_relevance, keyB: evalB?.answer_relevance, weight: "30%" },
    { label: "Context Precision", keyA: evalA?.context_precision, keyB: evalB?.context_precision, weight: "20%" },
    { label: "Context Recall", keyA: evalA?.context_recall, keyB: evalB?.context_recall, weight: "15%" },
    { label: "Overall Quality", keyA: evalA?.overall_score, keyB: evalB?.overall_score, weight: "100%", isOverall: true },
  ];

  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900 shadow-xl">
      <div className="px-4 py-3 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
        <h4 className="text-xs font-semibold text-slate-200 tracking-wide uppercase">
          Comparative Evaluation Scorecard
        </h4>
        <span className="text-[10px] font-mono text-slate-500">Auto-Evaluated via LLM Judge</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/40 text-slate-400">
              <th className="px-4 py-2.5 font-medium">Metric</th>
              <th className="px-4 py-2.5 font-medium text-right text-indigo-400">{nameA}</th>
              <th className="px-4 py-2.5 font-medium text-right text-purple-400">{nameB}</th>
              <th className="px-4 py-2.5 font-medium text-right text-slate-400">Delta</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-300">
            {metrics.map((m) => {
              const delta =
                m.keyA !== null && m.keyA !== undefined && m.keyB !== null && m.keyB !== undefined
                  ? m.keyA - m.keyB
                  : null;

              return (
                <tr
                  key={m.label}
                  className={m.isOverall ? "bg-indigo-950/20 font-semibold" : "hover:bg-slate-800/30"}
                >
                  <td className="px-4 py-2.5 flex items-center gap-1.5 font-sans">
                    <span>{m.label}</span>
                    <span className="text-[10px] text-slate-500 font-mono">({m.weight})</span>
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <ScoreBadge score={m.keyA} />
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <ScoreBadge score={m.keyB} />
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-[11px]">
                    {delta !== null ? (
                      <span
                        className={
                          delta > 0.01
                            ? "text-indigo-400 font-bold"
                            : delta < -0.01
                            ? "text-purple-400 font-bold"
                            : "text-slate-500"
                        }
                      >
                        {delta > 0 ? `+${(delta * 100).toFixed(1)}% (A)` : delta < 0 ? `+${(Math.abs(delta) * 100).toFixed(1)}% (B)` : "0.0%"}
                      </span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
