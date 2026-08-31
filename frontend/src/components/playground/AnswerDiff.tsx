import { useMemo } from "react";

interface AnswerDiffProps {
  textA: string;
  textB: string;
  nameA?: string;
  nameB?: string;
}

export function AnswerDiff({ textA, textB, nameA = "Pipeline A", nameB = "Pipeline B" }: AnswerDiffProps) {
  const { tokensA, tokensB } = useMemo(() => {
    const wordsA = textA.split(/\s+/).filter(Boolean);
    const wordsB = textB.split(/\s+/).filter(Boolean);
    const setA = new Set(wordsA.map((w) => w.toLowerCase()));
    const setB = new Set(wordsB.map((w) => w.toLowerCase()));

    const renderedA = wordsA.map((w, idx) => ({
      word: w,
      isUnique: !setB.has(w.toLowerCase()),
      key: `a-${idx}-${w}`,
    }));

    const renderedB = wordsB.map((w, idx) => ({
      word: w,
      isUnique: !setA.has(w.toLowerCase()),
      key: `b-${idx}-${w}`,
    }));

    return { tokensA: renderedA, tokensB: renderedB };
  }, [textA, textB]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono leading-relaxed">
      {/* Pipeline A panel */}
      <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800/80">
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
          <span className="font-semibold text-indigo-400">{nameA}</span>
          <span className="text-[10px] text-slate-500 font-sans">Highlighted words unique to A</span>
        </div>
        <p className="text-slate-300">
          {tokensA.map((item) => (
            <span
              key={item.key}
              className={
                item.isUnique
                  ? "bg-indigo-950/80 text-indigo-300 px-1 py-0.5 rounded border border-indigo-800/50 mr-1 inline-block"
                  : "mr-1 inline-block"
              }
            >
              {item.word}
            </span>
          ))}
        </p>
      </div>

      {/* Pipeline B panel */}
      <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800/80">
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
          <span className="font-semibold text-purple-400">{nameB}</span>
          <span className="text-[10px] text-slate-500 font-sans">Highlighted words unique to B</span>
        </div>
        <p className="text-slate-300">
          {tokensB.map((item) => (
            <span
              key={item.key}
              className={
                item.isUnique
                  ? "bg-purple-950/80 text-purple-300 px-1 py-0.5 rounded border border-purple-800/50 mr-1 inline-block"
                  : "mr-1 inline-block"
              }
            >
              {item.word}
            </span>
          ))}
        </p>
      </div>
    </div>
  );
}
