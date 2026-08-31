import { useState, useEffect, useRef } from "react";
import { Evaluation } from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export function useEvaluationStream(initialEvaluations: Evaluation[] = []) {
  const [evaluations, setEvaluations] = useState<Evaluation[]>(initialEvaluations);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (initialEvaluations && initialEvaluations.length > 0) {
      setEvaluations((prev) => {
        if (prev.length === 0) return initialEvaluations;
        const existingIds = new Set(prev.map((e) => e.run_id));
        const missing = initialEvaluations.filter((e) => !existingIds.has(e.run_id));
        if (missing.length === 0) return prev;
        return [...prev, ...missing];
      });
    }
  }, [initialEvaluations]);

  useEffect(() => {
    const streamUrl = `${API_BASE}/evaluations/stream`;
    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    es.onmessage = (event) => {
      try {
        if (!event.data) return;
        const data = JSON.parse(event.data);
        if (data.type === "keepalive") return;

        const newEval: Evaluation = {
          run_id: data.run_id,
          query: data.query,
          generation: data.generation,
          pipeline_name: data.pipeline_name || "Production RAG",
          faithfulness: data.faithfulness ?? null,
          answer_relevance: data.answer_relevance ?? null,
          context_precision: data.context_precision ?? null,
          context_recall: data.context_recall ?? null,
          overall_score: data.overall_score ?? null,
          judge_model: data.judge_model,
          evaluated_at: new Date().toISOString(),
          chunks: data.chunks || [],
        };

        setEvaluations((prev) => {
          // Avoid duplicate run IDs in live feed
          const exists = prev.some((e) => e.run_id === newEval.run_id);
          if (exists) {
            return prev.map((e) => (e.run_id === newEval.run_id ? newEval : e));
          }
          return [newEval, ...prev];
        });
      } catch (err) {
        console.error("Error parsing evaluation stream message:", err);
      }
    };

    es.onerror = () => {
      setIsConnected(false);
      setError("Evaluation feed disconnected. Reconnecting...");
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  return {
    evaluations,
    isConnected,
    error,
  };
}
