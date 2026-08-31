import { useState, useEffect, useRef, useCallback } from "react";
import { Citation, Source, RetrievalStageCounts } from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export interface SSEStreamState {
  text: string;
  sources: Source[];
  citations: Citation[];
  stageCounts: RetrievalStageCounts;
  isLoading: boolean;
  isStreaming: boolean;
  isComplete: boolean;
  error: string | null;
  latencyMs: number | null;
  modelUsed: string | null;
  runId: string | null;
}

export function useSSEStream(runId: string | null) {
  const [state, setState] = useState<SSEStreamState>({
    text: "",
    sources: [],
    citations: [],
    stageCounts: {},
    isLoading: false,
    isStreaming: false,
    isComplete: false,
    error: null,
    latencyMs: null,
    modelUsed: null,
    runId: null,
  });

  const eventSourceRef = useRef<EventSource | null>(null);

  const startStream = useCallback((activeRunId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setState({
      text: "",
      sources: [],
      citations: [],
      stageCounts: {},
      isLoading: true,
      isStreaming: true,
      isComplete: false,
      error: null,
      latencyMs: null,
      modelUsed: null,
      runId: activeRunId,
    });

    const streamUrl = `${API_BASE}/query/${activeRunId}/stream`;
    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setState((prev) => ({ ...prev, isLoading: false }));
    };

    es.onmessage = (event) => {
      try {
        if (!event.data) return;
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "retrieval_start":
            setState((prev) => ({ ...prev, isLoading: true }));
            break;

          case "retrieval_complete":
            setState((prev) => ({
              ...prev,
              isLoading: false,
              sources: data.sources || [],
              stageCounts: data.stage_counts || {},
            }));
            break;

          case "token":
            if (data.delta) {
              setState((prev) => ({
                ...prev,
                text: prev.text + data.delta,
                isLoading: false,
              }));
            }
            break;

          case "done":
            setState((prev) => ({
              ...prev,
              runId: data.run_id || prev.runId,
              text: data.generation || prev.text,
              citations: data.citations || [],
              sources: data.sources || prev.sources,
              stageCounts: data.stage_counts || prev.stageCounts,
              latencyMs: data.latency_ms || null,
              modelUsed: data.model_used || null,
              isStreaming: false,
              isComplete: true,
              isLoading: false,
            }));
            es.close();
            eventSourceRef.current = null;
            break;

          case "error":
            setState((prev) => ({
              ...prev,
              error: data.message || "An error occurred during generation",
              isStreaming: false,
              isLoading: false,
            }));
            es.close();
            eventSourceRef.current = null;
            break;

          case "keepalive":
            // keepalive ping
            break;

          default:
            break;
        }
      } catch (err) {
        console.error("Error parsing SSE frame:", err);
      }
    };

    es.onerror = (err) => {
      console.warn("EventSource error or stream closed:", err);
      setState((prev) => {
        // If we already have completed text, just finish
        if (prev.text && !prev.error) {
          return { ...prev, isStreaming: false, isComplete: true, isLoading: false };
        }
        return {
          ...prev,
          error: prev.text ? null : "Streaming connection failed or was closed.",
          isStreaming: false,
          isLoading: false,
        };
      });
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState((prev) => ({ ...prev, isStreaming: false, isLoading: false }));
  }, []);

  useEffect(() => {
    if (runId) {
      startStream(runId);
    }
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [runId, startStream]);

  return {
    ...state,
    startStream,
    stopStream,
  };
}
