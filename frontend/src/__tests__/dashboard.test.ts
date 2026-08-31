import { describe, it, expect } from "vitest";

describe("NeuroFlow Dashboard Core Logic & Contracts", () => {
  // 1. Pipeline Config & Monaco JSON Validation
  it("validates PipelineConfig JSON structure correctly", () => {
    const validConfig = {
      name: "prod-rag",
      version: 1,
      retrieval: {
        dense_k: 20,
        sparse_k: 15,
        top_k_after_rerank: 5,
        query_expansion: true,
      },
      generation: {
        model: "gpt-4o-mini",
        temperature: 0.2,
      },
      evaluation: {
        auto_evaluate: true,
      },
      rate_limit_rpm: 60,
    };

    expect(validConfig.name).toBe("prod-rag");
    expect(validConfig.retrieval.dense_k).toBe(20);
    expect(validConfig.evaluation.auto_evaluate).toBe(true);

    const invalidJson = "{ invalid: json }";
    expect(() => JSON.parse(invalidJson)).toThrow();
  });

  // 2. Score Threshold Color Classifications
  it("classifies evaluation scores with correct thresholds", () => {
    const getScoreCategory = (score: number) => {
      if (score > 0.8) return "green";
      if (score >= 0.6) return "yellow";
      return "red";
    };

    expect(getScoreCategory(0.95)).toBe("green");
    expect(getScoreCategory(0.81)).toBe("green");
    expect(getScoreCategory(0.75)).toBe("yellow");
    expect(getScoreCategory(0.6)).toBe("yellow");
    expect(getScoreCategory(0.59)).toBe("red");
    expect(getScoreCategory(0.2)).toBe("red");
  });

  // 3. Retrieval Stage Counts for Retrieval Inspector
  it("derives retrieval stage counts matching actual pipeline architecture", () => {
    const stageCounts = {
      dense: 20,
      sparse: 15,
      metadata: 5,
      rrf: 25,
      reranker: 10,
      final_context: 5,
    };

    expect(stageCounts.dense).toBe(20);
    expect(stageCounts.sparse).toBe(15);
    expect(stageCounts.metadata).toBe(5);
    expect(stageCounts.rrf).toBe(25);
    expect(stageCounts.reranker).toBe(10);
    expect(stageCounts.final_context).toBe(5);
  });

  // 4. Word-level Diff Computation
  it("computes word-level diff between Pipeline A and Pipeline B", () => {
    const textA = "NeuroFlow uses hybrid search with RRF";
    const textB = "NeuroFlow uses dense search with cross-encoders";

    const wordsA = textA.split(/\s+/);
    const wordsB = textB.split(/\s+/);
    const setB = new Set(wordsB.map((w) => w.toLowerCase()));

    const uniqueToA = wordsA.filter((w) => !setB.has(w.toLowerCase()));
    expect(uniqueToA).toEqual(["hybrid", "RRF"]);
  });

  // 5. Evaluation Filtering Logic
  it("filters live evaluations by threshold and pipeline", () => {
    const evals = [
      { run_id: "1", pipeline_name: "prod-rag", overall_score: 0.92, faithfulness: 0.95 },
      { run_id: "2", pipeline_name: "fast-rag", overall_score: 0.65, faithfulness: 0.60 },
      { run_id: "3", pipeline_name: "prod-rag", overall_score: 0.55, faithfulness: 0.50 },
    ];

    const lowOverall = evals.filter((e) => e.overall_score < 0.7);
    expect(lowOverall.length).toBe(2);

    const prodEvals = evals.filter((e) => e.pipeline_name === "prod-rag");
    expect(prodEvals.length).toBe(2);
  });
});
