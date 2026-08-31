"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  Node,
  Edge,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Search, Database, FileCode, Sliders, Layers, ArrowDownUp, CheckSquare } from "lucide-react";
import { RetrievalStageCounts } from "../../types";

interface RetrievalInspectorProps {
  query: string;
  stageCounts: RetrievalStageCounts;
  pipelineName?: string;
}

// Custom Node Component for Retrieval Stages
function StageNode({ data }: { data: any }) {
  const Icon = data.icon || Database;
  return (
    <div className="px-4 py-3 rounded-xl bg-slate-900 border border-slate-700 shadow-xl min-w-[200px] text-left transition-all hover:border-indigo-500">
      <Handle type="target" position={Position.Top} className="!bg-indigo-500 !w-2.5 !h-2.5" />
      <div className="flex items-center gap-2.5">
        <div className={`p-1.5 rounded-lg ${data.bgClass || "bg-indigo-950/80 text-indigo-400"} border border-slate-700/60`}>
          <Icon className="h-4 w-4" />
        </div>
        <div>
          <div className="text-xs font-semibold text-slate-200">{data.label}</div>
          <div className="text-[11px] font-mono text-indigo-400 font-medium mt-0.5">
            {data.count !== undefined ? `${data.count} chunks` : data.subLabel || "Ready"}
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-indigo-500 !w-2.5 !h-2.5" />
    </div>
  );
}

const nodeTypes = {
  stage: StageNode,
};

export function RetrievalInspector({ query, stageCounts, pipelineName = "Production-Hybrid-RAG" }: RetrievalInspectorProps) {
  const isDenseOnly = pipelineName.toLowerCase().includes("dense");

  const denseCount = stageCounts.dense ?? 1;
  const sparseCount = stageCounts.sparse ?? (isDenseOnly ? 0 : 1);
  const metadataCount = stageCounts.metadata ?? (isDenseOnly ? 0 : 1);
  const rrfCount = stageCounts.rrf ?? (isDenseOnly ? denseCount : 1);
  const rerankerCount = stageCounts.reranker ?? (isDenseOnly ? denseCount : 1);
  const finalContextCount = stageCounts.final_context ?? 1;

  const { nodes, edges } = useMemo(() => {
    if (isDenseOnly) {
      // 2-Stage Pure Dense Vector Search Graph
      const denseNodes: Node[] = [
        {
          id: "query",
          type: "stage",
          position: { x: 260, y: 30 },
          data: {
            label: "User Query",
            subLabel: query ? `"${query.slice(0, 30)}..."` : "Active Query",
            icon: Search,
            bgClass: "bg-blue-950/80 text-blue-400",
          },
        },
        {
          id: "dense",
          type: "stage",
          position: { x: 260, y: 170 },
          data: {
            label: "Dense Vector Search (halfvec)",
            count: denseCount,
            icon: Database,
            bgClass: "bg-purple-950/80 text-purple-400",
          },
        },
        {
          id: "final",
          type: "stage",
          position: { x: 260, y: 310 },
          data: {
            label: "Final Context Assembly",
            count: finalContextCount,
            icon: CheckSquare,
            bgClass: "bg-emerald-950/80 text-emerald-400",
          },
        },
      ];

      const denseEdges: Edge[] = [
        {
          id: "e-q-dense",
          source: "query",
          target: "dense",
          animated: true,
          style: { stroke: "#6366f1", strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#6366f1" },
        },
        {
          id: "e-dense-final",
          source: "dense",
          target: "final",
          animated: true,
          style: { stroke: "#10b981", strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#10b981" },
        },
      ];

      return { nodes: denseNodes, edges: denseEdges };
    }

    // 5-Stage Production Hybrid RAG Graph
    const hybridNodes: Node[] = [
      {
        id: "query",
        type: "stage",
        position: { x: 260, y: 20 },
        data: {
          label: "User Query",
          subLabel: query ? `"${query.slice(0, 30)}..."` : "Active Query",
          icon: Search,
          bgClass: "bg-blue-950/80 text-blue-400",
        },
      },
      {
        id: "dense",
        type: "stage",
        position: { x: 50, y: 140 },
        data: {
          label: "Dense Retrieval",
          count: denseCount,
          icon: Database,
          bgClass: "bg-purple-950/80 text-purple-400",
        },
      },
      {
        id: "sparse",
        type: "stage",
        position: { x: 260, y: 140 },
        data: {
          label: "Sparse (BM25)",
          count: sparseCount,
          icon: FileCode,
          bgClass: "bg-amber-950/80 text-amber-400",
        },
      },
      {
        id: "metadata",
        type: "stage",
        position: { x: 470, y: 140 },
        data: {
          label: "Metadata Filter",
          count: metadataCount,
          icon: Sliders,
          bgClass: "bg-emerald-950/80 text-emerald-400",
        },
      },
      {
        id: "rrf",
        type: "stage",
        position: { x: 260, y: 260 },
        data: {
          label: "RRF Fusion",
          count: rrfCount,
          icon: Layers,
          bgClass: "bg-cyan-950/80 text-cyan-400",
        },
      },
      {
        id: "reranker",
        type: "stage",
        position: { x: 260, y: 380 },
        data: {
          label: "Cross-Encoder Reranker",
          count: rerankerCount,
          icon: ArrowDownUp,
          bgClass: "bg-orange-950/80 text-orange-400",
        },
      },
      {
        id: "final",
        type: "stage",
        position: { x: 260, y: 500 },
        data: {
          label: "Final Assembled Context",
          count: finalContextCount,
          icon: CheckSquare,
          bgClass: "bg-emerald-950/80 text-emerald-400",
        },
      },
    ];

    const hybridEdges: Edge[] = [
      {
        id: "e-q-dense",
        source: "query",
        target: "dense",
        animated: true,
        style: { stroke: "#6366f1", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#6366f1" },
      },
      {
        id: "e-q-sparse",
        source: "query",
        target: "sparse",
        animated: true,
        style: { stroke: "#6366f1", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#6366f1" },
      },
      {
        id: "e-q-meta",
        source: "query",
        target: "metadata",
        animated: true,
        style: { stroke: "#6366f1", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#6366f1" },
      },
      {
        id: "e-dense-rrf",
        source: "dense",
        target: "rrf",
        style: { stroke: "#475569", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#475569" },
      },
      {
        id: "e-sparse-rrf",
        source: "sparse",
        target: "rrf",
        style: { stroke: "#475569", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#475569" },
      },
      {
        id: "e-meta-rrf",
        source: "metadata",
        target: "rrf",
        style: { stroke: "#475569", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#475569" },
      },
      {
        id: "e-rrf-reranker",
        source: "rrf",
        target: "reranker",
        style: { stroke: "#475569", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#475569" },
      },
      {
        id: "e-reranker-final",
        source: "reranker",
        target: "final",
        animated: true,
        style: { stroke: "#10b981", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#10b981" },
      },
    ];

    return { nodes: hybridNodes, edges: hybridEdges };
  }, [query, isDenseOnly, denseCount, sparseCount, metadataCount, rrfCount, rerankerCount, finalContextCount]);

  return (
    <div className="w-full h-[540px] rounded-xl bg-slate-950 border border-slate-800 relative overflow-hidden shadow-inner">
      <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
        <span className="px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-800/50 text-[11px] font-mono text-indigo-300">
          Architecture: {pipelineName}
        </span>
        <span className="text-[10px] font-mono text-slate-500">
          {isDenseOnly ? "2-Stage Vector Pipeline" : "5-Stage Hybrid RRF Pipeline"}
        </span>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.5}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={16} size={1} />
        <Controls className="!bg-slate-900 !border-slate-800 !text-slate-300" />
      </ReactFlow>
    </div>
  );
}
