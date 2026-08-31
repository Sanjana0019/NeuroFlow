"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileText, Upload, Clock, Hash, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import { fetchDocuments } from "../../lib/api";
import { DocumentItem } from "../../types";
import { StatusBadge } from "../../components/common/Badge";
import { DocumentUploadZone } from "../../components/documents/DocumentUploadZone";
import { DocumentDetailDrawer } from "../../components/documents/DocumentDetailDrawer";

export default function DocumentsPage() {
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);

  const {
    data: documents = [],
    isLoading,
    refetch,
  } = useQuery<DocumentItem[]>({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && data.some((d: DocumentItem) => d.status === "processing" || d.status === "pending")) {
        return 3000;
      }
      return false;
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <FileText className="h-5 w-5 text-indigo-500" />
            Document Manager
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Ingest unstructured documents, track processing states, and inspect vector chunks
          </p>
        </div>
      </div>

      {/* Upload Zone */}
      <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-3">
        <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <Upload className="h-4 w-4 text-indigo-400" />
          Ingest New Documents
        </h3>
        <DocumentUploadZone onUploadSuccess={() => refetch()} />
      </div>

      {/* Documents Table */}
      <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Indexed Documents ({documents.length})
          </h3>
          <span className="text-[11px] font-mono text-slate-500">Auto-refreshing status</span>
        </div>

        {isLoading ? (
          <div className="py-12 flex flex-col items-center justify-center gap-2 text-slate-400">
            <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
            <span className="text-xs font-mono">Loading documents...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="py-12 text-center text-xs text-slate-500 font-mono">
            No documents ingested yet. Upload files above to begin indexing.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/40">
                  <th className="px-4 py-3 font-medium">Filename</th>
                  <th className="px-4 py-3 font-medium">Source Type</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium text-right">Chunks</th>
                  <th className="px-4 py-3 font-medium text-right">Ingested At</th>
                  <th className="px-4 py-3 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {documents.map((doc) => (
                  <tr
                    key={doc.id}
                    onClick={() => setSelectedDoc(doc)}
                    className="hover:bg-slate-800/40 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3 font-medium text-slate-100 flex items-center gap-2 truncate max-w-xs">
                      <FileText className="h-4 w-4 text-indigo-400 shrink-0" />
                      <span className="truncate">{doc.filename}</span>
                    </td>
                    <td className="px-4 py-3 uppercase text-[11px] text-slate-400">{doc.source_type}</td>
                    <td className="px-4 py-3">
                      <StatusBadge status={doc.status} />
                    </td>
                    <td className="px-4 py-3 text-right font-semibold text-slate-200">{doc.chunk_count}</td>
                    <td className="px-4 py-3 text-right text-[11px] text-slate-400">
                      {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : "—"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedDoc(doc);
                        }}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-indigo-300 text-[11px] font-medium transition-colors"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Document Detail Drawer */}
      <DocumentDetailDrawer
        document={selectedDoc}
        onClose={() => setSelectedDoc(null)}
      />
    </div>
  );
}
