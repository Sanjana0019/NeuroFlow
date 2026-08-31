"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X, FileText, Search, Hash, Loader2, Sparkles } from "lucide-react";
import { DocumentItem, ChunkItem } from "../../types";
import { fetchDocumentChunks, findSimilarDocumentChunks } from "../../lib/api";
import { StatusBadge } from "../common/Badge";

interface DocumentDetailDrawerProps {
  document: DocumentItem | null;
  onClose: () => void;
}

export function DocumentDetailDrawer({ document, onClose }: DocumentDetailDrawerProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[] | null>(null);

  const { data: chunks, isLoading } = useQuery<ChunkItem[]>({
    queryKey: ["document-chunks", document?.id],
    queryFn: () => (document ? fetchDocumentChunks(document.id) : []),
    enabled: !!document,
  });

  if (!document) return null;

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }
    setIsSearching(true);
    try {
      const results = await findSimilarDocumentChunks(document.id, searchQuery, 5);
      setSearchResults(results);
    } catch (err) {
      console.error("Failed to find similar chunks:", err);
    } finally {
      setIsSearching(false);
    }
  };

  const displayedChunks = searchResults || chunks || [];

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end animate-in fade-in duration-200">
      <div
        className="w-full max-w-2xl bg-slate-900 border-l border-slate-800 h-full p-6 flex flex-col shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5 truncate max-w-md">
            <div className="p-2 rounded-lg bg-indigo-950/80 border border-indigo-800/40 text-indigo-400 shrink-0">
              <FileText className="h-5 w-5" />
            </div>
            <div className="truncate">
              <h3 className="font-semibold text-slate-100 text-sm truncate">{document.filename}</h3>
              <p className="text-xs text-slate-400 font-mono truncate">{document.id}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Metadata section */}
        <div className="my-5 grid grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Status</span>
            <StatusBadge status={document.status} />
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Chunk Count</span>
            <span className="text-xs font-mono font-semibold text-slate-200">{document.chunk_count} chunks</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Source Type</span>
            <span className="text-xs font-mono font-medium text-slate-200 uppercase">{document.source_type}</span>
          </div>
        </div>

        {/* Semantic Similarity Search in Document Chunks */}
        <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 mb-6">
          <div className="flex items-center gap-1.5 mb-2.5">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wide">
              Find Similar Chunks (Vector Search)
            </h4>
          </div>
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search semantic concept in this document..."
                className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
            <button
              type="submit"
              disabled={isSearching}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-all shadow-md shadow-indigo-600/30 flex items-center gap-1.5 disabled:opacity-50"
            >
              {isSearching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Search"}
            </button>
            {searchResults && (
              <button
                type="button"
                onClick={() => {
                  setSearchResults(null);
                  setSearchQuery("");
                }}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors"
              >
                Reset
              </button>
            )}
          </form>
          {searchResults && (
            <p className="text-[11px] font-mono text-indigo-400 mt-2">
              Showing top {searchResults.length} semantically similar chunks for query: &quot;{searchQuery}&quot;
            </p>
          )}
        </div>

        {/* Chunks List */}
        <div className="flex-1 flex flex-col min-h-0">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center justify-between">
            <span>Document Chunks</span>
            <span className="text-[11px] font-mono text-slate-500">{displayedChunks.length} displayed</span>
          </h4>

          {isLoading ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
              <span className="text-xs">Loading chunks...</span>
            </div>
          ) : displayedChunks.length === 0 ? (
            <div className="p-8 text-center text-xs text-slate-500 font-mono bg-slate-950/40 rounded-xl border border-slate-800">
              No chunks available for this document.
            </div>
          ) : (
            <div className="space-y-3 overflow-y-auto pr-1 flex-1">
              {displayedChunks.map((chunk, idx) => (
                <div
                  key={chunk.id || idx}
                  className={`p-3.5 rounded-xl bg-slate-950/90 border transition-all text-xs font-mono ${
                    chunk.similarity_score
                      ? "border-indigo-500/60 shadow-lg shadow-indigo-950/40"
                      : "border-slate-800/80"
                  }`}
                >
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/60">
                    <span className="flex items-center gap-1 text-slate-400">
                      <Hash className="h-3.5 w-3.5 text-indigo-400" />
                      Chunk #{chunk.chunk_index}
                    </span>
                    {chunk.similarity_score !== undefined && (
                      <span className="px-2 py-0.5 rounded bg-indigo-950 border border-indigo-800 text-indigo-300 text-[10px] font-semibold">
                        Similarity: {(chunk.similarity_score * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{chunk.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-4 mt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition-colors"
          >
            Close Detail
          </button>
        </div>
      </div>
    </div>
  );
}
