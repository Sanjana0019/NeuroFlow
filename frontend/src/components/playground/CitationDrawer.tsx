import { X, FileText, Hash, CheckCircle2, AlertCircle } from "lucide-react";
import { Citation, Source } from "../../types";

interface CitationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  citation: Citation | null;
  source: Source | null;
}

export function CitationDrawer({ isOpen, onClose, citation, source }: CitationDrawerProps) {
  if (!isOpen) return null;

  const filename = citation?.document_name || source?.filename || "Document Chunk";
  const pageNumber = citation?.page_number ?? source?.page_number;
  const chunkId = citation?.chunk_id || source?.chunk_id;
  const content = source?.content || citation?.content_preview || "No full chunk content available.";
  const isInvalid = citation?.invalid_citation;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end animate-in fade-in duration-200">
      <div
        className="w-full max-w-xl bg-slate-900 border-l border-slate-800 h-full p-6 flex flex-col shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-950/80 border border-indigo-800/40 text-indigo-400">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-100 text-sm">{citation?.reference || "Citation Detail"}</h3>
              <p className="text-xs text-slate-400 truncate max-w-xs">{filename}</p>
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
        <div className="my-5 grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Document</span>
            <span className="text-xs font-mono font-medium text-slate-200 truncate block" title={filename}>
              {filename}
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Page</span>
            <span className="text-xs font-mono font-medium text-slate-200">
              {pageNumber !== null && pageNumber !== undefined ? `Page ${pageNumber}` : "N/A"}
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Citation Status</span>
            {isInvalid ? (
              <span className="flex items-center text-xs font-medium text-rose-400">
                <AlertCircle className="h-3.5 w-3.5 mr-1" />
                Unverified
              </span>
            ) : (
              <span className="flex items-center text-xs font-medium text-emerald-400">
                <CheckCircle2 className="h-3.5 w-3.5 mr-1" />
                Ground Truth
              </span>
            )}
          </div>
        </div>

        {/* Chunk Content */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Hash className="h-3.5 w-3.5 text-indigo-400" />
              Retrieved Context Chunk
            </h4>
            {chunkId && <span className="text-[10px] font-mono text-slate-500 truncate max-w-[150px]">{chunkId}</span>}
          </div>
          <div className="flex-1 p-4 rounded-xl bg-slate-950/90 border border-slate-800 font-mono text-xs text-slate-300 leading-relaxed overflow-y-auto whitespace-pre-wrap selection:bg-indigo-600/30">
            {content}
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 mt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition-colors"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
}
