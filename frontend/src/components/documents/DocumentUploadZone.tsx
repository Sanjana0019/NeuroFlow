"use client";

import { useState, useRef } from "react";
import { UploadCloud, File, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { uploadDocument } from "../../lib/api";

interface DocumentUploadZoneProps {
  pipelineId?: string;
  onUploadSuccess: () => void;
}

interface UploadingFileState {
  file: File;
  progress: number;
  status: "uploading" | "success" | "error";
  errorMessage?: string;
}

export function DocumentUploadZone({ pipelineId, onUploadSuccess }: DocumentUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<UploadingFileState[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = async (newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    if (!fileArray.length) return;

    const initialStates: UploadingFileState[] = fileArray.map((f) => ({
      file: f,
      progress: 0,
      status: "uploading",
    }));

    setFiles((prev) => [...initialStates, ...prev]);

    for (let i = 0; i < fileArray.length; i++) {
      const targetFile = fileArray[i];
      try {
        await uploadDocument(targetFile, pipelineId, (percent) => {
          setFiles((prev) =>
            prev.map((item) =>
              item.file === targetFile ? { ...item, progress: percent } : item
            )
          );
        });

        setFiles((prev) =>
          prev.map((item) =>
            item.file === targetFile ? { ...item, status: "success", progress: 100 } : item
          )
        );
        onUploadSuccess();
      } catch (err: any) {
        const msg = err.response?.data?.detail || err.message || "Upload failed";
        setFiles((prev) =>
          prev.map((item) =>
            item.file === targetFile
              ? { ...item, status: "error", errorMessage: typeof msg === "string" ? msg : JSON.stringify(msg) }
              : item
          )
        );
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  return (
    <div className="space-y-4">
      {/* Drag & Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer ${
          isDragging
            ? "border-indigo-500 bg-indigo-950/20"
            : "border-slate-800 bg-slate-900/50 hover:border-slate-700 hover:bg-slate-900"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
          }}
        />
        <div className="p-3.5 rounded-2xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400 mb-3 shadow-lg shadow-indigo-600/10">
          <UploadCloud className="h-6 w-6" />
        </div>
        <h4 className="text-sm font-semibold text-slate-200 mb-1">
          Drag and drop files here, or click to browse
        </h4>
        <p className="text-xs text-slate-500 font-mono">
          Supports PDF, TXT, MD, DOCX, JSON, CSV (Max 100MB per file)
        </p>
      </div>

      {/* Upload Progress List */}
      {files.length > 0 && (
        <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
          {files.map((item, idx) => (
            <div
              key={`${item.file.name}-${idx}`}
              className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs font-mono"
            >
              <div className="flex items-center gap-2.5 truncate max-w-sm">
                <File className="h-4 w-4 text-indigo-400 shrink-0" />
                <span className="text-slate-200 font-medium truncate">{item.file.name}</span>
                <span className="text-slate-500 text-[11px]">
                  ({(item.file.size / 1024).toFixed(1)} KB)
                </span>
              </div>

              <div className="flex items-center gap-3">
                {item.status === "uploading" && (
                  <div className="flex items-center gap-2 text-indigo-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>{item.progress}%</span>
                  </div>
                )}
                {item.status === "success" && (
                  <span className="flex items-center gap-1 text-emerald-400">
                    <CheckCircle2 className="h-4 w-4" />
                    Uploaded
                  </span>
                )}
                {item.status === "error" && (
                  <span
                    className="flex items-center gap-1 text-rose-400 truncate max-w-[180px]"
                    title={item.errorMessage}
                  >
                    <AlertCircle className="h-4 w-4 shrink-0" />
                    {item.errorMessage || "Failed"}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
