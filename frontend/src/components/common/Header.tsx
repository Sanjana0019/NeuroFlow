"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Play, Workflow, Activity, FileText, Sparkles, Cpu } from "lucide-react";
import clsx from "clsx";

export function Header() {
  const pathname = usePathname();

  const navItems = [
    { name: "Query Playground", href: "/playground", icon: Play },
    { name: "Pipelines", href: "/pipelines", icon: Workflow },
    { name: "Live Evaluation Feed", href: "/evaluations", icon: Activity },
    { name: "Documents", href: "/documents", icon: FileText },
    { name: "Fine-Tuning", href: "/finetuning", icon: Sparkles },
  ];

  // Don't render dashboard header on landing page or about page
  if (pathname === "/" || pathname === "/about") return null;

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-[#0f172a]/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30">
              <Cpu className="h-5 w-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-semibold text-white tracking-tight text-base leading-none">
                NeuroFlow
              </span>
              <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                RAG Engine Dashboard
              </span>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={clsx(
                    "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
                    isActive
                      ? "bg-indigo-600/10 text-indigo-400 border border-indigo-500/30"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/about"
            className="text-xs text-slate-400 hover:text-slate-200 px-2.5 py-1 rounded-md hover:bg-slate-800/50 transition-colors"
          >
            About
          </Link>
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 text-xs font-mono">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            System Healthy
          </div>
        </div>
      </div>
    </header>
  );
}
