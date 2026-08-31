"use client";

import { usePathname } from "next/navigation";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "../lib/query-client";
import { Header } from "../components/common/Header";
import "./globals.css";

function LayoutContent({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLanding = pathname === "/" || pathname === "/about";

  return (
    <body className="min-h-screen bg-[#0b0f19] text-slate-100 antialiased flex flex-col selection:bg-indigo-600 selection:text-white">
      <QueryClientProvider client={queryClient}>
        <Header />
        {isLanding ? (
          <main className="flex-1 w-full">{children}</main>
        ) : (
          <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6">
            {children}
          </main>
        )}
      </QueryClientProvider>
    </body>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <LayoutContent>{children}</LayoutContent>
    </html>
  );
}
