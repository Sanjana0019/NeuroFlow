import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 60 * 1000, // 60s stale time: instant route switching using memory cache without loading spinners
      gcTime: 10 * 60 * 1000, // 10 minutes cache retention
    },
  },
});
