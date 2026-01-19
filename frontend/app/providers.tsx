'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/stores';
import { GenerationStatusPanel } from '@/components/generation';

interface ProvidersProps {
  children: React.ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  const { checkApiHealth, sidebarCollapsed } = useAppStore();

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();

    // Re-check every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, [checkApiHealth]);

  // Update main content padding based on sidebar state
  useEffect(() => {
    const mainContent = document.querySelector('div.min-h-screen');
    if (mainContent) {
      if (sidebarCollapsed) {
        mainContent.classList.remove('pl-64');
        mainContent.classList.add('pl-16');
      } else {
        mainContent.classList.remove('pl-16');
        mainContent.classList.add('pl-64');
      }
    }
  }, [sidebarCollapsed]);

  return (
    <>
      {children}
      {/* Global status panel for active generation jobs */}
      <GenerationStatusPanel />
    </>
  );
}

export default Providers;
