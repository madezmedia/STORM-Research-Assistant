'use client';

import { useEffect, useState } from 'react';
import { PageHeader } from '@/components/layout';
import { StatsGrid, RecentBriefs, QuickActions, SystemStatus } from '@/components/dashboard';
import { useBriefStore } from '@/stores';
import { Brief } from '@/types/api';

export default function DashboardPage() {
  const { briefs, fetchBriefs, isLoading } = useBriefStore();
  const [stats, setStats] = useState({
    briefs: 0,
    geoAnalyses: 0,
    slideshows: 0,
    totalContent: 0,
  });

  useEffect(() => {
    fetchBriefs();
  }, [fetchBriefs]);

  useEffect(() => {
    // Calculate stats from briefs
    const completedBriefs = briefs.filter((b: Brief) => b.status === 'complete');
    setStats({
      briefs: briefs.length,
      geoAnalyses: 0, // Will be populated when GEO store is implemented
      slideshows: 0, // Will be populated when slides store is implemented
      totalContent: completedBriefs.length,
    });
  }, [briefs]);

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Dashboard"
        description="Welcome to STORM Research Assistant"
      />

      <div className="space-y-8">
        {/* Stats Overview */}
        <StatsGrid {...stats} />

        {/* Quick Actions */}
        <QuickActions />

        {/* Two Column Layout */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Recent Briefs */}
          <RecentBriefs briefs={briefs} isLoading={isLoading} />

          {/* System Status */}
          <SystemStatus />
        </div>
      </div>
    </div>
  );
}
