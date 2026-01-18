'use client';

import { FileText, Globe, Presentation, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon: React.ElementType;
  trend?: { value: number; positive: boolean };
  className?: string;
}

function StatCard({ title, value, description, icon: Icon, trend, className }: StatCardProps) {
  return (
    <div className={cn(
      'glass rounded-xl p-6 transition-all duration-300 hover:glow-sm',
      className
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="mt-2 text-3xl font-bold text-foreground">{value}</p>
          {description && (
            <p className="mt-1 text-xs text-muted-foreground">{description}</p>
          )}
          {trend && (
            <div className={cn(
              'mt-2 flex items-center gap-1 text-xs font-medium',
              trend.positive ? 'text-success' : 'text-destructive'
            )}>
              <TrendingUp className={cn('h-3 w-3', !trend.positive && 'rotate-180')} />
              {trend.value}% from last week
            </div>
          )}
        </div>
        <div className="rounded-lg bg-primary/10 p-3">
          <Icon className="h-6 w-6 text-primary" />
        </div>
      </div>
    </div>
  );
}

interface StatsGridProps {
  briefs: number;
  geoAnalyses: number;
  slideshows: number;
  totalContent: number;
}

export function StatsGrid({ briefs, geoAnalyses, slideshows, totalContent }: StatsGridProps) {
  const stats = [
    {
      title: 'Content Briefs',
      value: briefs,
      description: 'Active briefs',
      icon: FileText,
    },
    {
      title: 'GEO Analyses',
      value: geoAnalyses,
      description: 'Website analyses',
      icon: Globe,
    },
    {
      title: 'Slideshows',
      value: slideshows,
      description: 'Generated presentations',
      icon: Presentation,
    },
    {
      title: 'Total Content',
      value: totalContent,
      description: 'Articles generated',
      icon: TrendingUp,
    },
  ];

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <StatCard key={stat.title} {...stat} />
      ))}
    </div>
  );
}

export default StatsGrid;
