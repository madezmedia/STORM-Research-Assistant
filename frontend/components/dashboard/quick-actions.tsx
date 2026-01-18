'use client';

import Link from 'next/link';
import { PenTool, Globe, Presentation, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface QuickActionProps {
  title: string;
  description: string;
  href: string;
  icon: React.ElementType;
  gradient: string;
}

function QuickAction({ title, description, href, icon: Icon, gradient }: QuickActionProps) {
  return (
    <Link
      href={href}
      className={cn(
        'group relative overflow-hidden rounded-xl p-6 transition-all duration-300',
        'bg-gradient-to-br hover:scale-[1.02] hover:shadow-lg',
        gradient
      )}
    >
      <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
      <div className="relative z-10">
        <div className="mb-4 inline-flex rounded-lg bg-white/20 p-3 backdrop-blur-sm">
          <Icon className="h-6 w-6 text-white" />
        </div>
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <p className="mt-1 text-sm text-white/80">{description}</p>
      </div>
      <Sparkles className="absolute right-4 top-4 h-5 w-5 text-white/30 group-hover:text-white/60 transition-colors" />
    </Link>
  );
}

export function QuickActions() {
  const actions: QuickActionProps[] = [
    {
      title: 'Create Content',
      description: 'Generate research-backed articles with AI',
      href: '/studio',
      icon: PenTool,
      gradient: 'from-blue-600 to-blue-800',
    },
    {
      title: 'GEO Analysis',
      description: 'Analyze websites for AI optimization',
      href: '/geo',
      icon: Globe,
      gradient: 'from-purple-600 to-purple-800',
    },
    {
      title: 'Create Slideshow',
      description: 'Transform content into presentations',
      href: '/slides',
      icon: Presentation,
      gradient: 'from-emerald-600 to-emerald-800',
    },
  ];

  return (
    <div className="glass rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Quick Actions</h2>
      <div className="grid gap-4 md:grid-cols-3">
        {actions.map((action) => (
          <QuickAction key={action.title} {...action} />
        ))}
      </div>
    </div>
  );
}

export default QuickActions;
